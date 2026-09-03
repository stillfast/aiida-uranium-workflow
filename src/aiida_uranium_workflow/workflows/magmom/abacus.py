from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from aiida_uranium_workflow.utils.labels import format_magmom_label

ChildWorkChain = WorkflowFactory("abacus.base")


class AbacusMagmomWorkChain(WorkChain):
    """Sweep over a list of initial magnetic configurations and gather the
    resulting magnetism outputs.

    For each ``magmom`` value (a list of per-atom magnetizations, e.g.
    ``[[1.0], [1.0]]``), one child ``AbacusBaseWorkChain`` is submitted with
    that ``stru.mag`` set in the ABACUS input parameters.

    Outline:  submit_children → gather_results

    Exit codes
    ----------
    * 0   ``SUCCESS``         — sweep completed normally.
    * 300 ``ERROR_CHILD``     — a child calculation failed.
    * 305 ``ERROR_PARSER``    — failed to parse magnetism outputs.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=(
                "abacus",
                "kpoints_distance",
                "pseudo_family",
                "abacus.metadata",
                "kpoints",
            ),
        )

        spec.input(
            "magmom_list",
            valid_type=orm.List,
            help=(
                "List of per-atom initial magnetization values for ABACUS. "
                "Each entry is a nested list, e.g. [[1.0], [-1.0]], matching "
                "the ``stru.mag`` schema in ABACUS."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "magmom scan completed successfully.")
        spec.exit_code(300, "ERROR_CHILD", "A child calculation failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse magnetism.")

    def submit_children(self):
        """Submit one child WorkChain per ``magmom`` configuration."""
        magmom_list = self.inputs.magmom_list.get_list()

        abacus_block = self.inputs.abacus

        if "parameters" not in abacus_block:
            self.report(
                f"no parameters under self.inputs.abacus; "
                f"available keys: {sorted(abacus_block.keys())}"
            )
            return self.exit_codes.ERROR_PARSER

        for idx, mag_value in enumerate(magmom_list):
            label = format_magmom_label(mag_value, index=idx)

            param_dict = abacus_block.parameters.get_dict()
            stru = param_dict.setdefault("stru", {})
            stru["mag"] = mag_value

            child_inputs = {
                "abacus": {
                    **abacus_block,
                    "parameters": orm.Dict(param_dict),
                },
                "metadata": {"label": label},
                "pseudo_family": self.inputs.pseudo_family,
            }

            if "kpoints" in self.inputs:
                child_inputs["kpoints"] = self.inputs.kpoints
            if "kpoints_distance" in self.inputs:
                child_inputs["kpoints_distance"] = self.inputs.kpoints_distance

            try:
                running = self.submit(ChildWorkChain, **child_inputs)
            except Exception as exc:
                self.report(
                    f"submission failed for magmom={mag_value}: {exc}"
                )
                raise

            self.report(
                f"submitted child pk={running.pk} for magmom={mag_value}"
            )
            self.to_context(**{label: running})

    def gather_results(self):
        magmom_list = self.inputs.magmom_list.get_list()

        all_finished_ok = True
        child_pks = []

        for idx, mag_value in enumerate(magmom_list):
            label = format_magmom_label(mag_value, index=idx)
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_magmom_results(
            child_pks=orm.List(child_pks),
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_magmom_results(child_pks):
    """Gather magmom child results into the report-side schema.

    Each child is parsed by :class:`AbacusChildParser` into a
    :class:`ChildRecord` (pk / exit status / energy eV / time / scf
    steps / atoms + ``final_magnetism`` etc. under ``data``), and the
    records are stored under the ``gather_schema`` layout defined in
    :mod:`utils.report.schema` — the same layout the magmom report
    consumes.

    If the new-schema path fails (unexpected child shape), falls back to
    the legacy pk-keyed layout and logs a warning — the report layer
    then renders the legacy dict through its own fallback.
    """
    import logging

    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parsers.child import AbacusChildParser
    from aiida_uranium_workflow.utils.report.schema import GatherResult

    logger = logging.getLogger(__name__)
    pks = child_pks.get_list()
    try:
        children = [
            AbacusChildParser().parse(load_node(pk)) for pk in pks
        ]
        return orm.Dict(
            GatherResult(backend="abacus", children=children).to_dict()
        )
    except Exception as exc:  # noqa: BLE001 — fall back to legacy layout
        logger.warning(
            "magmom gather (abacus): new-schema parse failed (%s); "
            "falling back to legacy layout",
            exc,
        )
        return _gather_magmom_legacy(child_pks)


def _gather_magmom_legacy(child_pks) -> orm.Dict:
    """Legacy pk-keyed gather layout (pre-schema WorkChain nodes)."""
    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parsers import fetch_summary

    magnetism = {}
    final_magnetism = {}
    nspin = {}
    final_energy = {}
    wall_time_seconds = {}
    scf_steps = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        # Capture exit status for every submitted child, even those
        # that did not finish OK. ``exit_status`` is ``None`` for
        # unfinished processes; we render that as ``-1``.
        status[pk] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )

        if not child.is_finished_ok:
            continue

        try:
            misc = child.outputs.misc.get_dict()
        except (AttributeError, KeyError):
            continue

        parameters = child.inputs.abacus.parameters.get_dict()
        nspin_value = parameters.get("input", {}).get("nspin")
        nspin[pk] = nspin_value

        magnetism[pk] = misc.get("magnetism")
        final_magnetism[pk] = misc.get("final_magnetism")

        # Unified parser entry (smear / convergence / magmom).
        summary = fetch_summary(child, "abacus")
        final_energy[pk] = summary["energy_ev"]
        wall_time_seconds[pk] = summary["time_s"]
        scf_steps[pk] = summary["scf_steps"]

    result = {
        "magnetism": magnetism,
        "final_magnetism": final_magnetism,
        "nspin": nspin,
        "final_energy": final_energy,
        "wall_time_seconds": wall_time_seconds,
        "scf_steps": scf_steps,
        "status": status,
    }
    return orm.Dict(result)
