"""QE (Quantum ESPRESSO) magmom WorkChain.

Sweeps over a list of initial magnetic configurations (per-species
``starting_magnetization`` values, e.g. ``{"U": 0.0}`` / ``{"U": 4.0}``)
and gathers the resulting magnetism from each ``PwBaseWorkChain`` child
(collinear, nspin=2 — non-collinear/SOC configurations are seeded via
``starting_magnetization`` + ``angle1``/``angle2`` in the SCF preset).

Outline:  submit_children → gather_results

Exit codes
----------
* 0   ``SUCCESS``     — sweep completed normally.
* 300 ``ERROR_CHILD`` — a child calculation failed.
* 305 ``ERROR_PARSER``— failed to parse magnetism outputs.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from aiida_uranium_workflow.utils.labels import format_magmom_label

ChildWorkChain = WorkflowFactory("quantumespresso.pw.base")


class QeMagmomWorkChain(WorkChain):
    """Sweep over initial magnetizations and gather QE magnetism outputs."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=[
                "pw",
                "kpoints",
                "kpoints_distance",
            ],
        )

        spec.input(
            "magmom_list",
            valid_type=orm.List,
            required=True,
            help=(
                "List of per-species initial magnetizations, one entry "
                "per child calculation, e.g. [{'U': 0.0}, {'U': 4.0}]. "
                "Each entry is injected as QE SYSTEM.starting_magnetization."
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
        """Submit one PwBaseWorkChain per magmom configuration."""
        from copy import deepcopy

        base_inputs = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        pw_base = dict(base_inputs.get("pw", {}))

        for idx, mag_value in enumerate(self.inputs.magmom_list.get_list()):
            label = format_magmom_label(mag_value, index=idx)

            params = deepcopy(pw_base["parameters"].get_dict())
            system = params.setdefault("SYSTEM", {})
            system["starting_magnetization"] = dict(mag_value)
            system["nspin"] = 2  # collinear sweep (SOC handled in preset)

            child_inputs = {
                **base_inputs,
                "pw": {
                    **pw_base,
                    "parameters": orm.Dict(dict=params),
                },
                "metadata": {"label": label},
            }
            running = self.submit(ChildWorkChain, **child_inputs)
            self.report(
                f"submitted child pk={running.pk} for magmom={mag_value}"
            )
            self.to_context(**{label: running})

    def gather_results(self):
        """Parse the QE magnetism outputs of each child and package them."""
        labels = [
            format_magmom_label(v, index=idx)
            for idx, v in enumerate(self.inputs.magmom_list.get_list())
        ]
        all_finished_ok = all(
            (getattr(self.ctx, label, None) is not None)
            and getattr(self.ctx, label).is_finished_ok
            for label in labels
        )

        results_node = parse_and_gather_qe_magmom_results(
            child_pks=orm.List(list=[
                getattr(self.ctx, label).pk
                for label in labels
                if getattr(self.ctx, label, None) is not None
            ]),
        )
        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_qe_magmom_results(child_pks):
    """Gather QE magmom child results into the report-side schema.

    Each child is parsed by :class:`QePwChildParser` into a
    :class:`ChildRecord` (pk / exit status / energy eV / time / atoms +
    per-species moments under ``data``), stored under the
    ``gather_schema`` layout defined in :mod:`utils.report.schema`.

    If the new-schema path fails, falls back to the legacy pk-keyed
    layout and logs a warning.
    """
    import logging

    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parsers.child import QePwChildParser
    from aiida_uranium_workflow.utils.report.schema import GatherResult

    logger = logging.getLogger(__name__)
    pks = child_pks.get_list()
    try:
        children = [QePwChildParser().parse(load_node(pk)) for pk in pks]
        return orm.Dict(
            GatherResult(backend="qe", children=children).to_dict()
        )
    except Exception as exc:  # noqa: BLE001 — fall back to legacy layout
        logger.warning(
            "magmom gather (qe): new-schema parse failed (%s); "
            "falling back to legacy layout",
            exc,
        )
        return _gather_qe_magmom_legacy(child_pks)


def _gather_qe_magmom_legacy(child_pks) -> orm.Dict:
    """Legacy pk-keyed gather layout (pre-schema WorkChain nodes)."""
    from aiida.orm import load_node

    magnetization = {}
    absolute_magnetization = {}
    atomic_moments = {}
    final_energy = {}
    wall_time_seconds = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        status[pk] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )
        if not child.is_finished_ok:
            continue

        try:
            para = child.outputs.output_parameters.get_dict()
        except (AttributeError, KeyError):
            continue

        magnetization[pk] = para.get("total_magnetization")
        absolute_magnetization[pk] = para.get("absolute_magnetization")
        atomic_moments[pk] = para.get("atomic_magnetic_moments")
        final_energy[pk] = para.get("energy")

        try:
            wall_time_seconds[pk] = float(para.get("wall_time_seconds"))
        except (TypeError, ValueError):
            pass

    return orm.Dict(dict={
        "magnetization": magnetization,
        "absolute_magnetization": absolute_magnetization,
        "atomic_magnetic_moments": atomic_moments,
        "final_energy": final_energy,
        "wall_time_seconds": wall_time_seconds,
        "status": status,
    })
