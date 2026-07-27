from __future__ import annotations

from aiida import orm
from aiida.engine import append_, calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from itertools import product

ChildWorkChain = WorkflowFactory("vasp.v2.vasp")


class VaspMagmomWorkChain(WorkChain):
    """Sweep over a list of ``magmom_mapping`` values and gather the
    resulting magnetism outputs.

    For each entry in ``magmom_list`` (a ``dict`` like ``{"Si": 1.0}`` or
    ``{"Si": [1.0, -1.0]}``), one child VaspWorkChain is submitted with that
    ``magmom_mapping`` set on its inputs.

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
            include=[
                "code",
                "structure",
                "kpoints",
                "kpoints_spacing",
                "parameters",
                "potential_family",
                "potential_mapping",
                "calc",
            ],
        )

        spec.input(
            "magmom_list",
            valid_type=orm.List,
            help=(
                "List of ``magmom_mapping`` dictionaries, one per child "
                "calculation. Each entry maps element symbol to its "
                "magnetization, e.g. {'Si': 1.0} or {'Si': [1.0, -1.0]}."
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
        """Submit one child VaspWorkChain per ``magmom_mapping`` entry."""
        magmom_list = self.inputs.magmom_list.get_list()

        base_inputs = self.exposed_inputs(ChildWorkChain, agglomerate=True)

        for idx, mag_value in enumerate(magmom_list):
            mag_label = _magmom_to_label(mag_value)
            label = f"magmom_{idx:03d}_{mag_label}"

            child_inputs = dict(base_inputs)
            child_inputs["magmom_mapping"] = orm.Dict(mag_value)
            child_inputs["metadata"] = {"label": label}
            child_inputs["calc"] = {
                **base_inputs.get("calc", {}),
            }

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
            mag_label = _magmom_to_label(mag_value)
            label = f"magmom_{idx:03d}_{mag_label}"
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
    """Parse the magnetism outputs of each child VASP calculation and gather them.

    The ``misc`` output of a VASP calculation exposes ``magnetization`` (total
    magnetization) and ``site_magnetization`` (per-site magnetization).
    """
    from aiida.orm import load_node

    magnetization = {}
    site_magnetization = {}
    final_energy = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        if not child.is_finished_ok:
            continue

        try:
            misc = child.outputs.misc.get_dict()
        except (AttributeError, KeyError):
            continue

        magnetization[pk] = misc.get("magnetization")
        site_magnetization[pk] = misc.get("site_magnetization")
        total_energies = misc.get("total_energies") or {}
        final_energy[pk] = total_energies.get("energy_extrapolated")

    result = {
        "magnetization": magnetization,
        "site_magnetization": site_magnetization,
        "final_energy": final_energy,
    }
    return orm.Dict(result)


def _magmom_to_label(mag_value):
    """Render a ``magmom_mapping`` dict as a filesystem-friendly label."""
    parts = []
    for element, value in mag_value.items():
        if isinstance(value, (list, tuple)):
            v = "_".join(f"{float(x):g}" for x in value)
        else:
            v = f"{float(value):g}"
        parts.append(f"{element}_{v}")
    return "__".join(parts).replace(".", "_").replace("-", "m")