from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from itertools import product
from aiida_uranium_workflow.utils.labels import format_abacus_convergence_label

ChildWorkChain = WorkflowFactory("abacus.base")


class AbacusConvergenceWorkChain(WorkChain):
    """Sweep over ``(ecutwfc, kpoints_distance)`` combinations and gather the
    electronic entropy.

    Outline:  submit_children → gather_results

    Exit codes
    ----------
    * 0   ``SUCCESS``              — sweep completed normally.
    * 300 ``ERROR_CHILD``          — a child calculation failed.
    * 305 ``ERROR_PARSER``         — failed to parse electronic entropy.
      mode; the outline ran to ``submit_children`` and the planned
      ``(ecutwfc, kpoints_distance)`` sweep was recorded, but no CalcJob was
      launched.
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
            "ecutwfc_list",
            valid_type=orm.List,
            help="List of ABACUS ecutwfc values in Ry.",
        )
        spec.input(
            "kpoints_distance_list",
            valid_type=orm.List,
            required=False,
            help="List of kpoints_distance values in A^-1.",
        )
        spec.input(
            "kpoints_list",
            valid_type=orm.List,
            required=False,
            help="List of kpoints values for ABACUS.",
        )
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "smear scan completed successfully.")
        spec.exit_code(300, "ERROR_CHILD", "A child calculation failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse electronic entropy.")

    def submit_children(self):
        """Submit one child WorkChain per (ecutwfc, kpoints) pair.

        Supports two modes:
        - kpoints_distance_list: List of kpoints distance values (A^-1)
        - kpoints_list: List of kpoints mesh tuples, e.g. [(2,2,2), (4,4,4)]

        Only one of the two inputs should be provided.
        """
        ecutwfc_list = self.inputs.ecutwfc_list.get_list()

        kpoints_distance_list = getattr(self.inputs, "kpoints_distance_list", None)
        kpoints_list = getattr(self.inputs, "kpoints_list", None)

        use_kpoints_distance = kpoints_distance_list is not None
        use_kpoints_mesh = kpoints_list is not None

        if use_kpoints_distance:
            kpoints_values = kpoints_distance_list.get_list()
        elif use_kpoints_mesh:
            kpoints_values = kpoints_list.get_list()
        else:
            raise ValueError(
                "Either kpoints_distance_list or kpoints_list must be provided"
            )

        planned = list(product(ecutwfc_list, kpoints_values))
        self.ctx.convergence_pairs = planned

        for ecutwfc, kpoints_val in planned:
            abacus_block = self.inputs.abacus

            if "parameters" not in abacus_block:
                self.report(
                    f"no parameters under self.inputs.abacus; "
                    f"available keys: {sorted(abacus_block.keys())}"
                )
                continue

            if use_kpoints_distance:
                label = format_abacus_convergence_label(ecutwfc, kpoints_val)
            else:
                label = format_abacus_convergence_label(ecutwfc, kpoints_val)

            param_dict = abacus_block.parameters.get_dict()
            param_dict.setdefault("input", {})["ecutwfc"] = ecutwfc

            child_inputs = {
                "abacus": {
                    **abacus_block,
                    "parameters": orm.Dict(param_dict),
                },
                "metadata": {"label": label},
                "pseudo_family": self.inputs.pseudo_family,
            }

            if use_kpoints_distance:
                child_inputs["kpoints_distance"] = kpoints_val
            else:
                from aiida.plugins import DataFactory

                KpointsData = DataFactory("core.array.kpoints")
                kpoints_mesh = KpointsData()
                kpoints_mesh.set_kpoints_mesh(list(kpoints_val))
                child_inputs["kpoints"] = kpoints_mesh

            try:
                running = self.submit(ChildWorkChain, **child_inputs)
            except Exception as exc:
                self.report(
                    f"submission failed for ecutwfc={ecutwfc}, kpoints={kpoints_val}: {exc}"
                )
                raise

            self.report(
                f"submitted child pk={running.pk} for ecutwfc={ecutwfc}, kpoints={kpoints_val}"
            )
            self.to_context(**{label: running})

    def gather_results(self):
        ecutwfc_list = self.inputs.ecutwfc_list.get_list()

        kpoints_distance_list = getattr(self.inputs, "kpoints_distance_list", None)
        if kpoints_distance_list is not None:
            kpoints_values = kpoints_distance_list.get_list()
            use_kpoints_distance = True
        else:
            kpoints_values = self.inputs.kpoints_list.get_list()
            use_kpoints_distance = False

        all_finished_ok = True
        child_pks = []

        for ecutwfc, kpoints_val in product(ecutwfc_list, kpoints_values):
            if use_kpoints_distance:
                label = format_abacus_convergence_label(ecutwfc, kpoints_val)
            else:
                label = format_abacus_convergence_label(ecutwfc, kpoints_val)
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_convergence_results(
            child_pks=orm.List(child_pks),
            kpoints_mode=orm.Str("distance" if use_kpoints_distance else "mesh"),
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_convergence_results(child_pks, kpoints_mode=None):
    """通过 calcfunction 将原始文件解析过程记录在 Provenance Graph 中。"""
    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parsers import fetch_summary

    total_energy = {}
    num_atoms = {}
    total_energy_per_atom = {}
    wall_time_seconds = {}
    scf_steps = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        if not child.is_finished_ok:
            # Even failed children get a status row in the report.
            param_dict = child.inputs.abacus.parameters.get_dict()
            input_block = param_dict.get("input", {})
            ecutwfc = input_block.get("ecutwfc")
            if hasattr(child.inputs, "kpoints_distance"):
                kpoints_val = child.inputs.kpoints_distance.value
                label = format_abacus_convergence_label(ecutwfc, kpoints_val)
            elif hasattr(child.inputs, "kpoints"):
                kpoints_mesh = child.inputs.kpoints.get_kpoints_mesh()[0]
                label = format_abacus_convergence_label(ecutwfc, kpoints_mesh)
            else:
                continue
            status[label] = (
                int(child.exit_status)
                if child.exit_status is not None
                else -1
            )
            continue
        param_dict = child.inputs.abacus.parameters.get_dict()
        input_block = param_dict.get("input", {})
        ecutwfc = input_block.get("ecutwfc")

        if hasattr(child.inputs, "kpoints_distance"):
            kpoints_val = child.inputs.kpoints_distance.value
            label = format_abacus_convergence_label(ecutwfc, kpoints_val)
        elif hasattr(child.inputs, "kpoints"):
            kpoints_mesh = child.inputs.kpoints.get_kpoints_mesh()[0]
            label = format_abacus_convergence_label(ecutwfc, kpoints_mesh)
        else:
            continue

        # Unified parser: energy / wall-time / natoms in one call.
        summary = fetch_summary(child, "abacus")
        energy = summary["energy_ev"]
        wall_time = summary["time_s"]
        n_atoms = summary["natoms"]
        total_energy[label] = energy
        wall_time_seconds[label] = wall_time
        scf_steps[label] = summary["scf_steps"]
        if n_atoms:
            num_atoms[label] = n_atoms
        if energy is not None and n_atoms:
            total_energy_per_atom[label] = energy / n_atoms
        status[label] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )
    result = {
        "total_energy": total_energy,
        "num_atoms": num_atoms,
        "total_energy_per_atom": total_energy_per_atom,
        "wall_time_seconds": wall_time_seconds,
        "status": status,
    }
    if kpoints_mode is not None:
        result["kpoints_mode"] = str(kpoints_mode)
    return orm.Dict(result)
