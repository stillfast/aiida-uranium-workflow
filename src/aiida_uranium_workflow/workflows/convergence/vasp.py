from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from itertools import product

from aiida_uranium_workflow.utils.labels import format_vasp_convergence_label

ChildWorkChain = WorkflowFactory("vasp.v2.vasp")


class VaspConvergenceWorkChain(WorkChain):
    """Sweep over ``(encut, kpoints_spacing)`` combinations and gather the total energy."""

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
            "kpoints_spacing_list",
            valid_type=orm.List,
            required=False,
            help="List of kpoints spacing values for VASP. the unit is A-1/2pi",
        )
        spec.input(
            "kpoints_list",
            valid_type=orm.List,
            required=False,
            help="List of kpoints values for VASP.",
        )
        spec.input(
            "encut_list",
            valid_type=orm.List,
            help="List of encut values (in eV) for VASP.",
        )
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "convergence scan completed successfully.")
        spec.exit_code(
            300, "ERROR_CHILD", "At least a single child calculation failed."
        )
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse total energy.")

    def submit_children(self):
        """Submit one VASP run for every (encut, kpoints) pair.

        Supports two modes:
        - kpoints_spacing_list: List of kpoints spacing values (A^-1 * 2pi)
        - kpoints_list: List of kpoints mesh tuples, e.g. [(2,2,2), (4,4,4)]

        Only one of the two inputs should be provided.
        """
        encut_list = self.inputs.encut_list.get_list()

        kpoints_spacing_list = getattr(self.inputs, "kpoints_spacing_list", None)
        kpoints_list = getattr(self.inputs, "kpoints_list", None)

        use_kpoints_spacing = kpoints_spacing_list is not None
        use_kpoints_mesh = kpoints_list is not None

        if use_kpoints_spacing:
            kpoints_values = kpoints_spacing_list.get_list()
            self.ctx.convergence_mode = "spacing"
        elif use_kpoints_mesh:
            kpoints_values = kpoints_list.get_list()
            self.ctx.convergence_mode = "mesh"
        else:
            raise ValueError(
                "Either kpoints_spacing_list or kpoints_list must be provided"
            )

        planned = list(product(kpoints_values, encut_list))
        self.ctx.convergence_pairs = planned

        base_inputs = self.exposed_inputs(ChildWorkChain, agglomerate=True)

        for kpoints_val, encut in planned:
            if use_kpoints_spacing:
                label = format_vasp_convergence_label(kpoints_val, encut)
            else:
                label = format_vasp_convergence_label(kpoints_val, encut)

            param_dict = base_inputs.parameters.get_dict()
            incar = param_dict.setdefault("incar", {})
            incar["encut"] = encut

            child_inputs = dict(base_inputs)
            child_inputs["parameters"] = orm.Dict(param_dict)

            if use_kpoints_spacing:
                child_inputs["kpoints_spacing"] = orm.Float(kpoints_val)
            else:
                # Drop any inherited ``kpoints_spacing`` so the child only sees
                # the explicit ``kpoints`` mesh and the calcfunction labels
                # this run as ``encut_*_kpoints_<NxNxN>`` instead of
                # ``encut_*_kpoints_spacing_<value>``.
                child_inputs.pop("kpoints_spacing", None)

                from aiida.plugins import DataFactory

                KpointsData = DataFactory("core.array.kpoints")
                kpoints_mesh = KpointsData()
                kpoints_mesh.set_kpoints_mesh(list(kpoints_val))
                child_inputs["kpoints"] = kpoints_mesh

            child_inputs["metadata"] = {"label": label}
            child_inputs["calc"] = {
                **base_inputs.get("calc", {}),
            }

            running = self.submit(ChildWorkChain, **child_inputs)

            self.to_context(**{label: running})

    def gather_results(self):
        encut_list = self.inputs.encut_list.get_list()

        kpoints_spacing_list = getattr(self.inputs, "kpoints_spacing_list", None)
        if kpoints_spacing_list is not None:
            kpoints_values = kpoints_spacing_list.get_list()
            use_kpoints_spacing = True
        else:
            kpoints_values = self.inputs.kpoints_list.get_list()
            use_kpoints_spacing = False

        all_finished_ok = True
        child_pks = []

        for kpoints_val, encut in product(kpoints_values, encut_list):
            if use_kpoints_spacing:
                label = format_vasp_convergence_label(kpoints_val, encut)
            else:
                label = format_vasp_convergence_label(kpoints_val, encut)
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_convergence_results(
            child_pks=orm.List(child_pks),
            kpoints_mode=orm.Str("spacing" if use_kpoints_spacing else "mesh"),
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
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        if not child.is_finished_ok:
            # Even failed children get a status row in the report.
            param_dict = child.inputs.parameters.get_dict()
            incar = param_dict.get("incar", {})
            encut = incar.get("encut")
            if hasattr(child.inputs, "kpoints_spacing"):
                kpoints_val = child.inputs.kpoints_spacing.value
                label = f"encut_{encut}_kpoints_spacing_{kpoints_val}".replace(".", "_")
            elif hasattr(child.inputs, "kpoints"):
                kpoints_mesh = child.inputs.kpoints.get_kpoints_mesh()[0]
                kpoints_str = "x".join(str(k) for k in kpoints_mesh)
                label = f"encut_{encut}_kpoints_{kpoints_str}".replace(".", "_")
            else:
                continue
            status[label] = (
                int(child.exit_status)
                if child.exit_status is not None
                else -1
            )
            continue
        param_dict = child.inputs.parameters.get_dict()
        incar = param_dict.get("incar", {})
        encut = incar.get("encut")

        if hasattr(child.inputs, "kpoints_spacing"):
            kpoints_val = child.inputs.kpoints_spacing.value
            label = f"encut_{encut}_kpoints_spacing_{kpoints_val}".replace(".", "_")
        elif hasattr(child.inputs, "kpoints"):
            kpoints_mesh = child.inputs.kpoints.get_kpoints_mesh()[0]
            kpoints_str = "x".join(str(k) for k in kpoints_mesh)
            label = f"encut_{encut}_kpoints_{kpoints_str}".replace(".", "_")
        else:
            continue

        # Unified parser: energy / wall-time / natoms in one call.
        summary = fetch_summary(child, "vasp")
        energy = summary["energy_ev"]
        wall_time = summary["time_s"]
        n_atoms = summary["natoms"]
        total_energy[label] = energy
        wall_time_seconds[label] = wall_time
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
