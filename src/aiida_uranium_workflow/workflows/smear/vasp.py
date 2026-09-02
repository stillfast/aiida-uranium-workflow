from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from itertools import product

import xml.etree.ElementTree as ET
from aiida_uranium_workflow.utils.labels import format_smear_label

ChildWorkChain = WorkflowFactory("vasp.v2.vasp")


class VaspSmearWorkChain(WorkChain):
    """Sweep over ``(ismear, sigma)`` combinations and gather the electronic entropy."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=[
                "code",
                "structure",
                "kpoints_spacing",
                "kpoints",
                "parameters",
                "potential_family",
                "potential_mapping",
                "calc",
            ],
        )
        spec.input(
            "smear", valid_type=orm.List, help="List of ISMEAR integers for VASP."
        )
        spec.input(
            "sigma", valid_type=orm.List, help="List of sigma values (in eV) for VASP."
        )
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "smear scan completed successfully.")
        spec.exit_code(
            300, "ERROR_CHILD", "At least a single child calculation failed."
        )
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse electronic entropy.")

    def submit_children(self):
        """Submit one VASP run for every (ismear, sigma) pair."""
        smear_list = self.inputs.smear.get_list()
        sigma_list = self.inputs.sigma.get_list()

        planned = list(product(smear_list, sigma_list))
        self.ctx.smear_pairs = planned

        base_inputs = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        base_calc_meta = getattr(base_inputs.get("calc", {}), "metadata", {})
        if hasattr(base_calc_meta, "get_dict"):
            base_calc_meta = base_calc_meta.get_dict()
        elif not isinstance(base_calc_meta, dict):
            base_calc_meta = {}

        for smear, sigma in planned:
            label = format_smear_label(smear, sigma)

            param_dict = base_inputs.parameters.get_dict()
            incar = param_dict.setdefault("incar", {})
            incar["ismear"] = smear
            incar["sigma"] = sigma

            child_inputs = dict(base_inputs)
            child_inputs["parameters"] = orm.Dict(param_dict)
            child_inputs["metadata"] = {"label": label}
            child_inputs["calc"] = {
                **base_inputs.get("calc", {}),
            }
            if hasattr(self.inputs, "kpoints"):
                child_inputs["kpoints"] = self.inputs.kpoints
                if "kpoints_spacing" in child_inputs:
                    del child_inputs["kpoints_spacing"]

            running = self.submit(ChildWorkChain, **child_inputs)

            self.to_context(**{label: running})

    def gather_results(self):

        smear_list = self.inputs.smear.get_list()
        sigma_list = self.inputs.sigma.get_list()

        all_finished_ok = True
        child_pks = []

        for smear, sigma in product(smear_list, sigma_list):
            label = format_smear_label(smear, sigma)
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_smear_results(
            child_pks=orm.List(child_pks)
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_smear_results(child_pks):
    """通过 calcfunction 将原始文件解析过程记录在 Provenance Graph 中。"""
    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parsers import fetch_summary

    eentropy = {}
    num_atoms = {}
    eentropy_per_atom = {}
    total_energy = {}
    wall_time_seconds = {}
    scf_steps = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        incar = child.inputs.parameters.get_dict().get("incar", {})
        smear = incar.get("ismear")
        sigma = incar.get("sigma")
        label = format_smear_label(smear, sigma)

        # Capture exit status for every submitted child, even those
        # that did not finish OK — the report needs to surface
        # failures explicitly. ``exit_status`` is ``None`` for
        # unfinished processes; we render that as ``-1`` so the
        # Markdown table shows a non-empty cell.
        status[label] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )

        if not child.is_finished_ok:
            continue

        # Energy + wall-time from misc (shared parser).
        try:
            summary = fetch_summary(child, "vasp")
            energy = summary["energy_ev"]
            wall_time = summary["time_s"]
            total_energy[label] = energy
            wall_time_seconds[label] = wall_time
            scf_steps[label] = summary["scf_steps"]
        except Exception as e:  # noqa: BLE001 — defensive: misc missing
            total_energy[label] = f"Parsing failed: {str(e)}"
            wall_time_seconds[label] = None

        try:
            retrieved = child.outputs.retrieved
            with retrieved.open("vasprun.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()

            calculation = root.find("calculation")
            eentropy_values = []
            if calculation is not None:
                for scstep in calculation.findall("scstep"):
                    energy = scstep.find("energy")
                    if energy is not None:
                        ee_node = energy.find('i[@name="eentropy"]')
                        if ee_node is not None:
                            eentropy_values.append(float(ee_node.text.strip()))

            if eentropy_values:
                last_eentropy = eentropy_values[-1]
                n_atoms = summary.get("natoms")

                eentropy[label] = last_eentropy
                if n_atoms:
                    num_atoms[label] = n_atoms
                    eentropy_per_atom[label] = last_eentropy / n_atoms
            else:
                eentropy[label] = None

        except Exception as e:
            eentropy[label] = f"Parsing failed: {str(e)}"

    return orm.Dict(
        {
            "eentropy": eentropy,
            "num_atoms": num_atoms,
            "eentropy_per_atom": eentropy_per_atom,
            "total_energy": total_energy,
            "wall_time_seconds": wall_time_seconds,
            "scf_steps": scf_steps,
            "status": status,
        }
    )
