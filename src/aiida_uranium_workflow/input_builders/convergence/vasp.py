"""VASP input builder for the convergence workflow."""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, List, Tuple


class VaspConvergenceAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspConvergenceWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.convergence"

    def _build_workchain_inputs(self, structure, include_spacing: bool = True) -> dict[str, Any]:
        from aiida import orm

        import copy

        entry = self.software_params
        incar = copy.deepcopy(entry.get("parameters", {}).get("incar", {}))

        options = self.metadata.get("options", {})

        inputs: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "structure": structure,
            "parameters": {"incar": incar},
            "potential_family": orm.Str(entry["potential_family"]),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
            "calc": {"metadata": {"options": options} if options else {}},
        }
        if include_spacing and "kpoints_spacing" in entry:
            inputs["kpoints_spacing"] = orm.Float(entry["kpoints_spacing"])
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            inputs["kpoints"] = kpoints_mesh
        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[List, List, str]:
        """Extract encut_list and kpoints lists from workflow_data.

        Returns:
            (encut_list, kpoints_values, mode) where mode is either 'spacing' or 'mesh'
        """
        lists = self.workflow_data.get("convergence_lists", {}).get("vasp", {})
        if not lists:
            return [], [], "spacing"
        encut_list = lists.get("encut_list", [])

        kpoints_mesh_list = lists.get("kpoints_mesh_list", [])
        if kpoints_mesh_list:
            return encut_list, kpoints_mesh_list, "mesh"

        kpoints_spacing_list = lists.get("kpoints_spacing_list", [])
        return encut_list, kpoints_spacing_list, "spacing"

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        encut_list, kpoints_values, kpoints_mode = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(
            structure, include_spacing=(kpoints_mode != "mesh")
        )
        if encut_list:
            inputs["encut_list"] = orm.List(list=encut_list)

        if kpoints_values:
            if kpoints_mode == "mesh":
                inputs["kpoints_list"] = orm.List(list=kpoints_values)
            else:
                inputs["kpoints_spacing_list"] = orm.List(list=kpoints_values)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
