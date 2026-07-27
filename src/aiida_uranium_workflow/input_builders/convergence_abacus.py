"""ABACUS input builder for the convergence workflow."""

from __future__ import annotations

from .base import SoftwareAdapter
from typing import Any, List, Tuple


class AbacusConvergenceAdapter(SoftwareAdapter):
    """Translate a ParamBundle into AbacusConvergenceWorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "abacus.convergence"

    def _build_workchain_inputs(self, structure, include_distance: bool = True) -> dict[str, Any]:
        from aiida import orm

        import copy

        params = copy.deepcopy(self.software_params["parameters"])

        code_label = self.code_label

        inputs: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(code_label),
                "parameters": orm.Dict(params),
                "structure": structure,
                "metadata": {},
            },
        }
        if "pseudo_family" in self.software_params:
            inputs["pseudo_family"] = orm.Str(
                str(self.software_params["pseudo_family"])
            )
        if include_distance and "kpoints_distance" in self.software_params:
            inputs["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(
                list(self.software_params["kpoints_mesh"])
            )
            inputs["kpoints"] = kpoints_mesh
        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[List, List, str]:
        """Extract ecutwfc_list and kpoints lists from workflow_data.

        Returns:
            (ecutwfc_list, kpoints_values, mode) where mode is either 'distance' or 'mesh'
        """
        lists = self.workflow_data.get("convergence_lists", {}).get("abacus", {})
        if not lists:
            return [], [], "distance"
        ecutwfc_list = lists.get("ecutwfc_list", [])

        kpoints_mesh_list = lists.get("kpoints_mesh_list", [])
        if kpoints_mesh_list:
            return ecutwfc_list, kpoints_mesh_list, "mesh"

        kpoints_distance_list = lists.get("kpoints_distance_list", [])
        return ecutwfc_list, kpoints_distance_list, "distance"

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        ecutwfc_list, kpoints_values, kpoints_mode = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(
            structure, include_distance=(kpoints_mode != "mesh")
        )
        if ecutwfc_list:
            inputs["ecutwfc_list"] = orm.List(list=ecutwfc_list)

        if kpoints_values:
            if kpoints_mode == "mesh":
                inputs["kpoints_list"] = orm.List(list=kpoints_values)
            else:
                inputs["kpoints_distance_list"] = orm.List(list=kpoints_values)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
