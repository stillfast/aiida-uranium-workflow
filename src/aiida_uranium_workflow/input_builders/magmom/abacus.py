"""ABACUS input builder for the magmom workflow.

Reads the per-backend ``mag_list`` from the workflow protocol section
(e.g. ``parameters/magmom.yml``) and assembles inputs for
``AbacusMagmomWorkChain``.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, Dict, List


class AbacusMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into AbacusMagmomWorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "abacus.magmom"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
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
        if "kpoints_distance" in self.software_params:
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
        inputs["pseudo_family"] = orm.Str(
            str(self.software_params.get("pseudo_family", ""))
        )
        return inputs

    def _prepare_workflow_inputs(self) -> List[Any]:
        """Extract the ``mag_list`` from workflow_data.

        Each entry is a nested list of per-atom magnetizations, e.g.
        ``[[1.0, 1.0], [1.0, -1.0]]`` matching ABACUS ``stru.mag``.
        """
        lists = self.workflow_data.get("magmom_lists", {}).get("abacus", {})
        if not lists:
            return []
        return list(lists.get("mag_list", []))

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        mag_list = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        if mag_list:
            inputs["magmom_list"] = orm.List(list=mag_list)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
