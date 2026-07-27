"""VASP input builder for the magmom workflow.

Reads the per-backend ``magmom_mapping_list`` from the workflow protocol
section (e.g. ``parameters/magmom.yml``) and assembles inputs for
``VaspMagmomWorkChain``.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, Dict, List


class VaspMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspMagmomWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.magmom"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
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
        if "kpoints_spacing" in entry:
            inputs["kpoints_spacing"] = orm.Float(entry["kpoints_spacing"])
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            inputs["kpoints"] = kpoints_mesh
        return inputs

    def _prepare_workflow_inputs(self) -> List[Dict[str, Any]]:
        """Extract the ``magmom_mapping_list`` from workflow_data.

        Each entry is a dict like ``{"Si": 1.0}`` or ``{"U": [1.0, -1.0]}``.
        """
        lists = self.workflow_data.get("magmom_lists", {}).get("vasp", {})
        if not lists:
            return []
        return list(lists.get("magmom_mapping_list", []))

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        magmom_mapping_list = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        if magmom_mapping_list:
            inputs["magmom_list"] = orm.List(list=magmom_mapping_list)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
