"""VASP input builder for the smear workflow."""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, List, Tuple

_VASP_ISMEAR_KW = {
    "gauss": 0,
    "mp": 1,
    "mp2": 2,
}


class VaspAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspSmearWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.smear"

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

    def _prepare_workflow_inputs(self) -> Tuple[List, List[float]]:
        lists = self.workflow_data.get("smear_lists", {})
        if not lists:
            return [], []
        smear = [_VASP_ISMEAR_KW[s] for s in lists.get("smear", [])]
        sigma = list(lists.get("sigma", []))  # VASP wants eV directly
        return smear, sigma
