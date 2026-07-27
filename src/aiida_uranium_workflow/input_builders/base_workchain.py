"""Input builders for direct ABACUS and VASP base WorkChains."""

from __future__ import annotations

import copy
from typing import Any

from .base import SoftwareAdapter


class AbacusBaseWorkChainAdapter(SoftwareAdapter):
    """Build inputs accepted directly by ``abacus.base``."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "abacus.base"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        entry = self.software_params
        inputs: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(copy.deepcopy(entry["parameters"])),
                "structure": structure,
                "metadata": {},
            }
        }
        if "kpoints_distance" in entry:
            inputs["kpoints_distance"] = orm.Float(float(entry["kpoints_distance"]))
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            kpoints = DataFactory("core.array.kpoints")()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            inputs["kpoints"] = kpoints
        if "pseudo_family" in entry:
            inputs["pseudo_family"] = orm.Str(str(entry["pseudo_family"]))
        return inputs


class VaspBaseWorkChainAdapter(SoftwareAdapter):
    """Build inputs accepted directly by ``vasp.v2.vasp``."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.v2.vasp"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        entry = self.software_params
        incar = copy.deepcopy(entry.get("parameters", {}).get("incar", {}))
        inputs: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "structure": structure,
            "parameters": orm.Dict(dict={"incar": incar}),
            "potential_family": orm.Str(str(entry["potential_family"])),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
            "calc": {"metadata": {}},
        }
        if "kpoints_spacing" in entry:
            inputs["kpoints_spacing"] = orm.Float(float(entry["kpoints_spacing"]))
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            kpoints = DataFactory("core.array.kpoints")()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            inputs["kpoints"] = kpoints
        return inputs
