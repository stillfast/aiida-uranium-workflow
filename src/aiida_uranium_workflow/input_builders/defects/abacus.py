"""ABACUS input builder for the defect workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the defect
settings (supercell matrix, defect definition, relax/scf mode, chemical
potentials) from the workflow protocol (``parameters/defects.yml``'s
``abacus`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.defects.abacus.AbacusDefectsWorkChain`.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any

import copy


class AbacusDefectsAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS defect WorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.defects.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        proto = dict(self.workflow_data.get("abacus", {}) or {})
        if "defect" not in proto:
            raise ValueError(
                "ABACUS defect preset is missing 'defect' — define "
                "{'type': 'vacancy'|'interstitial', ...} in parameters/defects.yml."
            )

        params = copy.deepcopy(self.software_params["parameters"])
        options = self.metadata.get("options", {})

        base: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(params),
                "metadata": {"options": options} if options else {},
            },
        }
        if "kpoints_distance" in self.software_params:
            base["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            base["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            base["pseudo_family"] = orm.Str(str(self.software_params["pseudo_family"]))

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "defect": orm.Dict(dict=dict(proto["defect"])),
            "wf_parameters": orm.Dict(
                dict=dict(proto.get("wf_parameters") or {"mode": "scf"})
            ),
        }
        if "supercell_matrix" in proto:
            inputs["supercell_matrix"] = orm.List(list=list(proto["supercell_matrix"]))
        if "relax_settings" in proto:
            inputs["relax_settings"] = orm.Dict(dict=dict(proto["relax_settings"]))
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []
