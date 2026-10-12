"""ABACUS input builder for the defect workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the defect
settings (supercell matrix, defect definition, relax/scf mode, chemical
potentials) from the workflow protocol (``parameters/defects.yml``'s
``abacus`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.defects.abacus.AbacusDefectsWorkChain`.
"""

from __future__ import annotations

from ..base import filter_scheduler_options, SoftwareAdapter
from typing import Any

from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

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
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )

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
        # Supercell matrices: a single matrix (legacy key) or a list for a
        # supercell-size scan (``supercell_matrices`` / a list under
        # ``supercell_matrix``). Normalised here so the WorkChain always
        # gets a well-formed value.
        spec = proto.get("supercell_cells")
        if spec is None:
            spec = proto.get("supercell_matrices")
        if spec is None:
            spec = proto.get("supercell_matrix")
        cells = normalize_supercell_cells(spec)
        if cells and all(set(cell) == {"matrix"} for cell in cells):
            # plain matrices → legacy inputs (one matrix per cell)
            matrices = [cell["matrix"] for cell in cells]
            if len(matrices) == 1:
                inputs["supercell_matrix"] = orm.List(list=matrices[0])
            else:
                inputs["supercell_matrices"] = orm.List(list=matrices)
        elif cells:
            inputs["supercell_cells"] = orm.List(list=cells)
        if "relax_settings" in proto:
            inputs["relax_settings"] = orm.Dict(dict=dict(proto["relax_settings"]))
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []
