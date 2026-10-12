"""FLEUR input builder for the defect workflow.

Reads the SCF base from ``parameters/fleur/scf.yml`` and the defect
settings (supercell matrix, defect definition, relax/scf mode, chemical
potentials) from the workflow protocol (``parameters/defects.yml``'s
``fleur`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.defects.fleur.FleurDefectsWorkChain`.
"""

from __future__ import annotations

from ..base import SoftwareAdapter

from aiida_uranium_workflow.utils.defects import normalize_supercell_cells
from typing import Any


class FleurDefectsAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the FLEUR defect WorkChain inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "uranium.defects.fleur"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR defect preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF base can't be built."
            )

        proto = dict(self.workflow_data.get("fleur", {}) or {})
        if "defect" not in proto:
            raise ValueError(
                "FLEUR defect preset is missing 'defect' — define "
                "{'type': 'vacancy'|'interstitial', ...} in parameters/defects.yml."
            )

        options = self.metadata.get("options", {})
        base: dict[str, Any] = {
            "fleur": orm.load_code(self.code_label),
            "wf_parameters": orm.Dict(dict=dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
        }
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            base["inpgen"] = orm.load_code(inpgen_label)
        if options:
            base["options"] = orm.Dict(dict=options)

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
        # ``supercell_matrix``).
        spec = proto.get("supercell_cells")
        if spec is None:
            spec = proto.get("supercell_matrices")
        if spec is None:
            spec = proto.get("supercell_matrix")
        cells = normalize_supercell_cells(spec)
        for cell in cells:
            if cell.get("kpoints_mesh") or cell.get("kpoints_distance"):
                raise ValueError(
                    "per-cell k-point overrides (kpoints_mesh / "
                    "kpoints_distance) are only supported by the ABACUS "
                    "backend; set the k-mesh in the FLEUR preset instead."
                )
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
