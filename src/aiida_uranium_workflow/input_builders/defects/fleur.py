"""FLEUR input builder for the defect workflow.

Reads the SCF base from ``parameters/fleur/scf.yml`` and the defect
settings (supercell matrix, defect definition, relax/scf mode, chemical
potentials) from the workflow protocol (``parameters/defects.yml``'s
``fleur`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.defects.fleur.FleurDefectsWorkChain`.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
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
        if "supercell_matrix" in proto:
            inputs["supercell_matrix"] = orm.List(list=list(proto["supercell_matrix"]))
        if "relax_settings" in proto:
            inputs["relax_settings"] = orm.Dict(dict=dict(proto["relax_settings"]))
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []
