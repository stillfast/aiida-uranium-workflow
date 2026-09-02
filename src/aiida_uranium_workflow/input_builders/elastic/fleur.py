"""FLEUR input builder for the elastic workflow.

Reads the SCF base from ``parameters/fleur/scf.yml`` and the strain
lists from the workflow protocol (``parameters/elastic.yml``'s
``fleur`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.elastic.fleur.FleurElasticWorkChain`.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Dict


class FleurElasticAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the FLEUR elastic WorkChain inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "uranium.elastic.fleur"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR elastic preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF base can't be built."
            )

        options = self.metadata.get("options", {})
        code_label = self.code_label
        elastic_proto = dict(self.workflow_data.get("fleur", {}) or {})

        inputs: dict[str, Any] = {
            "fleur": orm.load_code(code_label),
            "structure": structure,
            "wf_parameters": orm.Dict(dict=dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
            "options": orm.Dict(dict=options) if options else orm.Dict(dict={}),
        }
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            inputs["inpgen"] = orm.load_code(inpgen_label)

        norm = elastic_proto.get("norm_strains") or [-0.010, -0.005, 0.005, 0.010]
        shear = elastic_proto.get("shear_strains") or [-0.010, -0.005, 0.005, 0.010]
        inputs["norm_strains"] = orm.List(list=norm)
        inputs["shear_strains"] = orm.List(list=shear)

        # User-method combined-strain modes (biaxial / triaxial Voigt
        # templates) probe the off-diagonal constants C12 / C13 / C23.
        combined = elastic_proto.get("combined_strains")
        if combined:
            inputs["combined_strains"] = orm.List(list=list(combined))
        # Relax internal coordinates under each strain
        # (FleurRelaxWorkChain, relaxation_type='atoms', cell fixed) —
        # default True; set False for clamped-ion constants. For cells
        # without free internal coordinates (e.g. the 2-atom bcc U cell)
        # the plain FleurScfWorkChain (density-converged SCF) gives
        # EOS-quality energies and is preferable.
        inputs["relax_internal"] = orm.Bool(
            elastic_proto.get("relax_internal", True)
        )
        # Optional FleurRelaxWorkChain wf-parameter overrides for the
        # strained cells, e.g. {"run_final_scf": true} for meV-accurate
        # energies after the relaxation.
        relax_wf = elastic_proto.get("relax_wf_parameters")
        if relax_wf:
            inputs["relax_wf_parameters"] = orm.Dict(dict=dict(relax_wf))
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.elastic.fleur import (
            FleurElasticWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=FleurElasticWorkChain,
            inputs=inputs,
        )
