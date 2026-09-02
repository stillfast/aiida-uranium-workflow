"""FLEUR input builder for the magmom workflow.

Reads the SCF namespace (``wf_parameters`` / ``calc_parameters``) from
the FLEUR magmom preset (``parameters/fleur/magmom.yml``) — the same
format as the banddos FLEUR presets. The per-config ``magmom_list`` is
**only** taken from the workflow protocol (``parameters/magmom.yml``,
``fleur.magmom_list``) — there is no preset-level fallback; a missing /
empty list raises an error.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Dict


class FleurMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into FleurMagmomWorkChain inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "uranium.magmom.fleur"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR magmom preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF namespace can't be built."
            )

        options = self.metadata.get("options", {})
        code_label = self.code_label

        inputs: dict[str, Any] = {
            "fleur": orm.load_code(code_label),
            "structure": structure,
            "wf_parameters": orm.Dict(dict=dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
            "options": orm.Dict(dict=options) if options else orm.Dict(dict={}),
        }

        # ``inpgen`` lives alongside ``fleur`` in the top-level
        # ``input.json["code"]`` mapping; the orchestrator forwards the
        # whole mapping through ``self.extra_codes``.
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            inputs["inpgen"] = orm.load_code(inpgen_label)

        return inputs

    def _prepare_workflow_inputs(self) -> list[dict[str, Any]]:
        """Extract the FLEUR ``magmom_list`` from workflow_data.

        Each entry is a dict: ``{"label": "FM", "bmu": 4.0,
        "inpxml_changes": [...]}``. Empty when the protocol carries no
        ``fleur.magmom_list``.
        """
        lists = self.workflow_data.get("magmom_lists", {}).get("fleur", {})
        if not lists:
            return []
        return list(lists.get("magmom_list", []))

    def adapt(self, structure) -> AdaptedInputs:
        from aiida import orm

        options = self.metadata.get("options", {})
        magmom_list = self._prepare_workflow_inputs()

        if not magmom_list:
            raise ValueError(
                "FLEUR magmom needs a 'fleur.magmom_list' in the workflow "
                "protocol (parameters/magmom.yml, e.g. under 'test_mag') — "
                "no preset-level fallback is supported."
            )

        inputs = self._build_workchain_inputs(structure)
        inputs["magmom_list"] = orm.List(list=magmom_list)

        self._inject_options(inputs, options)

        from aiida_uranium_workflow.workflows.magmom.fleur import (
            FleurMagmomWorkChain,
        )

        return AdaptedInputs(
            workchain_cls=FleurMagmomWorkChain,
            inputs=inputs,
        )
