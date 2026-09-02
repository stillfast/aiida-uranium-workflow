"""FLEUR input builder for the eos workflow.

Reuses the plugin ``fleur.eos`` WorkChain
(:class:`aiida_fleur.workflows.eos.FleurEosWorkChain`) directly — no
custom FLEUR EOS WorkChain. Reads the SCF base from
``parameters/fleur/scf.yml`` and the EOS scan settings from
``parameters/eos.yml``'s ``fleur`` block.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any


class FleurEosAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the plugin ``fleur.eos`` inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "fleur.eos"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR eos preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF base can't be built."
            )

        options = self.metadata.get("options", {})
        code_label = self.code_label

        # The plugin exposes the SCF inputs under the ``scf`` namespace
        # (structure excluded — it is a top-level input of fleur.eos).
        scf: dict[str, Any] = {
            "fleur": orm.load_code(code_label),
            "wf_parameters": orm.Dict(dict=dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
        }
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            scf["inpgen"] = orm.load_code(inpgen_label)
        if options:
            scf["options"] = orm.Dict(dict=options)

        # EOS scan settings from the protocol's ``fleur`` block
        # (points / step / guess — the plugin's wf_parameters keys).
        eos_proto = dict(self.workflow_data.get("fleur", {}) or {})
        defaults = {"points": 9, "step": 0.005, "guess": 1.00}
        eos_settings = dict(defaults)
        eos_settings.update({k: v for k, v in eos_proto.items() if k in defaults})

        inputs: dict[str, Any] = {
            "structure": structure,
            "scf": scf,
            "wf_parameters": orm.Dict(dict=eos_settings),
            "metadata": {
                "label": "fleur-eos",
                "description": "FLEUR equation-of-state scan (fleur.eos)",
            },
        }
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation (submits the plugin WorkChain
    # resolved from ``_workchain_entry_point``).
