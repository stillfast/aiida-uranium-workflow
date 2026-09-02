"""FLEUR input builder for the relax workflow.

Reads the SCF base from ``parameters/fleur/scf.yml`` and the relax
settings from the workflow protocol (``parameters/relax.yml``'s
``fleur`` block), then assembles inputs for the **plugin** ``fleur.relax``
WorkChain (:class:`aiida_fleur.workflows.relax.FleurRelaxWorkChain`) —
no custom WorkChain is needed.

The plugin exposes the SCF inputs under the ``scf`` namespace
(``scf.fleur`` / ``scf.inpgen`` / ``scf.structure`` /
``scf.calc_parameters`` / ``scf.options`` ...) and the relax settings in
the top-level ``wf_parameters``; this adapter mirrors the layout that the
former wrapper composed internally.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any


class FleurRelaxAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the plugin ``fleur.relax`` inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "fleur.relax"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR relax preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF base can't be built."
            )

        options = self.metadata.get("options", {})
        code_label = self.code_label

        # The plugin's SCF namespace gets the SCF base + codes / options;
        # the top-level ``wf_parameters`` holds the relax settings (the
        # plugin forces the SCF ``mode`` to 'force' internally).
        scf: dict[str, Any] = {
            "fleur": orm.load_code(code_label),
            "structure": structure,
            "wf_parameters": orm.Dict(dict=dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
        }
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            scf["inpgen"] = orm.load_code(inpgen_label)
        if options:
            scf["options"] = orm.Dict(dict=options)

        # Relax settings from the protocol's ``fleur`` block
        # (force_criterion / relax_iter / run_final_scf / ...).
        fleur_proto = dict(self.workflow_data.get("fleur", {}) or {})
        relax_para = dict(fleur_proto)

        inputs: dict[str, Any] = {
            "wf_parameters": orm.Dict(dict=relax_para),
            "scf": scf,
            "metadata": {
                "label": "fleur-relax",
                "description": "FLEUR full relaxation (fleur.relax)",
            },
        }
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation, which submits the plugin
    # WorkChain resolved from ``_workchain_entry_point`` ("fleur.relax").
