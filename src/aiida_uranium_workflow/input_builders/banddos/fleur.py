"""FLEUR input builder for the banddos workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:class:`aiida_uranium_workflow.workflows.banddos.FleurBandAndDosWorkChain`.

The bundle's ``software_params['fleur'][preset_idx]`` is taken from
``parameters/fleur/scf.yml`` and is expected to be a flat dict with
these top-level keys (see :mod:`parameters.fleur.banddos`):

* ``wf_parameters``   — FLEUR ``FleurBaseWorkChain`` settings
                        (``mode='density'``, convergence criteria, …)
* ``calc_parameters`` — FLEUR input parameters (kmax / kpt / atom)

The band / DOS overrides (``band_wf`` / ``dos_wf``) come from the
**workflow protocol** (``parameters/banddos.yml``'s ``fleur`` block) —
the same single-source layout as magmom; the preset only provides the
SCF base. A missing protocol ``fleur`` block raises an error.

The adapter passes everything through to the new
``FleurBandAndDosWorkChain``, which itself submits two child
``FleurBandDosWorkChain`` (one mode='band', one mode='dos') sharing the
SCF namespace.
"""
from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Dict


class FleurBandAdapter(SoftwareAdapter):
    """Translate a ParamBundle into FleurBandAndDosWorkChain inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        # Entry point registered in pyproject.toml under
        # [project.entry-points."aiida.workflows"]
        return "uranium.banddos.fleur"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)  # copy; the YAML preset
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR banddos preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF namespace can't be built."
            )

        # The band / DOS ``wf_parameters`` overrides come from the
        # workflow protocol (``parameters/banddos.yml``'s ``fleur``
        # block), not from the preset — same single-source layout as
        # magmom. The preset only carries the SCF base. Check first so
        # a missing block fails before any AiiDA node is created.
        fleur_block = dict(self.workflow_data.get("fleur", {}))
        band_wf = fleur_block.get("band_wf")
        dos_wf = fleur_block.get("dos_wf")
        if not band_wf or not dos_wf:
            raise ValueError(
                "FLEUR banddos needs 'fleur.band_wf' / 'fleur.dos_wf' in "
                "the workflow protocol (parameters/banddos.yml, e.g. under "
                "'tdos') — the preset only provides the SCF base."
            )

        options = self.metadata.get("options", {})
        code_label = self.code_label

        # The SCF namespace is the YAML preset's wf_parameters /
        # calc_parameters. The scheduler fills in ``fleur`` / ``inpgen`` /
        # ``options`` etc. from the WorkChain's top-level inputs —
        # see ``FleurBandAndDosWorkChain._build_run_inputs``.
        scf = {
            "wf_parameters":   orm.Dict(dict(preset["wf_parameters"])),
            "calc_parameters": orm.Dict(dict(preset["calc_parameters"])),
            "structure":       structure,
        }

        # ``inpgen`` lives inside the SCF namespace as far as
        # ``FleurBandDosWorkChain`` is concerned — without it the
        # child WorkChain's input validation fails with
        # ``"no inpgen code was provided"``. The orchestrator passes
        # the full ``input.json["code"]`` mapping through
        # ``self.extra_codes``; we pull ``inpgen`` from there.
        inpgen_label = self.extra_codes.get("inpgen")
        if not inpgen_label:
            raise ValueError(
                "FLEUR banddos needs an 'inpgen' code label — add "
                "`\"inpgen\": \"<label>@<computer>\"` to the top-level "
                "`code` mapping in input.json (alongside 'fleur')."
            )
        scf["inpgen"] = orm.load_code(inpgen_label)

        # NOTE: ``scf`` must be a *plain Python dict* here, not
        # ``orm.Dict(dict=scf)``. The new WorkChain exposes the
        # ``scf`` namespace as a port and the AiiDA engine stores the
        # inputs as plain Python values; wrapping a Dict-of-Dict would
        # trip the JSON serializer at ``store_all()`` time.
        return {
            "structure": structure,
            "fleur":     orm.load_code(code_label),
            "options":   orm.Dict(dict=options) if options else orm.Dict(dict={}),
            "scf":       scf,
            "band_wf":   orm.Dict(dict=dict(band_wf)),
            "dos_wf":    orm.Dict(dict=dict(dos_wf)),
        }

    def _prepare_workflow_inputs(self):
        # Unused for banddos — the per-run overrides are read directly
        # from the workflow protocol's ``fleur`` block in
        # :meth:`_build_workchain_inputs`.
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        # Import directly to avoid the aiida.workflows entry-point
        # table — useful when the package is run from a checkout
        # without ``pip install -e .``.
        from aiida_uranium_workflow.workflows.banddos.fleur import (
            FleurBandAndDosWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=FleurBandAndDosWorkChain,
            inputs=inputs,
        )