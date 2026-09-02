"""Scheduler for the banddos workflow.

Binds the banddos workflow to:

* ``parameters/banddos.yml`` (protocol — one entry per preset with
  per-backend blocks: ``abacus`` (the ``band_settings`` namespace:
  ``run_bands`` / ``run_dos`` / ``band_kpoints_distance`` /
  ``dos_kpoints_distance`` / …) and ``fleur`` (``band_wf`` /
  ``dos_wf`` overrides). The per-backend layout mirrors
  ``parameters/magmom.yml``.
* ``parameters/abacus/banddos.yml`` / ``parameters/fleur/banddos.yml``
  (per-backend SCF presets — physical / SCF parameters only)
* a parser hook that splits the protocol section into the per-backend
  blocks inside ``workflow_data``
* the orchestrator below

Unlike smear / convergence / magmom, the banddos workflow is **not** a
parameter sweep — one ``AbacusBandWorkChain`` is submitted per preset
(ABACUS) or one ``FleurBandAndDosWorkChain`` per preset (FLEUR).
"""
from __future__ import annotations

from aiida_uranium_workflow.input_builders.banddos import (
    AbacusBandAdapter,
    FleurBandAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Banddos-specific protocol parser
# ---------------------------------------------------------------------------


def parse_banddos_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks.

    The ``parameters/banddos.yml`` file (protocol path) now uses the
    same per-backend layout as ``parameters/magmom.yml`` — each preset
    carries an ``abacus`` and a ``fleur`` block::

        tdos:
          abacus:
            run_bands: True
            run_dos: True
            band_kpoints_distance: 0.02
            dos_kpoints_distance: 0.1
            ...
          fleur:
            band_wf: {...}
            dos_wf: {...}

    The ``abacus`` block is forwarded verbatim as the
    ``AbacusBandWorkChain``'s ``band_settings`` namespace; the ``fleur``
    block (``band_wf`` / ``dos_wf``) is forwarded to the FLEUR adapter.
    """
    if not protocol:
        return {}
    return {
        "band_settings": dict(protocol.get("abacus", {})),
        "fleur": dict(protocol.get("fleur", {})),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BanddosWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the banddos workflow.

    Supports both ABACUS (``AbacusBandWorkChain``) and FLEUR
    (``FleurBandAndDosWorkChain``). The ``parameters/banddos.yml``
    protocol presets carry per-backend blocks — ``abacus`` (the
    ``band_settings`` namespace) and ``fleur`` (``band_wf`` /
    ``dos_wf``); the per-backend preset files under
    ``parameters/<backend>/scf.yml`` carry the SCF / physical
    parameters (shared with relax).
    """

    ADAPTERS = {
        "abacus": AbacusBandAdapter,
        "fleur":  FleurBandAdapter,
    }

    BACKENDS = ("abacus", "fleur")

    #: Per-backend preset-subkey mapping — both backends share the
    #: unified ``parameters/<backend>/scf.yml`` (same file as relax):
    #: ``"abacus": {"scf": [...]}``, ``"fleur": {"scf": "..."}``.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur":  "scf",
    }


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------


register_workflow(
    name="banddos",
    protocol_file="banddos.yml",
    parser_hook=parse_banddos_protocol,
    orchestrator_cls=BanddosWorkflowOrchestrator,
)