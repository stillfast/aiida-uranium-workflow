"""Scheduler for the eos workflow.

Binds the eos workflow to:

* ``parameters/eos.yml`` (protocol — per preset, an ``abacus`` and a
  ``fleur`` block with the EOS scan settings: points / step / guess)
* ``parameters/abacus/scf.yml`` / ``parameters/fleur/scf.yml``
  (per-backend SCF base, shared with the other workflows)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

ABACUS runs the in-repo :class:`AbacusEosWorkChain`
(``uranium.eos.abacus``); FLEUR reuses the plugin ``fleur.eos``
WorkChain directly. Like banddos / elastic / relax, eos is **not** a
parameter sweep — one EOS WorkChain is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.eos import (
    AbacusEosAdapter,
    FleurEosAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_eos_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks."""
    if not protocol:
        return {}
    return {
        "abacus": dict(protocol.get("abacus", {})),
        "fleur": dict(protocol.get("fleur", {})),
    }


class EosWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the eos workflow."""

    ADAPTERS = {
        "abacus": AbacusEosAdapter,
        "fleur": FleurEosAdapter,
    }

    BACKENDS = ("abacus", "fleur")

    #: Per-backend preset-subkey mapping — both backends share the
    #: unified ``parameters/<backend>/scf.yml``.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur": "scf",
    }


register_workflow(
    name="eos",
    protocol_file="eos.yml",
    parser_hook=parse_eos_protocol,
    orchestrator_cls=EosWorkflowOrchestrator,
)
