"""Scheduler for the defect workflow.

Binds the defect workflow to:

* ``parameters/defects.yml`` (protocol — per preset, an ``abacus`` and a
  ``fleur`` block with the supercell / defect / relax-or-scf / chemical
  potential settings; the same per-backend layout as magmom.yml)
* ``parameters/abacus/scf.yml`` / ``parameters/fleur/scf.yml``
  (per-backend SCF base)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

Like eos / elastic / relax, defects is **not** a parameter sweep — one
defect WorkChain is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.defects import (
    AbacusDefectsAdapter,
    FleurDefectsAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_defects_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks."""
    if not protocol:
        return {}
    return {
        "abacus": dict(protocol.get("abacus", {})),
        "fleur": dict(protocol.get("fleur", {})),
    }


class DefectsWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the defect workflow."""

    ADAPTERS = {
        "abacus": AbacusDefectsAdapter,
        "fleur": FleurDefectsAdapter,
    }

    BACKENDS = ("abacus", "fleur")

    #: Per-backend preset-subkey mapping — both backends share the
    #: unified ``parameters/<backend>/scf.yml``.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur": "scf",
    }


register_workflow(
    name="defects",
    protocol_file="defects.yml",
    parser_hook=parse_defects_protocol,
    orchestrator_cls=DefectsWorkflowOrchestrator,
)
