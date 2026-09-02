"""Scheduler for the supercell workflow.

Binds the supercell workflow to:

* ``parameters/supercell.yml`` (protocol — the supercell matrix list
  with per-cell SCF overrides, per backend)
* ``parameters/abacus/scf.yml`` (per-backend SCF base, shared with the
  other workflows)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

Like eos / relax / elastic, supercell is **not** a parameter sweep —
one :class:`SupercellScfWorkChain` is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.supercell import AbacusSupercellAdapter
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_supercell_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks."""
    if not protocol:
        return {}
    return {
        "abacus": dict(protocol.get("abacus", {})),
    }


class SupercellWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the supercell workflow."""

    ADAPTERS = {
        "abacus": AbacusSupercellAdapter,
    }

    BACKENDS = ("abacus",)

    #: Per-backend preset-subkey mapping — the SCF base is the shared
    #: ``parameters/abacus/scf.yml``.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
    }


register_workflow(
    name="supercell",
    protocol_file="supercell.yml",
    parser_hook=parse_supercell_protocol,
    orchestrator_cls=SupercellWorkflowOrchestrator,
)
