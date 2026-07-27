"""Scheduler for the smear workflow.

Binds the smear workflow to:

* ``parameters/protocol/smear.yml`` (protocol section)
* a parser hook that extracts ``smear_list`` / ``sigma_list``
* the orchestrator below

Once registered (``register_workflow(...)`` at import time), the
generic :func:`schedulers.get_orchestrator` can find it.
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders import AbacusAdapter, VaspAdapter
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Smear-specific protocol parser
# ---------------------------------------------------------------------------


def parse_smear_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Extract smear_list + sigma_list from the loaded protocol section."""
    smear = list(protocol.get("smear_list", []))
    sigma = [float(s) for s in protocol.get("sigma_list", [])]
    if not smear or not sigma:
        raise ValueError("Protocol smear_list / sigma_list must be non-empty")
    return {"smear_lists": {"smear": smear, "sigma": sigma}}


# ---------------------------------------------------------------------------
# Orchestrator (just wires the right input_builders + backends)
# ---------------------------------------------------------------------------


class SmearWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the smear workflow."""

    ADAPTERS = {
        "abacus": AbacusAdapter,
        "vasp": VaspAdapter,
    }

    BACKENDS = ("abacus", "vasp")

    #: Smear's ``input.json`` uses different sub-keys per backend:
    #: ``"abacus": {"smear": [...]}`` and ``"vasp": {"vasp": "test"}``.
    #: The base class's :meth:`_preset_names_for` honours this mapping.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "smear",
        "vasp": "vasp",
    }


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_workflow(
    name="smear",
    protocol_file="smear.yml",
    parser_hook=parse_smear_protocol,
    orchestrator_cls=SmearWorkflowOrchestrator,
)
