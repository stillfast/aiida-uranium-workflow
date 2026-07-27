"""Scheduler layer — wires configuration to AiiDA and the input_builders.

Public surface:

* :class:`WorkflowOrchestrator` (base) — generic ``run()`` loop
* :func:`register_workflow` / :func:`get_orchestrator` — workflow registry
* concrete ``SmearWorkflowOrchestrator`` is in :mod:`.smear`
* concrete ``ConvergenceWorkflowOrchestrator`` is in :mod:`.convergence`
* concrete ``MagmomWorkflowOrchestrator`` is in :mod:`.magmom`
"""

from . import convergence  # noqa: F401  -- triggers self-registration
from . import smear  # noqa: F401  -- triggers self-registration
from . import magmom  # noqa: F401  -- triggers self-registration
from . import base_workchain  # noqa: F401  -- triggers self-registration
from .base import (
    get_orchestrator,
    get_workflow_entry,
    register_workflow,
    SubmittedJob,
    WorkflowEntry,
    WorkflowOrchestrator,
)

__all__ = [
    "WorkflowEntry",
    "WorkflowOrchestrator",
    "SubmittedJob",
    "register_workflow",
    "get_orchestrator",
    "get_workflow_entry",
]
