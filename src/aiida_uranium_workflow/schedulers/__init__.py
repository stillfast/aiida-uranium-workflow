"""Scheduler layer — wires configuration to AiiDA and the input_builders.

Public surface:

* :class:`WorkflowOrchestrator` (base) — generic ``run()`` loop
* :func:`register_workflow` / :func:`get_orchestrator` — workflow registry
* concrete ``SmearWorkflowOrchestrator`` is in :mod:`.smear`
* concrete ``ConvergenceWorkflowOrchestrator`` is in :mod:`.convergence`
* concrete ``MagmomWorkflowOrchestrator`` is in :mod:`.magmom`
* concrete ``BanddosWorkflowOrchestrator`` is in :mod:`.banddos`
* concrete ``RelaxWorkflowOrchestrator`` is in :mod:`.relax`
* concrete ``PhonopyWorkflowOrchestrator`` is in :mod:`.phonopy`
* concrete ``EosWorkflowOrchestrator`` is in :mod:`.eos`
* concrete ``DefectsWorkflowOrchestrator`` is in :mod:`.defects`
"""

from . import convergence  # noqa: F401  -- triggers self-registration
from . import smear  # noqa: F401  -- triggers self-registration
from . import magmom  # noqa: F401  -- triggers self-registration
from . import banddos  # noqa: F401  -- triggers self-registration
from . import relax  # noqa: F401  -- triggers self-registration
from . import elastic  # noqa: F401  -- triggers self-registration
from . import phonopy  # noqa: F401  -- triggers self-registration
from . import eos  # noqa: F401  -- triggers self-registration
from . import defects  # noqa: F401  -- triggers self-registration
from . import supercell  # noqa: F401  -- triggers self-registration
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
