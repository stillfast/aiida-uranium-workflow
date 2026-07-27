"""Scheduler for direct ABACUS and VASP base WorkChains."""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.base_workchain import (
    AbacusBaseWorkChainAdapter,
    VaspBaseWorkChainAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    WorkflowOrchestrator,
    register_workflow,
)


class BaseWorkChainOrchestrator(WorkflowOrchestrator):
    """Submit plugin base WorkChains without a custom parent WorkChain."""

    ADAPTERS = {
        "abacus": AbacusBaseWorkChainAdapter,
        "vasp": VaspBaseWorkChainAdapter,
    }
    BACKENDS = ("abacus", "vasp")
    PRESET_SUBKEYS = {"abacus": "abacus", "vasp": "vasp"}


register_workflow(
    name="base",
    orchestrator_cls=BaseWorkChainOrchestrator,
)
