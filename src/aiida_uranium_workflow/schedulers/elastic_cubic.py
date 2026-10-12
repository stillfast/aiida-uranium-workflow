"""Scheduler for the elastic_cubic workflow (energy–strain method).

Binds the workflow to:

* ``parameters/elastic_cubic.yml`` (protocol — optional equilibrium
  ``B`` / ``V0`` / ``E0`` and crystal-system marker per preset; leaving
  the equilibrium out triggers the 7-point EOS fallback inside the
  WorkChain)
* ``parameters/abacus/scf.yml`` (per-backend SCF base, shared with the
  other abacus workflows)
* the orchestrator below.

The method is cubic-only by design.
"""

from __future__ import annotations

from typing import Any, Dict

from aiida_uranium_workflow.input_builders.elastic_cubic import (
    AbacusCubicElasticAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)


def parse_elastic_cubic_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Forward the preset verbatim as ``workflow_data['elastic_cubic']``.

    The adapter reads ``workflow_data['elastic_cubic']['equilibrium']``
    (bulk_modulus / volume_gs / energy_gs_ev), ``crystal_system``,
    ``eos_parameters`` and ``deltas`` from it; a preset without an
    ``equilibrium`` block triggers the EOS fallback.
    """
    return {"elastic_cubic": dict(protocol or {})}


class ElasticCubicWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the cubic elastic-constant workflow (ABACUS)."""

    ADAPTERS = {
        "abacus": AbacusCubicElasticAdapter,
    }

    BACKENDS = ("abacus",)

    #: Per-backend SCF base preset files.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
    }


register_workflow(
    name="elastic_cubic",
    protocol_file="elastic_cubic.yml",
    parser_hook=parse_elastic_cubic_protocol,
    orchestrator_cls=ElasticCubicWorkflowOrchestrator,
)
