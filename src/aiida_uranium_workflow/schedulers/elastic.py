"""Scheduler for the elastic workflow.

Binds the elastic workflow to:

* ``parameters/elastic.yml`` (protocol — per preset, an ``abacus``, a
  ``vasp`` and a ``fleur`` block with the strain magnitudes)
* ``parameters/abacus/scf.yml`` / ``parameters/fleur/scf.yml``
  (shared per-backend SCF base) and ``parameters/vasp/elastic.yml``
  (self-contained VASP INCAR presets)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

Like banddos / relax, elastic is **not** a parameter sweep — one
elastic WorkChain is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.elastic import (
    AbacusElasticAdapter,
    FleurElasticAdapter,
    QeElasticAdapter,
    VaspElasticAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_elastic_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks."""
    if not protocol:
        return {}
    return {
        "abacus": dict(protocol.get("abacus", {})),
        "vasp": dict(protocol.get("vasp", {})),
        "fleur": dict(protocol.get("fleur", {})),
        "qe": dict(protocol.get("qe", {})),
    }


class ElasticWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the elastic workflow."""

    ADAPTERS = {
        "abacus": AbacusElasticAdapter,
        "vasp": VaspElasticAdapter,
        "fleur": FleurElasticAdapter,
        "qe": QeElasticAdapter,
    }

    BACKENDS = ("abacus", "vasp", "fleur", "qe")

    #: ``parameters/<backend>/<subkey>`` preset lookup for each backend:
    #: abacus / fleur share the SCF base (``scf``), vasp / qe presets
    #: are self-contained (``elastic``).
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "vasp": "elastic",
        "fleur": "scf",
        "qe": "scf",
    }


register_workflow(
    name="elastic",
    protocol_file="elastic.yml",
    parser_hook=parse_elastic_protocol,
    orchestrator_cls=ElasticWorkflowOrchestrator,
)
