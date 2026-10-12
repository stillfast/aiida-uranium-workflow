"""Scheduler for the relax workflow.

Binds the relax workflow to:

* ``parameters/relax.yml`` (protocol — per preset, an ``abacus`` block
  with the volume-relax settings and a ``fleur`` block with the EOS
  volume-scan settings; the same per-backend layout as magmom.yml)
* ``parameters/abacus/scf.yml`` / ``parameters/fleur/scf.yml``
  (per-backend SCF base — the physical / SCF parameters only)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

Like banddos, relax is **not** a parameter sweep — one relax WorkChain
is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.relax import (
    AbacusRelaxAdapter,
    FleurRelaxAdapter,
    QeRelaxAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_relax_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks.

    The ``parameters/relax.yml`` file uses the same per-backend layout
    as ``parameters/magmom.yml``::

        test:
          abacus:
            relax_type: volume
            force_cutoff: 0.03
            meta_convergence: true
            ...
          fleur:
            eos:
              points: 9
              step: 0.005
              guess: 1.00

    Both blocks are forwarded verbatim to the respective adapters.
    """
    if not protocol:
        return {}
    return {
        "abacus": dict(protocol.get("abacus", {})),
        "fleur": dict(protocol.get("fleur", {})),
        "qe": dict(protocol.get("qe", {})),
    }


class RelaxWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the relax workflow."""

    ADAPTERS = {
        "abacus": AbacusRelaxAdapter,
        "fleur": FleurRelaxAdapter,
        "qe": QeRelaxAdapter,
    }

    BACKENDS = ("abacus", "fleur", "qe")

    #: Per-backend preset-subkey mapping. ``"abacus": {"scf": [...]}``
    #: picks the ``parameters/abacus/scf.yml`` preset; ``"fleur":
    #: {"scf": "..."}`` picks ``parameters/fleur/scf.yml``; ``"qe":
    #: {"relax": [...]}`` picks the self-contained
    #: ``parameters/qe/relax.yml`` presets.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur": "scf",
        "qe": "scf",
    }


register_workflow(
    name="relax",
    protocol_file="relax.yml",
    parser_hook=parse_relax_protocol,
    orchestrator_cls=RelaxWorkflowOrchestrator,
)
