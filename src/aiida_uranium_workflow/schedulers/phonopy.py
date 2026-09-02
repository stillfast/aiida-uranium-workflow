"""Scheduler for the phonopy workflow.

Binds the phonopy workflow to:

* ``parameters/phonopy.yml`` (protocol — phonopy-specific settings per
  preset: supercell / primitive matrix, band-path mode, DOS, ...)
* ``parameters/abacus/scf.yml`` (per-backend ABACUS SCF base, shared
  with banddos / elastic / relax)
* a parser hook that forwards the protocol section into
  ``workflow_data``
* the orchestrator below

Like banddos / elastic / relax, phonopy is **not** a parameter sweep —
one ``AbacusPhonopyWorkChain`` is submitted per (backend, preset).
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.phonopy import (
    AbacusPhonopyAdapter,
    FleurPhonopyAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_phonopy_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Forward the phonopy.yml preset verbatim as ``workflow_data['phonopy']``.

    The preset is the phonopy-specific block (supercell_matrix /
    primitive_matrix / band_mode / band_paths / dos / mesh / ...); the
    ABACUS SCF base is picked by the orchestrator from
    ``parameters/abacus/scf.yml`` via ``PRESET_SUBKEYS``.
    """
    return {"phonopy": dict(protocol or {})}


class PhonopyWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the phonopy workflow (ABACUS / FLEUR backends)."""

    ADAPTERS = {
        "abacus": AbacusPhonopyAdapter,
        "fleur": FleurPhonopyAdapter,
    }

    BACKENDS = ("abacus", "fleur")

    #: The per-backend SCF presets live in ``parameters/<backend>/scf.yml``
    #: (same files as banddos / elastic / relax / defects).
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur": "scf",
    }


register_workflow(
    name="phonopy",
    protocol_file="phonopy.yml",
    parser_hook=parse_phonopy_protocol,
    orchestrator_cls=PhonopyWorkflowOrchestrator,
)
