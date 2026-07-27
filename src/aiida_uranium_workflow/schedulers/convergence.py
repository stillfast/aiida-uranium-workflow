"""Scheduler for the convergence workflow.

Binds the convergence workflow to:

* ``parameters/convergence.yml`` (protocol section)
* a parser hook that extracts ``ecutwfc_list`` / ``kpoints_distance_list``
* the orchestrator below

Once registered (``register_workflow(...)`` at import time), the
generic :func:`schedulers.get_orchestrator` can find it.
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.convergence_abacus import (
    AbacusConvergenceAdapter,
)
from aiida_uranium_workflow.input_builders.convergence_vasp import (
    VaspConvergenceAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_convergence_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Extract convergence lists from the loaded protocol section.

    Supports both distance/spacing mode and mesh mode:
    - distance/spacing mode: kpoints_distance_list / kpoints_spacing_list
    - mesh mode: kpoints_mesh_list (list of [nx, ny, nz] tuples)

    Mesh mode takes priority if both are provided.
    """
    convergence_lists: Dict[str, Any] = {}
    if "abacus" in protocol:
        abacus_protocol = protocol["abacus"]
        ecutwfc_list = list(abacus_protocol.get("ecutwfc_list", []))

        kpoints_mesh_list = list(abacus_protocol.get("kpoints_mesh_list", []))
        if kpoints_mesh_list:
            kpoints_list = kpoints_mesh_list
            kpoints_key = "kpoints_mesh_list"
        else:
            kpoints_list = list(abacus_protocol.get("kpoints_distance_list", []))
            kpoints_key = "kpoints_distance_list"

        if not ecutwfc_list or not kpoints_list:
            raise ValueError(
                f"Protocol abacus ecutwfc_list / {kpoints_key} must be non-empty"
            )
        convergence_lists["abacus"] = {
            "ecutwfc_list": ecutwfc_list,
            kpoints_key: kpoints_list,
        }
    if "vasp" in protocol:
        vasp_protocol = protocol["vasp"]
        encut_list = list(vasp_protocol.get("encut_list", []))

        kpoints_mesh_list = list(vasp_protocol.get("kpoints_mesh_list", []))
        if kpoints_mesh_list:
            kpoints_list = kpoints_mesh_list
            kpoints_key = "kpoints_mesh_list"
        else:
            kpoints_list = list(vasp_protocol.get("kpoints_spacing_list", []))
            kpoints_key = "kpoints_spacing_list"

        if not encut_list or not kpoints_list:
            raise ValueError(
                f"Protocol vasp encut_list / {kpoints_key} must be non-empty"
            )
        convergence_lists["vasp"] = {
            "encut_list": encut_list,
            kpoints_key: kpoints_list,
        }
    return {"convergence_lists": convergence_lists}


class ConvergenceWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the convergence workflow."""

    ADAPTERS = {
        "abacus": AbacusConvergenceAdapter,
        "vasp": VaspConvergenceAdapter,
    }

    BACKENDS = ("abacus", "vasp")

    #: Convergence's ``input.json`` puts preset names under the
    #: ``"convergence"`` sub-key of each backend block, e.g.::
    #:
    #:     "parameters": {"abacus": {"convergence": ["pw", "pw_r"]}, "vasp": {"convergence": [...]}}
    #:
    #: Both backends share the same sub-key, so use the per-backend
    #: ``PRESET_SUBKEYS`` mapping so the output json records the actual
    #: preset names (``"pw"``/``"pw_r"``/...) rather than synthetic
    #: ``abacus#0/#1`` identifiers.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "convergence",
        "vasp": "convergence",
    }


register_workflow(
    name="convergence",
    protocol_file="convergence.yml",
    parser_hook=parse_convergence_protocol,
    orchestrator_cls=ConvergenceWorkflowOrchestrator,
)
