"""Scheduler for the magmom workflow.

Binds the magmom workflow to:

* ``parameters/magmom.yml`` (protocol section)
* a parser hook that extracts ``mag_list`` (abacus) /
  ``magmom_mapping_list`` / ``magmom_per_atom_list`` (vasp) /
  ``magmom_list`` (fleur)
* the orchestrator below

Once registered (``register_workflow(...)`` at import time), the
generic :func:`schedulers.get_orchestrator` can find it.
"""

from __future__ import annotations

from aiida_uranium_workflow.input_builders.magmom.abacus import (
    AbacusMagmomAdapter,
)
from aiida_uranium_workflow.input_builders.magmom.vasp import (
    VaspMagmomAdapter,
)
from aiida_uranium_workflow.input_builders.magmom.fleur import (
    FleurMagmomAdapter,
)
from aiida_uranium_workflow.input_builders.magmom.qe import (
    QeMagmomAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


def parse_magmom_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the per-backend magmom lists from the protocol section.

    Layout of ``magmom.yml``::

        test:
          abacus:
            mag_list:
              - [0.0, 0.0]
              - [1.0, 1.0]
          vasp:
            magmom_mapping_list:
              - {"U": 0.0}
              - {"U": 1.0}
          fleur:
            magmom_list:
              - {"label": "NM", "bmu": 0.0}
              - {"label": "FM", "bmu": 4.0,
                 "inpxml_changes": [["set_species", {...}]]}

        test_mag:
          vasp:
            magmom_per_atom_list:
              - [0.0, 0.0]
              - [4.0, -4.0]

    ``magmom_mapping_list`` and ``magmom_per_atom_list`` are mutually
    exclusive per vasp preset; at least one of them must be non-empty.
    """
    magmom_lists: Dict[str, Any] = {}
    if "abacus" in protocol:
        abacus_protocol = protocol["abacus"]
        mag_list = list(abacus_protocol.get("mag_list", []))
        if not mag_list:
            raise ValueError(
                "Protocol abacus mag_list must be non-empty"
            )
        magmom_lists["abacus"] = {"mag_list": mag_list}
    if "vasp" in protocol:
        vasp_protocol = protocol["vasp"]
        magmom_mapping_list = list(vasp_protocol.get("magmom_mapping_list", []))
        magmom_per_atom_list = list(vasp_protocol.get("magmom_per_atom_list", []))
        if not magmom_mapping_list and not magmom_per_atom_list:
            raise ValueError(
                "Protocol vasp magmom_mapping_list / magmom_per_atom_list "
                "must be non-empty"
            )
        magmom_lists["vasp"] = {
            "magmom_mapping_list": magmom_mapping_list,
            "magmom_per_atom_list": magmom_per_atom_list,
        }
    if "fleur" in protocol:
        fleur_protocol = protocol["fleur"]
        magmom_list = list(fleur_protocol.get("magmom_list", []))
        if not magmom_list:
            raise ValueError(
                "Protocol fleur magmom_list must be non-empty"
            )
        magmom_lists["fleur"] = {"magmom_list": magmom_list}
    if "qe" in protocol:
        qe_protocol = protocol["qe"]
        magmom_list = list(qe_protocol.get("magmom_list", []))
        if not magmom_list:
            raise ValueError("Protocol qe magmom_list must be non-empty")
        magmom_lists["qe"] = {"magmom_list": magmom_list}
    return {"magmom_lists": magmom_lists}


class MagmomWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the magmom workflow."""

    ADAPTERS = {
        "abacus": AbacusMagmomAdapter,
        "vasp": VaspMagmomAdapter,
        "fleur": FleurMagmomAdapter,
        "qe": QeMagmomAdapter,
    }

    BACKENDS = ("abacus", "vasp", "fleur", "qe")

    #: Magmom's ``input.json`` puts preset names under the
    #: ``"magmom"`` sub-key of each backend block, e.g.::
    #:
    #:     "parameters": {"abacus": {"magmom": ["test_magmom", "test_magmom_pw"]}}
    #:
    #: Without this mapping, the orchestrator falls back to synthetic
    #: ``abacus#0/#1`` identifiers in the output json; with it, the
    #: preset names (``"test_magmom"`` / ``"test_magmom_pw"`` / ...) are
    #: recorded verbatim.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "magmom",
        "vasp": "magmom",
        "fleur": "magmom",
        "qe": "magmom",
    }


register_workflow(
    name="magmom",
    protocol_file="magmom.yml",
    parser_hook=parse_magmom_protocol,
    orchestrator_cls=MagmomWorkflowOrchestrator,
)