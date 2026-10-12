"""Scheduler for the smear workflow.

Binds the smear workflow to:

* ``parameters/smear.yml`` (protocol section)
* a parser hook that extracts each backend's own smear/sigma lists
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

#: Backends the smear protocol may carry a section for, mapped to the
#: keys that section must provide.  The key names mirror the backend's own
#: input keywords (ABACUS ``smearing_method`` / ``smearing_sigma``, VASP
#: INCAR ``sigma``) so a protocol section can be read next to the preset
#: it drives.
_SMEAR_PROTOCOL_KEYS: Dict[str, tuple[str, str]] = {
    "abacus": ("smearing_method_list", "smearing_sigma_list"),
    "vasp": ("smear_list", "sigma_list"),
}


def parse_smear_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the per-backend smear/sigma grids from the protocol section.

    Preferred layout of ``parameters/smear.yml`` — one section per backend,
    carrying *that backend's own* keywords and units::

        test_simple:
          abacus:
            smearing_method_list:      # ABACUS smearing_method keywords
              - mp
              - gauss
            smearing_sigma_list:       # the unit is Ry (ABACUS-native)
              - 0.0225
              - 0.0150
          vasp:
            smear_list:                # ISMEAR integers
              - 0
              - 1
            sigma_list:                # the unit is eV (VASP-native)
              - 0.3
              - 0.2

    Because every section is already backend-native, each adapter consumes
    its own section verbatim — no keyword mapping and no unit conversion
    happens on the way in.

    The legacy shared layout is still accepted so the ``test`` / ``gauss``
    protocol presets keep working::

        test:
          smear_list:      # canonical method names (gauss / mp / mp2)
            - gauss
            - mp
          sigma_list:      # the unit is eV (converted per backend)
            - 0.3
            - 0.1

    The two layouts are distinguished by the returned value: the native
    layout nests one entry per backend under ``smear_lists``, the legacy
    layout keeps the historical flat ``{"smear": …, "sigma": …}`` pair
    that the adapters translate themselves.
    """
    native: Dict[str, Any] = {}
    for backend, (smear_key, sigma_key) in _SMEAR_PROTOCOL_KEYS.items():
        section = protocol.get(backend)
        if not isinstance(section, dict):
            continue
        smear = list(section.get(smear_key, []))
        sigma = [float(s) for s in section.get(sigma_key, [])]
        if not smear or not sigma:
            raise ValueError(
                f"Protocol {backend} {smear_key} / {sigma_key} must be non-empty"
            )
        native[backend] = {smear_key: smear, sigma_key: sigma}

    if native:
        return {"smear_lists": native}

    # Legacy shared layout: canonical method names + sigma in eV.
    smear = list(protocol.get("smear_list", []))
    sigma = [float(s) for s in protocol.get("sigma_list", [])]
    if not smear or not sigma:
        raise ValueError(
            "Protocol must either carry a per-backend section "
            f"({', '.join(sorted(_SMEAR_PROTOCOL_KEYS))}) or a shared "
            "smear_list / sigma_list pair"
        )
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

    #: Smear's ``input.json`` names each backend's SCF presets by category::
    #:
    #:     "parameters": {"abacus": {"test": ["smear_lcao", "pw"]},
    #:                    "vasp":   {"test": "test_smear"}}
    #:
    #: The dict key is the *category* — it names the file the presets are
    #: read from (``parameters/abacus/test.yml``).  This mapping only pins
    #: the historical category (``"smear"``, used by the pre-refactor
    #: layout where each method had its own per-backend file); when the
    #: category differs, the base class's
    #: :meth:`_preset_names_for` picks the given category's preset names up
    #: anyway, so ``output.json`` still records real names instead of
    #: synthetic ``abacus#0`` identifiers.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "smear",
        "vasp": "smear",
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
