"""Shared helpers used by *every* workchain (smear, plus future ones).

Anything in this module must be backend-agnostic and reusable.

AiiDA-coupled types (``AdaptedInputs``, ``SoftwareAdapter``) live in
:mod:`aiida_uranium_workflow.input_builders`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Package-level path constants (resolves to the installed source tree).
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
PKG_ROOT = _HERE.parent
PARAMETERS_DIR = PKG_ROOT / "parameters"
PROTOCOL_DIR = PARAMETERS_DIR  # protocol YAMLs live next to backend params
STATIC_DIR = PKG_ROOT / "static"


# ---------------------------------------------------------------------------
# Generic config container
# ---------------------------------------------------------------------------


@dataclass
class ParamBundle:
    """A single section of an input JSON + its associated YAML tables.

    Holds:
      * ``input_params``    — the raw JSON the user passed.
      * ``protocol``        — the ``protocol/<name>`` YAML section.
      * ``workflow_data``   — workflow-specific data parsed from protocol.
      * ``software_params`` — backend → list of loaded YAML entries (one per
        requested preset, see :func:`ConfigLoader._load_all_software_params`).
      * ``software_preset_names`` — backend → preset names, index-aligned
        with ``software_params`` (entry ``i`` was requested under name
        ``software_preset_names[backend][i]``). The resolved entries are
        plain YAML sections that do not carry their own name, so the
        orchestrator / ``check`` CLI need this parallel list to label
        each submission.
      * ``metadata``        — ``static/metadata.yml`` entry.
      * ``workflow_presets`` / ``workflow_data_map`` — when
        ``parameters[<workflow_key>]`` is a *list* of protocol preset names,
        one WorkChain is submitted per preset, each with its own parsed
        ``workflow_data`` (``workflow_data_map[preset]``). Empty for the
        single-preset / protocol-free layouts.
    """

    input_params: dict[str, Any]
    protocol: dict[str, Any]
    workflow_data: dict[str, Any]
    software_params: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
    software_preset_names: dict[str, list[str]] = field(default_factory=dict)
    workflow_presets: list[str] = field(default_factory=list)
    workflow_data_map: dict[str, dict[str, Any]] = field(default_factory=dict)
