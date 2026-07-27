"""Shared helpers used by *every* workchain (smear, plus future ones).

Anything in this module must be backend-agnostic and reusable.

AiiDA-coupled types (``AdaptedInputs``, ``SoftwareAdapter``) live in
:mod:`aiida_uranium_workflow.input_builders`.
"""

from __future__ import annotations

from dataclasses import dataclass
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
      * ``metadata``        — ``static/metadata.yml`` entry.
    """

    input_params: dict[str, Any]
    protocol: dict[str, Any]
    workflow_data: dict[str, Any]
    software_params: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
