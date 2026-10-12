"""Shared helpers used by *every* workchain (smear, plus future ones).

Anything in this module must be backend-agnostic and reusable.

AiiDA-coupled types (``AdaptedInputs``, ``SoftwareAdapter``) live in
:mod:`aiida_uranium_workflow.input_builders`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

# ---------------------------------------------------------------------------
# Package-level path constants (resolves to the installed source tree).
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
PKG_ROOT = _HERE.parent

#: The package's own copies — the fallback for every lookup.
PKG_PARAMETERS_DIR = PKG_ROOT / "parameters"
PKG_PROTOCOL_DIR = PKG_PARAMETERS_DIR  # protocol YAMLs sit next to backend params
PKG_STATIC_DIR = PKG_ROOT / "static"

#: Backwards-compatible aliases: the *package* copies.  Runtime reads go
#: through :func:`config_path`, which may resolve elsewhere (see below).
PARAMETERS_DIR = PKG_PARAMETERS_DIR
PROTOCOL_DIR = PKG_PROTOCOL_DIR
STATIC_DIR = PKG_STATIC_DIR


# ---------------------------------------------------------------------------
# Config root: ``parameters/`` + ``static/`` are per scripts tree
# ---------------------------------------------------------------------------
#
# A scripts tree may carry its own ``parameters/`` and ``static/`` (e.g.
# ``aiida-uranium-scripts-v14/parameters`` holds the 14-valence-electron
# presets).  The active root is resolved in this order:
#
#   1. :func:`set_config_root` — the CLI's ``--config-root DIR``
#   2. ``$AIIDA_URANIUM_CONFIG_ROOT``
#   3. discovery: walk up from the ``input.json`` (or the CWD) until a
#      directory that contains ``parameters/`` or ``static/`` is found
#   4. the package itself (default — behaviour is unchanged without 1-3)
#
# Resolution is **per file**: a file missing from the override root falls
# back to the package copy, so a scripts tree only has to carry what it
# changes (a full copy works too).

CONFIG_ROOT_ENV = "AIIDA_URANIUM_CONFIG_ROOT"

_config_root: Path | None = None
_config_root_source: str = "package"
_config_root_from_env: bool = False


def find_config_root(start: "str | Path") -> Path | None:
    """Nearest ancestor of ``start`` that carries ``parameters/`` or ``static/``.

    This is the scripts-tree layout (``<tree>/parameters/...``,
    ``<tree>/static/...``); returns ``None`` when there is none.
    """
    try:
        path = Path(start).expanduser().resolve()
    except OSError:  # pragma: no cover — ELOOP / permission
        return None
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "parameters").is_dir() or (candidate / "static").is_dir():
            return candidate
    return None


def set_config_root(root: "str | Path | None" = None, *, source: str | None = None) -> Path | None:
    """Pin the config root; ``None`` re-resolves from the environment.

    Returns the root that is now active (``None`` = package copies).
    """
    global _config_root, _config_root_source, _config_root_from_env

    if root is None:
        env = os.environ.get(CONFIG_ROOT_ENV)
        if env:
            _config_root = Path(env).expanduser().resolve()
            _config_root_source = f"${CONFIG_ROOT_ENV}"
        else:
            _config_root = None
            _config_root_source = "package default"
        _config_root_from_env = True
        return _config_root

    _config_root = Path(root).expanduser().resolve()
    _config_root_source = source or "--config-root"
    _config_root_from_env = True
    return _config_root


def config_root() -> Path | None:
    """The active override root, or ``None`` when the package copies are used."""
    if not _config_root_from_env:
        set_config_root(None)  # pick up $AIIDA_URANIUM_CONFIG_ROOT lazily
    return _config_root


def config_root_source() -> str:
    """Human-readable explanation of where the active root came from."""
    config_root()
    return _config_root_source


def config_path(*parts: str) -> Path:
    """Resolve ``parts`` under the active root, falling back to the package.

    Per-file: a file that does not exist in the override root resolves to
    the package copy, so a scripts tree only carries what it changes.
    """
    root = config_root()
    if root is not None:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return PKG_ROOT.joinpath(*parts)


def config_root_label() -> str:
    """One-line description of the active root, for CLI output."""
    root = config_root()
    if root is None:
        return f"{PKG_ROOT} (package default)"
    return f"{root} ({config_root_source()})"


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
