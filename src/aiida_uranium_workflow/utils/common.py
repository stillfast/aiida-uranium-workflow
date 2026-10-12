"""Shared helpers used by *every* workchain (smear, plus future ones).

Anything in this module must be backend-agnostic and reusable.

AiiDA-coupled types (``AdaptedInputs``, ``SoftwareAdapter``) live in
:mod:`aiida_uranium_workflow.input_builders`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Data root: where ``parameters/`` and ``static/`` live
# ---------------------------------------------------------------------------
#
# The preset tables (``parameters/``) and the site tables (``static/``:
# queue names, wall-times, structure registry) change with the *site* and
# the *project*, not with the library code — so they are resolved at run
# time instead of being hard-wired to the installed package:
#
#   1. an explicit override (``aiida-uranium run --data-dir DIR`` /
#      ``ConfigLoader(..., data_root=DIR)``);
#   2. ``$AIIDA_URANIUM_DATA``;
#   3. the nearest ancestor of the ``input.json`` that carries a
#      ``parameters/`` directory (a project keeps its own data root, e.g.
#      ``aiida-uranium-scripts/``);
#   4. the set shipped inside the installed package — a **frozen
#      fallback snapshot**, not the place to edit presets.
#
# Exactly one root is used per run (no layering): pick it, then read both
# ``parameters/`` and ``static/`` from it.  A root that provides only one
# of the two fails loudly instead of silently mixing in the packaged half.

#: Environment variable naming a data root (overridden by ``--data-dir``).
DATA_ROOT_ENV = "AIIDA_URANIUM_DATA"

#: Sub-directory names a data root must provide.
PARAMETERS_SUBDIR = "parameters"
STATIC_SUBDIR = "static"

_HERE = Path(__file__).resolve().parent
PKG_ROOT = _HERE.parent

#: The data root shipped inside the package (fallback / public snapshot).
DEFAULT_DATA_ROOT = PKG_ROOT

#: Packaged paths — kept as module constants for backwards compatibility
#: (tests and third-party code import them).  New code should go through
#: :func:`resolve_data_root` instead of reading these directly.
PARAMETERS_DIR = PKG_ROOT / PARAMETERS_SUBDIR
PROTOCOL_DIR = PARAMETERS_DIR  # protocol YAMLs live next to backend params
STATIC_DIR = PKG_ROOT / STATIC_SUBDIR


@dataclass(frozen=True)
class DataRoot:
    """One resolved data root: ``<path>/parameters`` + ``<path>/static``.

    ``source`` records *why* this root won, so ``aiida-uranium check`` (and
    any bug report) can show it: ``"cli"``, ``"env:AIIDA_URANIUM_DATA"``,
    ``"input-json:<dir>"`` or ``"package-default"``.
    """

    path: Path
    source: str

    @property
    def parameters_dir(self) -> Path:
        return self.path / PARAMETERS_SUBDIR

    @property
    def protocol_dir(self) -> Path:
        """Protocol YAMLs live next to the backend preset tables."""
        return self.parameters_dir

    @property
    def static_dir(self) -> Path:
        return self.path / STATIC_SUBDIR

    def missing_parts(self) -> list[str]:
        """Return the required sub-directories this root does not provide."""
        return [
            name
            for name in (PARAMETERS_SUBDIR, STATIC_SUBDIR)
            if not (self.path / name).is_dir()
        ]

    def describe(self) -> str:
        return f"{self.path} (source: {self.source})"


def iter_candidate_roots(start: Path) -> Iterator[Path]:
    """Yield ``start`` and every ancestor that could be a data root.

    ``start`` may be a file (its directory is used) or a directory.
    """
    current = start if start.is_dir() else start.parent
    yield from (current, *current.parents)


def find_data_root(start: str | Path) -> Path | None:
    """Return the nearest ancestor of ``start`` that carries ``parameters/``.

    ``None`` when no ancestor qualifies — the caller then decides between
    the packaged fallback and an error.
    """
    for candidate in iter_candidate_roots(Path(start).resolve()):
        if (candidate / PARAMETERS_SUBDIR).is_dir():
            return candidate
    return None


def _snapshot_files(root: Path) -> dict[str, Path]:
    """Map ``"<subdir>/<relpath>"`` → path for every table under ``root``."""
    found: dict[str, Path] = {}
    for subdir in (PARAMETERS_SUBDIR, STATIC_SUBDIR):
        base = root / subdir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in (".yml", ".yaml", ".json"):
                found[f"{subdir}/{path.relative_to(base).as_posix()}"] = path
    return found


def snapshot_drift(
    data_root: "DataRoot | str | Path",
    packaged: str | Path | None = None,
) -> dict[str, list[str]]:
    """Compare the active data root with the packaged snapshot.

    The package ships a **frozen** copy of ``parameters/`` + ``static/``;
    the active data root is a *different* directory that the user owns.
    Editing the packaged copy therefore has no effect — the classic silent
    mistake this function exists to catch.

    Returns
    -------
    dict
        * ``"newer_in_packaged"`` — files whose packaged copy is newer than
          the data-root copy: the signature of "edited the wrong copy".
        * ``"only_in_packaged"`` / ``"only_in_root"`` — files present on
          one side only (informational; the two roots are allowed to carry
          different preset sets).
    """
    root = data_root.path if isinstance(data_root, DataRoot) else Path(data_root)
    packaged_root = (
        Path(packaged) if packaged is not None else DEFAULT_DATA_ROOT
    )

    active = _snapshot_files(root)
    frozen = _snapshot_files(packaged_root)

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:  # pragma: no cover — racing deletion
            return 0.0

    newer_in_packaged = sorted(
        name
        for name, frozen_path in frozen.items()
        if name in active and mtime(frozen_path) > mtime(active[name])
    )
    return {
        "newer_in_packaged": newer_in_packaged,
        "only_in_packaged": sorted(set(frozen) - set(active)),
        "only_in_root": sorted(set(active) - set(frozen)),
    }


def resolve_data_root(
    data_root: str | Path | None = None,
    input_json: str | Path | None = None,
    *,
    warn_on_fallback: bool = True,
) -> DataRoot:
    """Resolve the data root to use for one run.

    Resolution order (first hit wins, see the module note above):

    1. ``data_root`` — explicit argument (``--data-dir`` / API);
    2. ``$AIIDA_URANIUM_DATA``;
    3. the nearest ancestor of ``input_json`` holding ``parameters/``;
    4. the packaged fallback snapshot.

    When a root outside the package is selected, a warning is emitted if
    any *packaged* table is newer than its counterpart there — the
    signature of having edited the frozen copy by mistake (see
    :func:`snapshot_drift`).

    Raises
    ------
    FileNotFoundError
        The explicitly requested root does not exist.
    ValueError
        The selected root is missing ``parameters/`` or ``static/`` — a
        half-populated root is refused rather than silently completed from
        the packaged data.
    """
    import os
    import warnings

    root: Path | None = None
    source = ""

    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        source = "cli"
        if not root.is_dir():
            raise FileNotFoundError(
                f"data root {root} (from --data-dir / data_root=) is not a directory"
            )
    else:
        from_env = os.environ.get(DATA_ROOT_ENV)
        if from_env:
            root = Path(from_env).expanduser().resolve()
            source = f"env:{DATA_ROOT_ENV}"
            if not root.is_dir():
                raise FileNotFoundError(
                    f"data root {root} (from ${DATA_ROOT_ENV}) is not a directory"
                )
        elif input_json is not None:
            found = find_data_root(input_json)
            if found is not None:
                root = found
                source = f"input-json:{found}"

    if root is None:
        resolved = DataRoot(DEFAULT_DATA_ROOT, "package-default")
        if warn_on_fallback and input_json is not None:
            warnings.warn(
                f"no data root next to {input_json}; using the packaged preset "
                f"snapshot at {DEFAULT_DATA_ROOT} (pass --data-dir DIR or set "
                f"${DATA_ROOT_ENV} to use your own parameters/ + static/)",
                stacklevel=2,
            )
        missing = resolved.missing_parts()
        if missing:
            raise ValueError(
                f"the packaged data root {resolved.path} is missing "
                f"{', '.join(missing)}/ — the installation looks broken"
            )
        return resolved

    resolved = DataRoot(root, source)
    missing = resolved.missing_parts()
    if missing:
        raise ValueError(
            f"data root {root} (source: {source}) provides "
            f"{PARAMETERS_SUBDIR}/ but no {', '.join(missing)}/ — a data root must "
            f"carry both. Copy the missing director{'y' if len(missing) == 1 else 'ies'} "
            f"next to '{PARAMETERS_SUBDIR}/', or point --data-dir at a complete root."
        )

    if warn_on_fallback and resolved.path.resolve() != DEFAULT_DATA_ROOT.resolve():
        newer = snapshot_drift(resolved)["newer_in_packaged"]
        if newer:
            preview = ", ".join(newer[:3]) + (" …" if len(newer) > 3 else "")
            warnings.warn(
                f"{len(newer)} packaged preset file(s) are NEWER than the active "
                f"data root {resolved.path} ({preview}). Edits made in the "
                f"package copy have no effect — edit "
                f"{resolved.path}/{'parameters'} instead.",
                stacklevel=2,
            )

    return resolved


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
      * ``data_root``       — the :class:`DataRoot` every table above was
        read from (``None`` only for hand-built bundles).
    """

    input_params: dict[str, Any]
    protocol: dict[str, Any]
    workflow_data: dict[str, Any]
    software_params: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
    software_preset_names: dict[str, list[str]] = field(default_factory=dict)
    workflow_presets: list[str] = field(default_factory=list)
    workflow_data_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_root: DataRoot | None = None
