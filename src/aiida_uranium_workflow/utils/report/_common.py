"""Shared rendering primitives for the smear / convergence / magmom reports.

Three report modules (``smear``, ``convergence``, ``magmom``) all need
the same handful of building blocks — a 2-D grid renderer for
``smear`` × ``sigma`` and ``ecutwfc`` × ``kpoints`` sweeps, a per-child
single-column renderer for magmom, plus the report header/footer
boilerplate. This module centralises those primitives so each report
module only has to declare its own axis layout and call the
appropriate renderer.

Public surface
--------------
* :class:`AxisSpec` — declarative description of one axis in a 2-D grid.
* :func:`parse_axes` — split a workflow-generated label string into
  ``(row_value, col_value, ...)`` according to a sequence of AxisSpec.
* :func:`format_scalar` — uniform ``None`` / ``str`` / numeric formatting
  (``"—"`` for missing, ``"%.Nf"`` for floats).
* :func:`sort_axis_values` — natural sort for mesh strings (``"11x11x11"``)
  while keeping plain floats ordered.
* :func:`render_2d_grid` — generic 2-D Markdown table.
* :func:`render_per_child_table` — generic per-child single-column table.
* :func:`render_report_header` / :func:`render_report_footer` — boilerplate.

Backward compatibility
----------------------
The three public ``generate_*`` entry points in each report module keep
their existing signatures; this module is a refactor of the *internals*
they delegate to. The 344 lines of ``tests/test_report_energy_time.py``
exercise those public functions end-to-end and must continue to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# AxisSpec + label parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisSpec:
    """Declarative description of one axis in a 2-D sweep label.

    Attributes:
        name: Display name (used in error messages / column headers).
        keyword: The primary token that marks the start of this axis in
            the label string. The parser locates it via
            ``parts.index(keyword)``; the value spans from one past that
            token up to the next axis's keyword (or end-of-label).
        keyword_aliases: Additional tokens accepted as the start of
            this axis. The first alias found in the label wins, so the
            caller can describe backend-specific naming variants
            (e.g. ``"ecutwfc"`` (ABACUS) and ``"encut"`` (VASP) for the
            energy-cutoff axis) under a single AxisSpec.
        kind: ``"string"`` (smear method), ``"float"`` (ecutwfc / sigma
            in non-mesh mode), or ``"auto"`` (numeric when the chunk
            parses as a float, otherwise a string with ``_`` → ``.``).
        qualifiers: Tokens to skip immediately after ``keyword`` because
            they disambiguate the axis mode rather than carry a value.
            For convergence, ``"kpoints"`` may be followed by
            ``"spacing"`` or ``"distance"`` to mark the spacing-mode
            value; mesh-mode labels (e.g. ``kpoints_11x11x11``) have no
            qualifier. The default is an empty tuple.
        unit: Optional unit annotation rendered in the column header
            (e.g. ``"[eV]"``).
    """

    name: str
    keyword: str
    keyword_aliases: Tuple[str, ...] = ()
    kind: str = "auto"  # "string" | "float" | "auto"
    qualifiers: Tuple[str, ...] = ()
    unit: str = ""


def _coerce_axis_value(raw: str, kind: str) -> Union[str, float, None]:
    """Convert one axis's raw text fragment into its typed value."""
    if kind == "string":
        return raw
    if kind == "float":
        try:
            return float(raw.replace("_", "."))
        except ValueError:
            return None
    # auto: prefer float, fall back to string.
    try:
        return float(raw.replace("_", "."))
    except ValueError:
        return raw


def parse_axes(
    label: str,
    axes: Sequence[AxisSpec],
) -> List[Union[str, float, None]]:
    """Parse a label of the form ``<kw1>_<v1>_<kw2>_<v2>_...`` into axis values.

    Strategy:

    1. Split on ``_``.
    2. Walk axes in declaration order. For each axis, find the *first*
       occurrence of any of its keywords that hasn't been claimed by an
       earlier axis.
    3. The value is everything between this keyword and the keyword
       belonging to the *next* axis in declaration order. The keyword of
       the *next* axis is pre-computed (so axis N doesn't have to wait
       for axis N+1 to run before knowing where it ends).
    4. If no such boundary exists, the value runs to the end of the
       label (this is the case for the *last* axis).

    Unparseable labels return ``[None, None, ...]`` so the caller can skip
    them. Axis values that themselves contain underscores (e.g. smear
    method ``mp`` followed by ``sigma``) are handled by the keyword
    boundary lookup.

    Examples::

        parse_axes(
            "ecutwfc_80_kpoints_distance_0_1",
            [
                AxisSpec("ecut", ("ecutwfc", "encut"), kind="float"),
                AxisSpec("kpoints", ("kpoints_spacing", "kpoints_distance", "kpoints_mesh"),
                         kind="auto"),
            ],
        )
        # → [80.0, 0.1]

        parse_axes(
            "smear_mp_sigma_0_02",
            [
                AxisSpec("smear", ("smear",), kind="string"),
                AxisSpec("sigma", ("sigma",), kind="float"),
            ],
        )
        # → ["mp", 0.02]
    """
    if not isinstance(label, str) or not axes:
        return [None] * len(axes)

    parts = label.split("_")

    # Pre-compute the keyword index for every axis.
    # ``kw_indices[axis_idx]`` is the parts-index of the chosen keyword,
    # or None when the axis has no matching keyword in this label.
    kw_indices: List[Optional[int]] = [None] * len(axes)
    for axis_idx, axis in enumerate(axes):
        # Try ``keyword`` first, then each alias in declaration order.
        # The first token found in ``parts`` wins, except tokens that
        # have already been claimed by an earlier axis.
        candidates: list[str] = [axis.keyword, *axis.keyword_aliases]
        chosen: Optional[int] = None
        for kw in candidates:
            try:
                idx = parts.index(kw)
            except ValueError:
                continue
            if any(kw_indices[i] == idx for i in range(axis_idx)):
                continue
            if chosen is None or idx < chosen:
                chosen = idx
        kw_indices[axis_idx] = chosen

    values: list[Union[str, float, None]] = [None] * len(axes)

    for axis_idx, axis in enumerate(axes):
        kw_idx = kw_indices[axis_idx]
        if kw_idx is None:
            continue

        # The value runs to the keyword of the *next* axis (in declaration
        # order) that has a keyword in this label, or to end-of-label
        # when no later axis matches.
        end = len(parts)
        for next_idx in range(axis_idx + 1, len(axes)):
            nxt = kw_indices[next_idx]
            if nxt is not None and nxt > kw_idx:
                end = nxt
                break

        if end - kw_idx < 2:
            # Keyword is the trailing token; no value follows.
            values[axis_idx] = None
            continue

        value_start = kw_idx + 1
        # Skip an optional disambiguating qualifier that immediately
        # follows the keyword (e.g. ``kpoints_spacing_0_2`` → skip
        # ``spacing``; ``kpoints_11x11x11`` has no qualifier).
        if (
            axis.qualifiers
            and value_start < end
            and parts[value_start] in axis.qualifiers
        ):
            value_start += 1

        if value_start >= end:
            values[axis_idx] = None
            continue

        raw = "_".join(parts[value_start:end])
        values[axis_idx] = _coerce_axis_value(raw, axis.kind)

    return values


# ---------------------------------------------------------------------------
# Sort key for axis values
# ---------------------------------------------------------------------------


def _mesh_key(value: Any) -> Tuple[Any, ...]:
    """Natural sort key for mesh strings like ``"11x11x11"``."""
    if isinstance(value, str) and "x" in value:
        try:
            return tuple(int(p) for p in value.split("x"))
        except ValueError:
            return (value,)
    return (value,)


def sort_axis_values(values: Iterable[Any]) -> List[Any]:
    """Sort a list of axis values, with mesh strings ordered by integer chunks.

    Mirrors :func:`utils.report.convergence._sort_kpoints_values` byte-for-byte
    so the existing convergence test fixture continues to pass.
    """
    values = list(values)
    if not values:
        return values
    try:
        return sorted(values, key=_mesh_key)
    except TypeError:
        return sorted(values)


# ---------------------------------------------------------------------------
# Scalar formatter
# ---------------------------------------------------------------------------


def format_scalar(
    value: Any,
    *,
    fmt: str = "%.6f",
    missing: str = "—",
) -> str:
    """Format a single scalar for inclusion in a Markdown table cell.

    Rules (mirrors the three sites this replaces):

    * ``None`` or non-numeric ``str`` → ``missing`` (``"—"`` by default).
    * numeric ``int`` / ``float`` → ``fmt % float(value)``.
    * anything else → ``str(value)``.
    """
    if value is None or isinstance(value, str):
        return missing
    try:
        return fmt % float(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# 2-D grid renderer
# ---------------------------------------------------------------------------


def render_2d_grid(
    data: dict,
    axes: Sequence[AxisSpec],
    *,
    row_header: str,
    col_header: str,
    cell_format: Callable[[Any], str] = format_scalar,
    empty_placeholder: str = "—",
) -> str:
    """Render a 2-D Markdown table from a ``{label: value}`` dict.

    Args:
        data: ``{label: value}`` — each label is parsed by
            :func:`parse_axes` against ``axes``.
        axes: exactly two :class:`AxisSpec` (row axis, then column axis).
        row_header: Header text for the row-axis column.
        col_header: Header text for the column-axis column.
        cell_format: per-cell formatter (default :func:`format_scalar`).
        empty_placeholder: what to render when no row×col intersection
            exists in ``data``.

    Returns:
        Markdown table as a string, or ``"No data available."`` when
        ``data`` is empty.
    """
    if not axes or len(axes) != 2:
        raise ValueError("render_2d_grid requires exactly two axes")

    if not data:
        return "No data available."

    # Bucket values by (row_value, col_value); unparseable labels drop out.
    buckets: dict[tuple, Any] = {}
    for label, value in data.items():
        values = parse_axes(label, list(axes))
        if values[0] is None or values[1] is None:
            continue
        buckets[(values[0], values[1])] = value

    if not buckets:
        return "No data available."

    row_values = sort_axis_values({k[0] for k in buckets})
    col_values = sort_axis_values({k[1] for k in buckets})

    header = (
        f"| {row_header}\\{col_header} | "
        + " | ".join(f"{c}" for c in col_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in col_values) + " |"

    rows: list[str] = []
    for row_value in row_values:
        cells = [f"| {row_value}"]
        for col_value in col_values:
            if (row_value, col_value) in buckets:
                cells.append(f"| {cell_format(buckets[(row_value, col_value)])}")
            else:
                cells.append(f"| {empty_placeholder}")
        cells.append("|")
        rows.append(" ".join(cells))

    return "\n".join([header, separator] + rows)


# ---------------------------------------------------------------------------
# Per-child single-column renderer (magmom-style)
# ---------------------------------------------------------------------------


def render_per_child_table(
    data: dict,
    *,
    column_header: str,
    column_name: str,
    cell_format: Callable[[Any], str] = format_scalar,
) -> str:
    """Render a per-child single-column Markdown table.

    The header row is::

        | child pk | <column_header> |
        | --- | --- |

    Each ``data`` entry becomes one row. Keys are rendered with ``str(key)``
    so both integer pks and string identifiers work.

    Args:
        data: ``{child_identifier: value}``.
        column_header: Header for the value column.
        column_name: Short name used in the ``child pk | <column_name> [s]``
            format — kept for API compatibility with the existing
            ``wall_time`` table that uses ``"wall_time [s]"``.
        cell_format: per-cell formatter (default :func:`format_scalar`).
    """
    if not data:
        return "No data available."

    header = f"| child pk | {column_header} |"
    separator = "| --- | --- |"
    rows: list[str] = []
    for key, value in data.items():
        rows.append(f"| {key} | {cell_format(value)} |")
    return "\n".join([header, separator] + rows)


# ---------------------------------------------------------------------------
# Report header / footer
# ---------------------------------------------------------------------------


def render_report_header(
    *,
    title: str,
    workflow_type: str,
    pk: int,
    timestamp: Optional[datetime] = None,
) -> str:
    """Render the standard report preamble (title + workflow + generated).

    Mirrors the pre-amble used by the three ``generate_report`` entry
    points in :mod:`.smear`, :mod:`.convergence`, :mod:`.magmom`.
    """
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# {title} (PK: {pk})\n"
        "\n"
        f"**Workflow Type**: {workflow_type.upper()}\n"
        f"**Generated**: {ts}\n"
    )


def render_report_footer() -> str:
    """Render the standard report tail."""
    return "---\n*Generated by aiida-uranium-workflow*"
