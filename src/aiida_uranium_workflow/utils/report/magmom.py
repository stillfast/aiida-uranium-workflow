"""Magmom WorkChain report generation module.

Extracts magnetism data from ``VaspMagmomWorkChain`` or
``AbacusMagmomWorkChain`` ``output_parameters`` and renders a
Markdown report.

The per-child single-column tables (``status`` / ``energy`` /
``wall_time``) and the report header / footer are shared with the
smear / convergence reports via :mod:`utils.report._common`. The
``generate_*`` entry points keep their original signatures so existing
callers and tests are unaffected.

Per-backend key layout produced by the magmom workflows:

* ABACUS (``AbacusMagmomWorkChain``)
    - ``magnetism``        — ``{pk: [<per-atom mag>, ...] | None}``
    - ``final_magnetism``  — ``{pk: <total magnetization> | None}``
    - ``nspin``            — ``{pk: <nspin value> | None}``
* VASP (``VaspMagmomWorkChain``)
    - ``magnetization``        — ``{pk: <total magnetization> | None}``
    - ``site_magnetization``   — ``{pk: <per-site magnetization> | None}``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from ._common import (
    format_scalar,
    render_per_child_table,
    render_report_footer,
    render_report_header,
)


# ---------------------------------------------------------------------------
# Value formatters (workflow-specific scalar coercion)
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Render a single magnetism value in a Markdown-friendly form."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        formatted = ", ".join(_format_value(v) for v in value)
        return f"[{formatted}]"
    return str(value)


def _coerce_scalar(value: Any) -> Optional[Union[float, List[float]]]:
    """Reduce a number/list/dict to a float or list of floats.

    ABACUS uses ``{"total_magnetism": ..., "absolute_magnetism": ...}``.
    VASP exposes ``magnetization`` either as a scalar, a single-element
    list ``[x]`` (total per axis), a multi-element list ``[x, y, z]``
    (vector components, e.g. for SOC runs), or a dict with a
    ``full_cell`` key.

    Returns:
        * ``float`` for scalars / single-element lists.
        * ``list[float]`` for multi-element lists (preserves every
          component).
        * ``None`` for anything that cannot be unambiguously converted.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("total_magnetism", "absolute_magnetism", "full_cell", "total"):
            if key in value:
                return _coerce_scalar(value[key])
        return None
    if isinstance(value, (list, tuple)):
        numeric: List[float] = []
        for x in value:
            coerced = _coerce_scalar(x)
            if isinstance(coerced, list):
                numeric.extend(coerced)
            elif coerced is not None:
                numeric.append(coerced)
        if not numeric:
            return None
        if len(numeric) == 1:
            return numeric[0]
        return numeric
    return None


def _resolve_final_magnetism(
    final_value: Any, site_value: Any
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(total, absolute)`` magnetization for a single child."""
    if isinstance(final_value, dict):
        total = _coerce_scalar(final_value.get("total_magnetism"))
        absolute = _coerce_scalar(final_value.get("absolute_magnetism"))
        if total is None:
            total = _coerce_scalar(final_value)
        if absolute is None:
            if isinstance(total, list):
                absolute = [abs(v) for v in total] if total else None
            else:
                absolute = abs(total) if total is not None else None
        return total, absolute

    total = _coerce_scalar(final_value)
    if total is None:
        return None, None
    if isinstance(total, list):
        return total, [abs(v) for v in total]
    return total, abs(total)


def _initial_mag_from_list_entry(entry: Any) -> str:
    """Render an entry from ``magmom_list`` as a human-readable initial
    magnetization string.

    For ABACUS each entry is a nested list of per-atom magnetization
    vectors (e.g. ``[[1.0], [1.0]]``); for VASP it is a mapping dict
    (e.g. ``{"U": 1.0}``).
    """
    if isinstance(entry, dict):
        parts = []
        for element, value in entry.items():
            if isinstance(value, (list, tuple)):
                v = ", ".join(f"{float(x):g}" for x in value)
            else:
                v = f"{float(value):g}"
            parts.append(f"{element}={v}")
        return "; ".join(parts)
    if isinstance(entry, (list, tuple)):
        per_atom = []
        for atom in entry:
            if isinstance(atom, (list, tuple)):
                per_atom.append(
                    "[" + ", ".join(f"{float(x):g}" for x in atom) + "]"
                )
            else:
                per_atom.append(f"{float(atom):g}")
        return "[" + ", ".join(per_atom) + "]"
    return str(entry)


# ---------------------------------------------------------------------------
# Backend-specific magnetism section
# ---------------------------------------------------------------------------


def _render_abacus_section(output_params: Dict[str, Any]) -> List[str]:
    """Render the ABACUS magnetism section."""
    lines: List[str] = []
    lines.append("## Magnetism")
    lines.append("")

    magnetism = output_params.get("magnetism", {}) or {}
    final_magnetism = output_params.get("final_magnetism", {}) or {}
    nspin = output_params.get("nspin", {}) or {}

    if not magnetism:
        lines.append("No magnetism data available.")
        return lines

    header = "| child pk | nspin | final_magnetism | magnetism (per atom) |"
    separator = "| --- | --- | --- | --- |"
    lines.append(header)
    lines.append(separator)

    for pk in magnetism:
        per_atom = magnetism.get(pk)
        total = final_magnetism.get(pk)
        spin = nspin.get(pk)
        lines.append(
            f"| {pk} | {spin if spin is not None else '—'} "
            f"| {_format_value(total)} | {_format_value(per_atom)} |"
        )

    return lines


def _render_vasp_section(output_params: Dict[str, Any]) -> List[str]:
    """Render the VASP magnetism section."""
    lines: List[str] = []
    lines.append("## Magnetism")
    lines.append("")

    magnetization = output_params.get("magnetization", {}) or {}
    site_magnetization = output_params.get("site_magnetization", {}) or {}

    if not magnetization:
        lines.append("No magnetization data available.")
        return lines

    header = "| child pk | magnetization | site_magnetization |"
    separator = "| --- | --- | --- |"
    lines.append(header)
    lines.append(separator)

    for pk in magnetization:
        total = magnetization.get(pk)
        per_site = site_magnetization.get(pk)
        lines.append(
            f"| {pk} | {_format_value(total)} | {_format_value(per_site)} |"
        )

    return lines


# ---------------------------------------------------------------------------
# Per-child single-column tables
# ---------------------------------------------------------------------------


def generate_status_table(status: Dict[str, int]) -> str:
    """Render the per-child exit-code status as a Markdown table."""
    if not status:
        return "No status data available."

    lines = ["| child pk | exit_status |", "| --- | --- |"]
    for label, exit_code in status.items():
        lines.append(f"| {label} | {exit_code} |")
    return "\n".join(lines)


def generate_energy_table(final_energy: Dict[Any, Any]) -> str:
    """Render a Markdown table of per-child final energy."""
    return render_per_child_table(
        final_energy,
        column_header="final_energy",
        column_name="final_energy",
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
    )


def generate_wall_time_table(wall_time_seconds: Dict[Any, Any]) -> str:
    """Render a Markdown table of per-child wall-clock time (seconds)."""
    return render_per_child_table(
        wall_time_seconds,
        column_header="wall_time [s]",
        column_name="wall_time",
        cell_format=lambda v: format_scalar(v, fmt="%.3f"),
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Generate a summary table with basic info."""
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    if "magmom_list" in output_params:
        lines.append(
            f"| magmom_list entries | {len(output_params['magmom_list'])} |"
        )

    backend_keys = {
        "abacus": ("magnetism", "final_magnetism"),
        "vasp": ("magnetization", "site_magnetization"),
    }
    for backend, (primary, secondary) in backend_keys.items():
        primary_data = output_params.get(primary, {}) or {}
        secondary_data = output_params.get(secondary, {}) or {}
        if primary_data:
            lines.append(f"| backend | {backend} |")
            lines.append(f"| {primary} entries | {len(primary_data)} |")
            if secondary_data:
                lines.append(f"| {secondary} entries | {len(secondary_data)} |")
            break

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Magmom convergence table
# ---------------------------------------------------------------------------


def generate_magmom_summary_table(
    output_params: Dict[str, Any],
    workflow_type: str = "abacus",
) -> str:
    """Generate a compact Markdown table for the magmom scan.

    Columns: ``initial magmom | final total magmom | final_energy``
    """
    magmom_list = output_params.get("magmom_list") or []
    final_energy = output_params.get("final_energy") or {}

    if workflow_type == "abacus":
        final_magnetism = output_params.get("final_magnetism") or {}
    else:
        final_magnetism = output_params.get("magnetization") or {}

    candidate_keys = list(final_magnetism.keys())
    if not candidate_keys:
        candidate_keys = list(final_energy.keys())

    if magmom_list:
        rows = [
            (idx, _initial_mag_from_list_entry(entry))
            for idx, entry in enumerate(magmom_list)
        ]
    else:
        rows = [(idx, "—") for idx in range(len(candidate_keys))]

    lines = [
        "| initial magmom | final total magmom | final_energy |",
        "| --- | --- | --- |",
    ]

    for idx, initial in rows:
        pk = candidate_keys[idx] if idx < len(candidate_keys) else None

        if pk is None:
            lines.append(f"| {initial} | — | — |")
            continue

        total = _coerce_scalar(final_magnetism.get(pk))
        energy = final_energy.get(pk)

        if total is None:
            total_str = "—"
        elif isinstance(total, list):
            total_str = "[" + ", ".join(f"{float(v):.6f}" for v in total) + "]"
        else:
            total_str = f"{float(total):.6f}"
        energy_str = "—" if energy is None else f"{float(energy):.6f}"
        lines.append(f"| {initial} | {total_str} | {energy_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report for a magmom WorkChain."""
    lines: List[str] = [
        render_report_header(
            title="Magmom WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params),
        "",
    ]

    if "status" in output_params:
        lines += [
            "## Calculation Status",
            "",
            generate_status_table(output_params["status"]),
            "",
        ]

    if "final_energy" in output_params and output_params["final_energy"]:
        lines += [
            "## Final Energy",
            "",
            generate_energy_table(output_params["final_energy"]),
            "",
        ]

    if "wall_time_seconds" in output_params and output_params["wall_time_seconds"]:
        lines += [
            "## Wall Time [s]",
            "",
            generate_wall_time_table(output_params["wall_time_seconds"]),
            "",
        ]

    # Backend-specific magnetism section.
    if workflow_type == "abacus":
        lines += _render_abacus_section(output_params)
    else:
        lines += _render_vasp_section(output_params)

    lines += [
        "",
        "## Magmom Convergence",
        "",
        generate_magmom_summary_table(output_params, workflow_type),
        "",
        render_report_footer(),
    ]

    return "\n".join(lines)
