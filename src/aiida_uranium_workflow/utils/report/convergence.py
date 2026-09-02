"""Convergence WorkChain report generation module.

Extracts total energy data from ``VaspConvergenceWorkChain`` or
``AbacusConvergenceWorkChain`` ``output_parameters`` and renders a
Markdown report with tables.

The 2-D ``ecutwfc`` × ``kpoints`` grid renderer is shared with the
smear / magmom reports via :mod:`utils.report._common`. The public
``generate_*`` entry points keep their original signatures so existing
callers and tests are unaffected.

Per-backend key layout produced by the convergence workflows:

* ABACUS ``AbacusConvergenceWorkChain`` — ``ecutwfc`` / ``kpoints_distance`` /
  ``kpoints_mesh`` (with `ecutwfc` energy in Ry)
* VASP    ``VaspConvergenceWorkChain`` — ``encut`` / ``kpoints_spacing`` /
  ``kpoints_mesh`` (with `encut` energy in eV)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ._common import (
    append_overview_section,
    AxisSpec,
    format_scalar,
    render_2d_grid,
    render_report_footer,
    render_report_header,
    sort_axis_values,
)


# ---------------------------------------------------------------------------
# Axis layout (shared by every 2-D table in this module)
# ---------------------------------------------------------------------------

CONVERGENCE_AXES: Tuple[AxisSpec, AxisSpec] = (
    AxisSpec(
        name="ecut",
        keyword="ecutwfc",
        keyword_aliases=("encut",),
        kind="float",
    ),
    AxisSpec(
        name="kpoints",
        keyword="kpoints",
        kind="auto",
        qualifiers=("spacing", "distance"),
    ),
)


# ---------------------------------------------------------------------------
# Backend- / mode-aware header resolution
# ---------------------------------------------------------------------------


def _get_kpoints_mode(output_params: Dict[str, Any]) -> Optional[str]:
    """Determine kpoints mode from ``output_params``.

    Returns one of: ``"mesh"``, ``"spacing"``, ``"distance"``, or
    ``None`` if undetermined. Priority:

    1. Explicit ``"kpoints_mode"`` field in ``output_params``.
    2. Presence of ``"kpoints_mesh_list"`` / ``"kpoints_list"`` field
       (legacy convention: mesh sweep stores its list under either
       key).
    """
    mode = output_params.get("kpoints_mode")
    if mode:
        return str(mode)

    if "kpoints_mesh_list" in output_params or "kpoints_list" in output_params:
        return "mesh"

    return None


def _resolve_kpoints_label(workflow_type: str, mode: Optional[str]) -> str:
    """Return the column header label for the kpoints axis."""
    if mode == "mesh":
        return "kpoints_mesh"
    if workflow_type == "vasp":
        return "kpoints_spacing (A^-1 * 2pi)"
    return "kpoints_distance (A^-1)"


def _ecut_label(workflow_type: str) -> str:
    return "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"


def _ecut_unit(workflow_type: str) -> str:
    return "eV" if workflow_type == "vasp" else "Ry"


# ---------------------------------------------------------------------------
# Status / Energy / Wall-time tables (2-D grid)
# ---------------------------------------------------------------------------


def _status_cell(value: Any) -> str:
    """Status cells use bare integer formatting (no ``%.6f``)."""
    if value is None or isinstance(value, str):
        return "—"
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def generate_status_table(
    status: Dict[str, int],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a status table showing each child's exit code."""
    mode = _get_kpoints_mode(output_params or {})
    return render_2d_grid(
        status,
        CONVERGENCE_AXES,
        row_header=_ecut_label(workflow_type),
        col_header=_resolve_kpoints_label(workflow_type, mode),
        cell_format=_status_cell,
        empty_placeholder="-",
    )


def generate_energy_table(
    total_energy: Dict[str, Any],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the total-energy table on the ``ecutwfc`` × ``kpoints`` grid."""
    mode = _get_kpoints_mode(output_params or {})
    return render_2d_grid(
        total_energy,
        CONVERGENCE_AXES,
        row_header=_ecut_label(workflow_type),
        col_header=_resolve_kpoints_label(workflow_type, mode),
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
        empty_placeholder="—",
    )


def generate_wall_time_table(
    wall_time_seconds: Dict[str, Any],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the wall-clock-time table on the ``ecutwfc`` × ``kpoints`` grid."""
    mode = _get_kpoints_mode(output_params or {})
    return render_2d_grid(
        wall_time_seconds,
        CONVERGENCE_AXES,
        row_header=_ecut_label(workflow_type),
        col_header=_resolve_kpoints_label(workflow_type, mode),
        cell_format=lambda v: format_scalar(v, fmt="%.3f"),
        empty_placeholder="—",
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Generate a summary table with basic info."""
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    mode = _get_kpoints_mode(output_params)
    if mode:
        lines.append(f"| kpoints_mode | {mode} |")

    if "total_energy" in output_params:
        lines.append(
            f"| total_energy entries | {len(output_params['total_energy'])} |"
        )

    if "num_atoms" in output_params and output_params["num_atoms"]:
        first_key = next(iter(output_params["num_atoms"].keys()))
        lines.append(f"| num_atoms | {output_params['num_atoms'][first_key]} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convergence-difference tables (encut / kpoints deltas)
# ---------------------------------------------------------------------------
# These two tables are workflow-specific (convergence is the only place
# that exposes (ecut, kpoints) deltas) and so cannot ride the shared
# ``render_2d_grid`` renderer. Kept here as plain data preparation +
# plain string assembly, but the (row, col) bucketing logic is
# factored out into :func:`_bucket_grid` so the two tables share it.


def _bucket_grid(
    data: Dict[str, float],
) -> Tuple[
    list,
    list,
    Dict[Tuple[Any, Any], Any],
]:
    """Bucket ``{label: value}`` into row/col/lookup using ``CONVERGENCE_AXES``.

    Returns:
        ``(row_values, col_values, buckets)`` where ``buckets[(row, col)]``
        is the value (or ``None`` when the label parsed but the label
        is missing for the given (row, col) intersection — though in
        practice callers rely on lookups with both keys present).
    """
    from ._common import parse_axes

    buckets: Dict[Tuple[Any, Any], Any] = {}
    for label, value in data.items():
        values = parse_axes(label, list(CONVERGENCE_AXES))
        if values[0] is None or values[1] is None:
            continue
        buckets[(values[0], values[1])] = value

    if not buckets:
        return [], [], {}

    row_values = sort_axis_values({k[0] for k in buckets})
    col_values = sort_axis_values({k[1] for k in buckets})
    return row_values, col_values, buckets


def _render_diff_table(
    total_energy_per_atom: Dict[str, float],
    *,
    workflow_type: str,
    mode: Optional[str],
    diff_along: str,
) -> str:
    """Render a Markdown table with row-by-row energy differences.

    ``diff_along == "row"`` (encut convergence): the first row is ``-``;
    subsequent rows show ``(curr_row - prev_row) * 1000`` (meV/atom)
    for each column. ``diff_along == "col"`` (kpoints convergence): the
    last column is ``-``; preceding columns show
    ``(curr_col - next_col) * 1000``. For mesh mode, columns are
    reversed so the difference reads as
    ``E(higher density) - E(lower density)``.
    """
    if not total_energy_per_atom:
        return "No data available."

    row_values, col_values, buckets = _bucket_grid(total_energy_per_atom)
    if not buckets:
        return "No data available."

    if diff_along == "col" and mode == "mesh":
        col_values = list(reversed(col_values))

    ecut_label = _ecut_label(workflow_type)
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label}\\{kpoints_label} | "
        + " | ".join(f"{c}" for c in col_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in col_values) + " |"

    rows: list[str] = []
    for idx, row_value in enumerate(row_values):
        cells = [f"| {row_value}"]
        if diff_along == "row":
            if idx == 0:
                cells.append("| -")
                cells.extend("| -" for _ in col_values[1:])
                cells.append("|")
                rows.append(" ".join(cells))
                continue
            prev_row = row_values[idx - 1]
        else:
            # diff_along == "col" — cells iterate over col_values.
            pass

        for col_idx, col_value in enumerate(col_values):
            if diff_along == "row":
                curr = buckets.get((row_value, col_value))
                prev = buckets.get((prev_row, col_value))
            else:
                if col_idx == len(col_values) - 1:
                    cells.append("| -")
                    continue
                next_col = col_values[col_idx + 1]
                curr = buckets.get((row_value, col_value))
                prev = buckets.get((row_value, next_col))

            if curr is not None and prev is not None:
                diff = (curr - prev) * 1000
                cells.append(f"| {diff:+.4f}")
            else:
                cells.append("| N/A")
        cells.append("|")
        rows.append(" ".join(cells))

    return "\n".join([header, separator] + rows)


def generate_encut_convergence_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Markdown table showing energy difference when increasing encut."""
    mode = _get_kpoints_mode(output_params or {})
    return _render_diff_table(
        total_energy_per_atom,
        workflow_type=workflow_type,
        mode=mode,
        diff_along="row",
    )


def generate_kpoints_convergence_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Markdown table showing energy difference when decreasing kpoints density."""
    mode = _get_kpoints_mode(output_params or {})
    return _render_diff_table(
        total_energy_per_atom,
        workflow_type=workflow_type,
        mode=mode,
        diff_along="col",
    )


# ---------------------------------------------------------------------------
# Per-cell raw energy tables (total / per-atom)
# ---------------------------------------------------------------------------


def generate_total_energy_table(
    total_energy: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Markdown table for total energy (raw, no delta)."""
    mode = _get_kpoints_mode(output_params or {})
    return render_2d_grid(
        total_energy,
        CONVERGENCE_AXES,
        row_header=_ecut_label(workflow_type),
        col_header=_resolve_kpoints_label(workflow_type, mode),
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
        empty_placeholder="",
    )


def generate_total_energy_per_atom_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Markdown table for total energy per atom (raw, no delta)."""
    mode = _get_kpoints_mode(output_params or {})
    return render_2d_grid(
        total_energy_per_atom,
        CONVERGENCE_AXES,
        row_header=_ecut_label(workflow_type),
        col_header=_resolve_kpoints_label(workflow_type, mode),
        cell_format=lambda v: format_scalar(v, fmt="%.8f"),
        empty_placeholder="",
    )


# ---------------------------------------------------------------------------
# Converged-parameter recommendation
# ---------------------------------------------------------------------------


def find_converged_parameters(
    total_energy_per_atom: Dict[str, float],
    energy_threshold: float = 1e-5,
) -> Dict[str, Any]:
    """Find converged ecutwfc and kpoints parameters.

    Iterating over ascending ecutwfc and kpoints values, we identify
    the smallest parameter set where the energy difference between
    consecutive values is below the threshold.

    Args:
        total_energy_per_atom: Dict of label -> total energy per atom values
        energy_threshold: Maximum allowed energy difference (default: 1e-5 Ry/atom)

    Returns:
        Dict with ``"ecutwfc"`` and ``"kpoints"`` keys mapping to the
        recommended values, or ``None`` if convergence is not achieved.
        kpoints value can be a float (spacing/distance) or string (mesh).
    """
    ecutwfc_data: Dict[float, list[tuple[float | str, float]]] = {}

    row_values, col_values, buckets = _bucket_grid(total_energy_per_atom)
    for (ecutwfc, kpoints_val), value in buckets.items():
        if value is None or isinstance(value, str):
            continue
        if ecutwfc is None or kpoints_val is None:
            continue
        ecutwfc_data.setdefault(ecutwfc, []).append((kpoints_val, float(value)))

    converged_ecutwfc: float | None = None
    converged_kpoints: float | str | None = None

    if ecutwfc_data:
        sorted_ecutwfcs = sorted(ecutwfc_data.keys())

        for i in range(1, len(sorted_ecutwfcs)):
            prev_ecutwfc = sorted_ecutwfcs[i - 1]
            curr_ecutwfc = sorted_ecutwfcs[i]

            prev_data = sorted(ecutwfc_data[prev_ecutwfc], key=lambda x: x[0])
            curr_data = sorted(ecutwfc_data[curr_ecutwfc], key=lambda x: x[0])

            if prev_data and curr_data:
                for prev_k, prev_energy in prev_data:
                    for curr_k, curr_energy in curr_data:
                        if prev_k == curr_k:
                            diff = abs(curr_energy - prev_energy)
                            if diff < energy_threshold:
                                converged_ecutwfc = curr_ecutwfc
                                converged_kpoints = curr_k
                                break
                    if converged_ecutwfc is not None:
                        break
            if converged_ecutwfc is not None:
                break

    return {
        "ecutwfc": converged_ecutwfc,
        "kpoints": converged_kpoints,
    }


def generate_converged_section(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate the converged parameters recommendation section."""
    lines = []
    lines.append("## Recommended Converged Parameters")
    lines.append("")

    ecut_label = "encut" if workflow_type == "vasp" else "ecutwfc"
    ecut_unit = _ecut_unit(workflow_type)

    converged = find_converged_parameters(total_energy_per_atom)
    kpoints_val = converged.get("kpoints")

    mode = _get_kpoints_mode(output_params or {})
    if mode == "mesh" or isinstance(kpoints_val, str):
        kpoints_label = "kpoints_mesh"
        kpoints_unit = ""
    elif workflow_type == "vasp":
        kpoints_label = "kpoints_spacing"
        kpoints_unit = " A^-1 * 2pi"
    else:
        kpoints_label = "kpoints_distance"
        kpoints_unit = " A^-1"

    lines.append(
        f"This section recommends the smallest {ecut_label} and {kpoints_label} "
        f"where the total energy difference between consecutive values is < 1e-5 {ecut_unit}/atom."
    )
    lines.append("")

    if not converged["ecutwfc"] or not kpoints_val:
        lines.append("No convergence achieved within the tested parameter range.")
        return "\n".join(lines)

    lines.append("| Parameter | Recommended Value |")
    lines.append("| --- | --- |")
    lines.append(f"| {ecut_label} | {converged['ecutwfc']} {ecut_unit} |")
    lines.append(f"| {kpoints_label} | {kpoints_val}{kpoints_unit} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report."""
    energy_unit = _ecut_unit(workflow_type)

    report_lines = [
        render_report_header(
            title="Convergence WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params),
        "",
    ]

    append_overview_section(report_lines, output_params)

    if "status" in output_params:
        report_lines += [
            "## Calculation Status",
            "",
            generate_status_table(
                output_params["status"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
        ]

    if "total_energy" in output_params and output_params["total_energy"]:
        report_lines += [
            f"## Total Energy [{energy_unit}]",
            "",
            generate_energy_table(
                output_params["total_energy"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
        ]

    if "wall_time_seconds" in output_params and output_params["wall_time_seconds"]:
        report_lines += [
            "## Wall Time [s]",
            "",
            generate_wall_time_table(
                output_params["wall_time_seconds"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
        ]

    if "total_energy_per_atom" in output_params:
        report_lines += [
            f"## Total Energy per Atom ({energy_unit}/atom)",
            "",
            generate_total_energy_per_atom_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
        ]

    if (
        "total_energy_per_atom" in output_params
        and output_params["total_energy_per_atom"]
    ):
        report_lines += [
            "## Encut Convergence (meV/atom)",
            "",
            generate_encut_convergence_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
            "## Kpoints Convergence (meV/atom)",
            "",
            generate_kpoints_convergence_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
            generate_converged_section(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            ),
            "",
        ]

    report_lines += [render_report_footer()]

    return "\n".join(report_lines)
