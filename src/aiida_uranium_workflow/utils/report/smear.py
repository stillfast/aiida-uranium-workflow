"""Smear WorkChain report generation module.

Extracts electronic entropy data from ``VaspSmearWorkChain`` or
``AbacusSmearWorkChain`` ``output_parameters`` and renders a Markdown
report with tables.

The 2-D ``smear`` × ``sigma`` grid renderer is shared with the
convergence / magmom reports via :mod:`utils.report._common`. The
``generate_status_table`` / ``generate_energy_table`` /
``generate_wall_time_table`` / ``generate_eentropy_table`` entry points
keep their original signatures so existing callers and tests are
unaffected.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from ._common import (
    AxisSpec,
    append_overview_section,
    format_scalar,
    render_2d_grid,
    render_report_footer,
    render_report_header,
)


# ---------------------------------------------------------------------------
# Axis layout (shared by every 2-D table in this module)
# ---------------------------------------------------------------------------

SMEAR_AXES: tuple[AxisSpec, AxisSpec] = (
    AxisSpec(name="smear", keyword="smear", kind="string"),
    AxisSpec(name="sigma", keyword="sigma", kind="float"),
)


def _unit_label(sigma_unit: str) -> str:
    return "Ry" if sigma_unit == "ry" else "eV"


ROW_HEADER = "smearing_method"


# ---------------------------------------------------------------------------
# Per-signal 2-D tables
# ---------------------------------------------------------------------------


def _status_cell(value: Any) -> str:
    """Status cells use bare integer formatting (no ``%.6f``)."""
    if value is None or isinstance(value, str):
        return "—"
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def generate_status_table(
    status: Dict[str, int], sigma_unit: str = "ev"
) -> str:
    """Generate a status table showing each child's exit code.

    Each entry maps ``Smear_{method}_Sigma_{sigma}`` to the exit code
    reported by ``verdi process status``: ``0`` means the calculation
    finished successfully, anything else represents the corresponding
    AiiDA exit status (e.g. ``300`` for ``ERROR_CHILD``).
    """
    return render_2d_grid(
        status,
        SMEAR_AXES,
        row_header=ROW_HEADER,
        col_header=f"smearing_sigma [{_unit_label(sigma_unit)}]",
        cell_format=_status_cell,
        empty_placeholder="-",
    )


def generate_energy_table(
    total_energy: Dict[str, Any], sigma_unit: str = "ev"
) -> str:
    """Render the total-energy table on the ``smear`` × ``sigma`` grid."""
    return render_2d_grid(
        total_energy,
        SMEAR_AXES,
        row_header=ROW_HEADER,
        col_header=f"total_energy [{_unit_label(sigma_unit)}]",
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
        empty_placeholder="—",
    )


def generate_wall_time_table(
    wall_time_seconds: Dict[str, Any],
) -> str:
    """Render the wall-clock-time table on the ``smear`` × ``sigma`` grid."""
    return render_2d_grid(
        wall_time_seconds,
        SMEAR_AXES,
        row_header=ROW_HEADER,
        col_header="wall_time [s]",
        cell_format=lambda v: format_scalar(v, fmt="%.3f"),
        empty_placeholder="—",
    )


def generate_eentropy_table(
    eentropy_per_atom: Dict[str, float], sigma_unit: str = "ev"
) -> str:
    """Generate a Markdown table for eentropy per atom.

    Rows are smear methods, columns are sigma values. ``sigma_unit``
    controls the unit annotation in the table header: ``"ry"`` for
    ABACUS (sigma in Rydberg), ``"ev"`` for VASP.
    """
    return render_2d_grid(
        eentropy_per_atom,
        SMEAR_AXES,
        row_header=ROW_HEADER,
        col_header=f"smearing_sigma [{_unit_label(sigma_unit)}]",
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
        empty_placeholder="",
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Generate a summary table with basic info."""
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    if "smear_pairs" in output_params:
        lines.append(f"| smear_pairs | {len(output_params['smear_pairs'])} |")

    if "eentropy" in output_params:
        lines.append(f"| eentropy entries | {len(output_params['eentropy'])} |")

    if "num_atoms" in output_params and output_params["num_atoms"]:
        first_key = next(iter(output_params["num_atoms"].keys()))
        lines.append(f"| num_atoms | {output_params['num_atoms'][first_key]} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimal-sigma recommendation
# ---------------------------------------------------------------------------


def find_optimal_sigma(
    eentropy_per_atom: Dict[str, float],
    threshold: float = 0.001,
    sigma_unit: str = "ev",
) -> Dict[str, float | None]:
    """Find the optimal sigma for each smear method.

    According to VASP Wiki: SIGMA should be as large as possible while keeping
    the difference between free energy and total energy (entropy T*S term)
    negligible (< 1 meV/atom = 0.001 eV/atom).

    Recommendation algorithm
    ------------------------
    Iterating over ascending ``sigma`` values, we mark a sigma as the
    candidate *iff* its per-atom entropy is below ``threshold`` AND
    the delta against the previous sigma is below ``threshold`` as
    well.  If no sigma satisfies this — for instance ``smear=gauss``
    where the entropy is always well above 1 meV/atom — the
    recommendation for that smear method is ``None``.

    The function returns sigma values **in the input unit**; the
    caller is responsible for any unit formatting in the rendered
    table.  ``abacus`` reports sigma in Rydberg, ``vasp`` in eV.
    """
    del sigma_unit  # kept for API compatibility; sigma values are
    # returned in the input unit, no implicit conversion happens here.

    from ._common import parse_axes

    smear_data: Dict[str, list[tuple[float, float]]] = {}

    for label, value in eentropy_per_atom.items():
        if value is None or isinstance(value, str):
            continue
        smear, sigma = parse_axes(label, list(SMEAR_AXES))
        if smear is not None and isinstance(sigma, (int, float)):
            smear_data.setdefault(smear, []).append((float(sigma), float(value)))

    optimal_sigma: Dict[str, float | None] = {}

    for smear, data in smear_data.items():
        if not data:
            optimal_sigma[smear] = None
            continue

        data.sort(key=lambda x: x[0])

        accepted: float | None = None
        prev_value: float | None = None
        for sigma, eentropy in data:
            per_atom = abs(eentropy)
            if prev_value is None:
                prev_value = per_atom
                continue
            diff = abs(per_atom - prev_value)
            if diff < threshold and per_atom < threshold:
                accepted = sigma
            prev_value = per_atom

        optimal_sigma[smear] = round(accepted, 2) if accepted is not None else None

    return optimal_sigma


def generate_optimal_sigma_section(
    eentropy_per_atom: Dict[str, float], sigma_unit: str = "ev"
) -> str:
    """Generate the optimal sigma recommendation section."""
    lines = []
    lines.append("## Recommended Sigma Values")
    lines.append("")
    lines.append(
        "According to the VASP Wiki Smearing technique documentation: "
        "SIGMA should be as large as possible while keeping the entropy T*S term "
        "negligible (< 1 meV/atom = 0.001 eV/atom)."
    )
    lines.append("")
    lines.append(
        "This section recommends the largest sigma for each smear method "
        "where the eentropy change from the smallest sigma is < 0.001 eV/atom "
        "AND the absolute eentropy value < 0.001 eV/atom."
    )
    lines.append("")

    optimal = find_optimal_sigma(eentropy_per_atom, sigma_unit=sigma_unit)

    unit_label = _unit_label(sigma_unit)

    if not optimal:
        lines.append("No valid data available for recommendation.")
        return "\n".join(lines)

    lines.append(f"| smearing_method | recommended_sigma ({unit_label}) |")
    lines.append("| --- | --- |")

    for smear in sorted(optimal.keys()):
        sigma = optimal[smear]
        lines.append(f"| {smear} | {sigma} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report.

    Args:
        output_params: The output_parameters dict from the WorkChain
        pk: The WorkChain pk
        workflow_type: 'vasp' or 'abacus'

    Returns:
        Markdown report as string
    """
    sigma_unit = "ry" if workflow_type == "abacus" else "ev"

    report_lines = [
        render_report_header(
            title="Smear WorkChain Report",
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
            generate_status_table(output_params["status"], sigma_unit=sigma_unit),
            "",
        ]

    if "total_energy" in output_params and output_params["total_energy"]:
        report_lines += [
            f"## Total Energy [{_unit_label(sigma_unit)}]",
            "",
            generate_energy_table(
                output_params["total_energy"], sigma_unit=sigma_unit
            ),
            "",
        ]

    if "wall_time_seconds" in output_params and output_params["wall_time_seconds"]:
        report_lines += [
            "## Wall Time [s]",
            "",
            generate_wall_time_table(output_params["wall_time_seconds"]),
            "",
        ]

    if "eentropy_per_atom" in output_params:
        report_lines += [
            "## Electronic Entropy per Atom (eV/atom)",
            "",
            generate_eentropy_table(
                output_params["eentropy_per_atom"], sigma_unit=sigma_unit
            ),
            "",
        ]

    if "eentropy" in output_params:
        report_lines += [
            "## Electronic Entropy (eV)",
            "",
            generate_eentropy_table(output_params["eentropy"], sigma_unit=sigma_unit),
            "",
        ]

    if "eentropy_per_atom" in output_params and output_params["eentropy_per_atom"]:
        report_lines += [
            generate_optimal_sigma_section(
                output_params["eentropy_per_atom"], sigma_unit=sigma_unit
            ),
            "",
        ]

    report_lines += [render_report_footer()]

    return "\n".join(report_lines)
