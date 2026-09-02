"""Defect WorkChain report generation.

Renders the neutral defect formation energy, the raw energy difference,
the chemical-potential correction and the host / defect cell summary
from the ``output_parameters`` of either backend's defect WorkChain.
"""

from __future__ import annotations

from ._common import (
    render_report_footer,
    render_report_header,
)
from typing import Any, Dict, List


def _fmt(value: Any, prec: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, prec) for v in value) + "]"
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Render the formation-energy summary."""
    defect = output_params.get("defect") or {}
    rows = [
        ("defect type", defect.get("type")),
        ("defect label", defect.get("label")),
        ("mode", output_params.get("mode")),
        ("host atoms", output_params.get("host_natoms")),
        ("defect atoms", output_params.get("defect_natoms")),
        ("host energy", f"{_fmt(output_params.get('host_energy_ev'))} eV"),
        ("defect energy", f"{_fmt(output_params.get('defect_energy_ev'))} eV"),
        ("E_defect − E_host", f"{_fmt(output_params.get('energy_difference_ev'))} eV"),
        ("formation energy", f"{_fmt(output_params.get('formation_energy_ev'))} eV"),
        ("formula", output_params.get("formula")),
    ]
    lines = ["| Property | Value |", "| --- | --- |"]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    workchain_node=None,
    figure_dir: str | None = None,
    report_stem: str | None = None,
) -> str:
    """Generate a complete Markdown report for a defect WorkChain."""
    lines: List[str] = [
        render_report_header(
            title="Defect Formation Energy Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params),
        "",
    ]
    lines += [render_report_footer()]
    return "\n".join(lines)
