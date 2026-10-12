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
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        # supercell matrices / k-meshes are integers — no decimals
        return str(value)
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Render the formation-energy summary."""
    defect = output_params.get("defect") or {}
    n_ok = output_params.get("n_cells")
    n_req = output_params.get("n_cells_requested")
    n_failed = output_params.get("n_cells_failed")
    status = output_params.get("status")
    if status is None:
        status = "partial" if n_failed else "complete"
    rows = [
        ("status", f"**{status}**" if status == "partial" else status),
        (
            "cells computed",
            f"{n_ok} of {n_req}"
            + (f" — **{n_failed} failed**" if n_failed else "")
            if n_req is not None
            else None,
        ),
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
        if value is None:
            continue
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


def generate_failures_table(output_params: Dict[str, Any]) -> str:
    """Render the cells that could not be computed.

    Present only for a partial scan: one row per failed supercell with the
    reason and the child WorkChain pks / exit statuses, so the missing
    data points are traceable from the report alone.
    """
    failures = output_params.get("failures") or []
    if not failures:
        return ""
    lines = [
        "| supercell | matrix | reason | host pk (exit) | defect pk (exit) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in failures:
        host = "—"
        if f.get("host_pk") is not None:
            host = f"{f['host_pk']} (exit={_fmt(f.get('host_exit_status'), 0)})"
        defect = "—"
        if f.get("defect_pk") is not None:
            defect = f"{f['defect_pk']} (exit={_fmt(f.get('defect_exit_status'), 0)})"
        reason = str(f.get("reason") or "—")
        if f.get("error"):
            reason += f": {f['error']}"
        lines.append(
            "| {label} | {matrix} | {reason} | {host} | {defect} |".format(
                label=f.get("label") or "—",
                matrix=_fmt(f.get("matrix")),
                reason=reason,
                host=host,
                defect=defect,
            )
        )
    return "\n".join(lines)


def generate_cells_table(output_params: Dict[str, Any]) -> str:
    """Render the per-supercell formation energies (multi-cell scans).

    One row per computed supercell matrix; a supercell-size scan shows
    whether E_f is converged with respect to the cell size. The runtime
    columns mirror the supercell report (``time`` / ``scf steps``) — here
    split into the host and defect legs, since every cell runs both.
    """
    cells = output_params.get("cells") or []
    lines = [
        "| supercell | matrix | k-mesh | N_host | N_defect | E_host (eV) | "
        "E_defect (eV) | E_f (eV) | t_host (s) | t_defect (s) | "
        "steps host | steps defect |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        mesh = cell.get("kpoints_mesh")
        if mesh is None and cell.get("kpoints_distance") is not None:
            mesh = f"{_fmt(cell['kpoints_distance'])} Å⁻¹"
        steps_host = cell.get("scf_steps_host")
        steps_defect = cell.get("scf_steps_defect")
        lines.append(
            "| {label} | {matrix} | {mesh} | {nh} | {nd} | {eh} | {ed} | {ef} | "
            "{th} | {td} | {sh} | {sd} |".format(
                label=cell.get("label") or "—",
                matrix=_fmt(cell.get("matrix")),
                mesh=_fmt(mesh) if mesh is not None else "preset",
                nh=cell.get("host_natoms"),
                nd=cell.get("defect_natoms"),
                eh=_fmt(cell.get("host_energy_ev"), 4),
                ed=_fmt(cell.get("defect_energy_ev"), 4),
                ef=_fmt(cell.get("formation_energy_ev"), 4),
                th=_fmt(cell.get("time_host_s"), 0),
                td=_fmt(cell.get("time_defect_s"), 0),
                sh=steps_host if steps_host is not None else "—",
                sd=steps_defect if steps_defect is not None else "—",
            )
        )
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
    cells = output_params.get("cells") or []
    failures = output_params.get("failures") or []
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
    if failures:
        lines += [
            "> **Partial result**: "
            f"{output_params.get('n_cells_failed', len(failures))} of "
            f"{output_params.get('n_cells_requested', len(cells) + len(failures))} "
            "supercell(s) failed and are missing below — the remaining "
            "formation energies are complete and usable.",
            "",
        ]
    if workchain_node is not None and not getattr(workchain_node, "is_finished_ok", True):
        lines += [
            "> **WorkChain exit status**: "
            f"`{getattr(workchain_node, 'exit_status', None)}` "
            f"({getattr(workchain_node, 'exit_message', '') or 'no message'}) — "
            "the report covers every cell that finished OK.",
            "",
        ]
    if len(cells) > 1 or failures:
        defect = output_params.get("defect") or {}
        lines += [
            f"## Supercell convergence ({len(cells)} cells"
            + (f", {len(failures)} failed" if failures else "")
            + ")",
            "",
            f"Defect: **{defect.get('type', '?')}**"
            f"{' (' + str(defect.get('label')) + ')' if defect.get('label') else ''}"
            f" — E_f = E_defect − E_host × N_defect/N_host.",
            "",
            generate_cells_table(output_params),
            "",
        ]
        if failures:
            lines += [
                "## Failed cells",
                "",
                generate_failures_table(output_params),
                "",
            ]
    lines += [render_report_footer()]
    return "\n".join(lines)
