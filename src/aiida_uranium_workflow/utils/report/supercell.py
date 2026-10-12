"""Supercell SCF WorkChain report generation.

Renders one table row per supercell (matrix / natoms / volume / total
energy / average atomic energy / wall time (integer seconds) / scf
steps) from the
``output_parameters`` Dict
produced by
:class:`aiida_uranium_workflow.workflows.supercell.abacus.SupercellScfWorkChain`.

``time_s`` / ``scf_steps`` are only stored in the output dict by recent
workflow runs. For older nodes (or when the dict lacks them) the report
re-derives them from each SCF child node (``scf_pk``) on the fly, so the
columns never render empty for a finished WorkChain.
"""

from __future__ import annotations

from typing import Any, Dict

from ._common import (
    render_report_footer,
    render_report_header,
)


def _fmt(value: Any, prec: int = 6) -> str:
    """Render a scalar (or list of scalars) with fixed precision."""
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, prec) for v in value) + "]"
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def _energy_per_atom(cell: Dict[str, Any]) -> Any:
    """Average atomic energy (eV/atom) of a cell.

    Uses the stored ``energy_per_atom`` when present (recent runs) and
    otherwise derives it from ``energy``/``natoms`` so older WorkChains
    still show the column.
    """
    epa = cell.get("energy_per_atom")
    if epa is not None:
        return epa
    energy, natoms = cell.get("energy"), cell.get("natoms")
    if energy is None or not natoms:
        return None
    try:
        return float(energy) / float(natoms)
    except (TypeError, ValueError):
        return None


def _child_scf_stats(scf_pk: Any) -> Dict[str, Any]:
    """Best-effort fetch of wall time / scf steps from an SCF child.

    Returns ``{}`` when the child can't be loaded or parsed (the table
    then renders dashes). Guards against report-time DB hiccups.
    """
    try:
        from aiida.orm import load_node

        from aiida_uranium_workflow.utils.parsers import fetch_summary

        child = load_node(int(scf_pk))
        summary = fetch_summary(child, "abacus")
        return {
            "time_s": summary.get("time_s"),
            "scf_steps": summary.get("scf_steps"),
        }
    except Exception:  # noqa: BLE001 — report must never crash on a child
        return {}


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Render the per-supercell energy / volume table."""
    cells = output_params.get("cells") or []
    if not cells:
        return "No supercell data available."

    lines = [
        "| label | matrix | natoms | volume (Å³) | energy (eV) "
        "| energy/atom (eV) | time (s) | scf steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        stats = {}
        if cell.get("status") == "failed" and cell.get("energy") is None:
            # deliberately failed cell (a parameter value outside PEXSI's
            # pole table, a phase scan that cannot converge, ...) — keep the
            # row, there is simply no data for it
            lines.append(
                f"| {cell.get('label', '—')} | {_fmt(cell.get('matrix'), 0)} "
                f"| {cell.get('natoms', '—')} | {_fmt(cell.get('volume'))} "
                f"| — | — | — | — |"
            )
            continue
        if cell.get("time_s") is None or cell.get("scf_steps") is None:
            # Older output dicts lack the SCF stats — re-derive from the
            # child node so already-run WorkChains still show them.
            stats = _child_scf_stats(cell.get("scf_pk"))
        lines.append(
            f"| {cell.get('label', '—')} "
            f"| {_fmt(cell.get('matrix'), 0)} "
            f"| {cell.get('natoms', '—')} "
            f"| {_fmt(cell.get('volume'))} "
            f"| {_fmt(cell.get('energy'))} "
            f"| {_fmt(_energy_per_atom(cell))} "
            f"| {_fmt(stats.get('time_s', cell.get('time_s')), 0)} "
            f"| {stats.get('scf_steps') or cell.get('scf_steps') or '—'} |"
        )
    return "\n".join(lines)


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    workchain_node=None,
) -> str:
    """Generate a complete Markdown report for a supercell SCF WorkChain."""
    data = dict(output_params)
    if workchain_node is not None:
        try:
            data = workchain_node.outputs.output_parameters.get_dict()
        except (AttributeError, KeyError):
            pass

    lines: list[str] = [
        render_report_header(
            title="Supercell SCF WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Supercell SCF Results",
        "",
        generate_summary_table(data),
        "",
    ]
    lines += [render_report_footer()]
    return "\n".join(lines)
