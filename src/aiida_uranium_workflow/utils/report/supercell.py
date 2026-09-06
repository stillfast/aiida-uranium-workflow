"""Supercell SCF WorkChain report generation.

Renders one table row per supercell (matrix / natoms / volume / total
energy / wall time / scf steps) from the ``output_parameters`` Dict
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
        "| time (s) | scf steps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        stats = {}
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
            f"| {_fmt(stats.get('time_s', cell.get('time_s')), 3)} "
            f"| {stats.get('scf_steps', cell.get('scf_steps', '—'))} |"
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
