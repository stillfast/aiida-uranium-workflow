"""Phonon WorkChain report generation.

Renders the phonon summary (structure / supercell / band-path settings,
frequency range, imaginary modes, provenance) from the
``AbacusPhonopyWorkChain`` ``output_parameters``.

When ``figure_dir`` is provided (the report CLI passes the report
directory), the band-structure + DOS figure is rendered next to the
Markdown report and embedded in it — see
:func:`aiida_uranium_workflow.utils.plot.phonon.render_phonon_figure`.
"""

from __future__ import annotations

from ._common import (
    render_report_footer,
    render_report_header,
)
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt(value: Any, prec: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        try:
            return "[" + ", ".join(f"{float(v):.{prec}f}" for v in value) + "]"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        f = float(value)
        if f != 0.0 and (abs(f) < 1e-3 or abs(f) >= 1e6):
            return f"{f:.3e}"
        return f"{f:.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def _band_mode(phonopy_parameters: Optional[Dict[str, Any]]) -> str:
    if not phonopy_parameters:
        return "—"
    band = phonopy_parameters.get("band")
    if isinstance(band, str):
        return f"auto (BAND={band})"
    if "band_paths" in phonopy_parameters:
        return f"auto (BAND_PATHS={phonopy_parameters['band_paths']})"
    if isinstance(band, list):
        return "manual (explicit BAND)"
    return "—"


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Render the phonon run summary table."""
    ph_params = output_params.get("phonopy_parameters") or {}
    rows = [
        ("backend", output_params.get("backend")),
        ("structure", output_params.get("structure_formula")),
        ("displaced supercells", output_params.get("n_supercells")),
        ("supercell matrix", output_params.get("supercell_matrix")),
        ("primitive matrix", output_params.get("primitive_matrix")),
        ("symprec", output_params.get("symprec")),
        ("band mode", _band_mode(ph_params)),
        ("band labels", output_params.get("band_labels")),
        ("frequency min", f"{_fmt(output_params.get('frequency_min_thz'))} THz"),
        ("frequency max", f"{_fmt(output_params.get('frequency_max_thz'))} THz"),
        ("imaginary modes", output_params.get("n_imaginary_modes")),
        ("phonopy calc pk", output_params.get("phonopy_pk")),
        ("phonopy calc uuid", output_params.get("phonopy_uuid")),
    ]
    lines = ["| Property | Value |", "| --- | --- |"]
    for name, value in rows:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = _fmt(value)
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


def _render_figure_markdown(
    output_params: Dict[str, Any], pk: int, figure_dir: str | Path
) -> str:
    """Render the phonon figure next to the report; return Markdown to embed."""
    calc_uuid = output_params.get("phonopy_uuid") or output_params.get("phonopy_pk")
    if not calc_uuid:
        return "*no phonopy calc uuid in output_parameters — figure skipped*"

    from aiida.orm import load_node
    from aiida_uranium_workflow.utils.plot.phonon import render_phonon_figure

    try:
        calc = load_node(calc_uuid)
        figure_path = Path(figure_dir) / f"phonon_bands_dos_pk{pk}.png"
        render_phonon_figure(
            calc,
            figure_path,
            band_labels=output_params.get("band_labels"),
            title=f"Phonon bands (workchain pk={pk})",
        )
    except Exception as exc:
        return f"*figure rendering failed: {exc}*"

    return f"![phonon bands + DOS]({figure_path.name})"


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    figure_dir: Optional[str | Path] = None,
) -> str:
    """Generate a complete Markdown report for an ABACUS phonon WorkChain.

    ``figure_dir`` is optional: when given, the band + DOS figure is
    rendered into that directory and embedded in the report.
    """
    lines: list[str] = [
        render_report_header(
            title="Phonon Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params),
        "",
    ]

    if figure_dir:
        lines += [
            "## Band structure / DOS",
            "",
            _render_figure_markdown(output_params, pk, figure_dir),
            "",
        ]

    lines += [render_report_footer()]
    return "\n".join(lines)
