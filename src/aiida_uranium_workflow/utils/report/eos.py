"""EOS WorkChain report generation.

Renders the volume–energy scan list and the fitted equation-of-state
curve parameters (V0 / B0 / B0′ / E0 and the scaling at the minimum)
for either backend, plus — when ``figure_dir`` is given — a plot of the
EOS curve with the fitted Birch-Murnaghan function overlaid. Two data
sources:

* ``output_params`` — the ``output_parameters`` of the in-repo
  ``AbacusEosWorkChain`` (or a legacy dict).
* ``workchain_node`` (preferred for FLEUR) — the plugin
  ``FleurEosWorkChain`` node; its ``output_eos_wc_para`` (and, when
  present, ``output_eos_wc_structure``) are read directly when given.

Key mapping notes
-----------------

* ABACUS stores the scan under ``scales`` / ``volumes`` /
  ``energies_ev`` and the fit under ``volume_gs`` / ``energy_gs_ev`` /
  ``bulk_modulus`` / ``bulk_deriv``.
* FLEUR (plugin ``fleur.eos``) stores the scan under ``scaling`` /
  ``volumes`` / ``total_energy`` and the fit under ``volume_gs`` /
  ``bulk_modulus`` / ``bulk_deriv`` / ``scaling_gs``. It does **not**
  report ``energy_gs_ev`` (derived here as the minimum of the scan) nor
  ``lattice_constant_gs`` (derived from ``output_eos_wc_structure``).
"""

from __future__ import annotations

from ._common import (
    render_report_footer,
    render_report_header,
)
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _fmt(value: Any, prec: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        try:
            return "[" + ", ".join(f"{float(v):.{prec}f}" for v in value[:8])
            +("…]" if len(value) > 8 else "]")
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map either backend's result dict onto common keys."""
    # FLEUR's plugin stores the scan energies under ``total_energy``,
    # ABACUS under ``energies_ev``.
    energies = data.get("energies_ev", data.get("energies", data.get("total_energy")))
    volumes = data.get("volumes")
    natoms = data.get("natoms")

    # The FLEUR plugin does not report E0; take the scan minimum.
    energy_gs = data.get("energy_gs_ev", data.get("energy_gs"))
    if energy_gs is None and energies:
        try:
            energy_gs = min(float(e) for e in energies if e is not None)
        except (TypeError, ValueError):
            energy_gs = None

    # Per-atom ground-state quantities: prefer the explicit keys
    # (ABACUS fit output), else derive from the cell totals (FLEUR).
    volume_gs = data.get("volume_gs")
    volume_gs_per_atom = data.get("volume_gs_per_atom")
    if volume_gs_per_atom is None and volume_gs is not None and natoms:
        volume_gs_per_atom = float(volume_gs) / natoms
    energy_gs_per_atom = data.get("energy_gs_per_atom_ev")
    if energy_gs_per_atom is None and energy_gs is not None and natoms:
        energy_gs_per_atom = float(energy_gs) / natoms

    return {
        "volume_gs": volume_gs,
        "volume_gs_per_atom": volume_gs_per_atom,
        "lattice_constant_gs": data.get("lattice_constant_gs"),
        "scaling_gs": data.get("scaling_gs"),
        "bulk_modulus": data.get("bulk_modulus", data.get("bulk_modulus_gpa")),
        "bulk_deriv": data.get("bulk_deriv"),
        "energy_gs_ev": energy_gs,
        "energy_gs_per_atom_ev": energy_gs_per_atom,
        "fit": data.get("fit", "birchmurnaghan"),
        "scales": data.get("scales", data.get("scaling")),
        "volumes": volumes,
        "volumes_per_atom": data.get("volumes_per_atom"),
        "energies_ev": energies,
        "energies_per_atom_ev": data.get("energies_per_atom_ev"),
        "natoms": natoms,
        "n_points": data.get("n_points") or (len(volumes or []) or None),
        "residuals": data.get("residuals"),
        "error": data.get("error"),
    }


def generate_summary_table(data: Dict[str, Any]) -> str:
    """Render the fitted EOS curve parameters."""
    data = _normalise(data)  # accept raw backend dicts (idempotent)
    rows = [
        ("fit", data.get("fit")),
        ("volume_gs", f"{_fmt(data.get('volume_gs'))} Å³"),
        ("volume_gs_per_atom", f"{_fmt(data.get('volume_gs_per_atom'))} Å³/atom"),
        ("lattice_constant_gs", f"{_fmt(data.get('lattice_constant_gs'))} Å"),
        ("scaling_gs", data.get("scaling_gs")),
        ("bulk_modulus", f"{_fmt(data.get('bulk_modulus'))} GPa"),
        ("bulk_deriv", data.get("bulk_deriv")),
        ("energy_gs", f"{_fmt(data.get('energy_gs_ev'))} eV"),
        ("energy_gs_per_atom", f"{_fmt(data.get('energy_gs_per_atom_ev'))} eV/atom"),
        ("points", data.get("n_points")),
        ("natoms", data.get("natoms")),
    ]
    if data.get("residuals") is not None:
        rows.append(("residuals", data.get("residuals")))
    lines = ["| Property | Value |", "| --- | --- |"]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


def generate_eos_table(data: Dict[str, Any]) -> str:
    """Render the per-point scale / volume / energy list.

    When the number of atoms is known, per-atom volume and energy
    columns are added (the EOS fit itself is done per atom).
    """
    data = _normalise(data)  # accept raw backend dicts (idempotent)
    scales = data.get("scales") or []
    volumes = data.get("volumes") or []
    energies = data.get("energies_ev") or []
    if not volumes:
        return "No EOS scan data available."

    natoms = data.get("natoms")
    volumes_per_atom = data.get("volumes_per_atom") or (
        [v / natoms for v in volumes] if natoms else []
    )
    energies_per_atom = data.get("energies_per_atom_ev") or (
        [e / natoms for e in energies] if natoms else []
    )

    if natoms:
        header = "| scale | volume (Å³) | volume/atom (Å³) | energy (eV) | energy/atom (eV) |"
        sep = "| --- | --- | --- | --- | --- |"
    else:
        header = "| scale | volume (Å³) | energy (eV) |"
        sep = "| --- | --- | --- |"
    lines = [header, sep]
    for i, vol in enumerate(volumes):
        scale = scales[i] if i < len(scales) else None
        energy = energies[i] if i < len(energies) else None
        if natoms:
            v_per = volumes_per_atom[i] if i < len(volumes_per_atom) else None
            e_per = energies_per_atom[i] if i < len(energies_per_atom) else None
            lines.append(
                f"| {_fmt(scale, 4)} | {_fmt(vol, 4)} | {_fmt(v_per, 4)} "
                f"| {_fmt(energy, 6)} | {_fmt(e_per, 6)} |"
            )
        else:
            lines.append(f"| {_fmt(scale, 4)} | {_fmt(vol, 4)} | {_fmt(energy, 6)} |")
    return "\n".join(lines)


def _lattice_constant_from_structure(structure) -> Any:
    """Return the lattice constant (Å) for cubic cells, else None.

    Mirrors the ABACUS EOS fit: a single value only makes sense when all
    three cell lengths agree.
    """
    cell = np.asarray(structure.cell, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if np.allclose(lengths, lengths[0], atol=1e-3):
        return float(lengths[0])
    return None


def _render_figure_markdown(
    data: Dict[str, Any], pk: int, figure_dir: str | Path, report_stem: str | None = None
) -> str:
    """Render the EOS curve next to the report; return Markdown to embed.

    The image is named after the report file when ``report_stem`` is
    given (``<report_stem>_eos_curve.png``, e.g. the CLI writes
    ``report_nosoc_e1a9fcac_eos_curve.png`` next to
    ``report_nosoc_e1a9fcac.md``); otherwise it falls back to
    ``eos_curve_pk<pk>.png``. Falls back to a note (no crash) when the
    scan data or matplotlib is unavailable.
    """
    volumes = data.get("volumes") or []
    energies = data.get("energies_ev") or []
    if not volumes or not energies:
        return "*no EOS scan data — figure skipped*"

    figure_name = (
        f"{report_stem}_eos_curve.png" if report_stem else f"eos_curve_pk{pk}.png"
    )
    figure_path = Path(figure_dir) / figure_name
    try:
        from aiida_uranium_workflow.utils.plot.eos import render_eos_figure

        render_eos_figure(
            figure_path,
            volumes,
            energies,
            volume_gs=data.get("volume_gs"),
            energy_gs=data.get("energy_gs_ev"),
            bulk_modulus_gpa=data.get("bulk_modulus"),
            bulk_deriv=data.get("bulk_deriv"),
            natoms=data.get("natoms"),
            fit_name=str(data.get("fit") or "birchmurnaghan"),
            title=f"EOS fit ({str(data.get('fit') or 'birchmurnaghan')})",
        )
    except Exception as exc:
        return f"*figure rendering failed: {exc}*"

    return f"![EOS curve]({figure_name})"


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    workchain_node=None,
    figure_dir: str | Path | None = None,
    report_stem: str | None = None,
) -> str:
    """Generate a complete Markdown report for an EOS WorkChain.

    ``figure_dir`` is optional: when given, the energy-volume curve is
    rendered into that directory and embedded in the report (the report
    CLI passes the report output directory automatically). ``report_stem``
    (the report filename without extension) names the figure after the
    report, e.g. ``report_nosoc_e1a9fcac_eos_curve.png`` next to
    ``report_nosoc_e1a9fcac.md``.
    """
    if workchain_node is not None:
        # FLEUR plugin: read output_eos_wc_para (+ ground-state structure
        # for the lattice constant).
        try:
            para_node = workchain_node.outputs.output_eos_wc_para
            output_params = dict(output_params)
            output_params.update(para_node.get_dict())
        except (AttributeError, KeyError):
            pass

        # Derive the equilibrium lattice constant from the GS structure
        # whenever the fit dict does not carry it.
        if output_params.get("lattice_constant_gs") is None:
            for port_name in ("output_eos_wc_structure", "optimized_structure"):
                try:
                    structure = workchain_node.outputs[port_name]
                except (AttributeError, KeyError):
                    continue
                lattice = _lattice_constant_from_structure(structure)
                if lattice is not None:
                    output_params = dict(output_params)
                    output_params["lattice_constant_gs"] = lattice
                break

    data = _normalise(output_params)

    lines: List[str] = [
        render_report_header(
            title="Equation of State Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(data),
        "",
        "## EOS scan (volume vs energy)",
        "",
        generate_eos_table(data),
        "",
    ]

    if figure_dir:
        lines += [
            "## EOS curve",
            "",
            _render_figure_markdown(data, pk, figure_dir, report_stem),
            "",
        ]

    if data.get("error"):
        lines += ["## Error", "", f"*{data['error']}*", ""]
    lines += [render_report_footer()]
    return "\n".join(lines)
