"""Relax WorkChain report generation module.

Renders the relaxed lattice constants, volume and energy for the
**plugin** relax WorkChains (``abacus.relax`` / ``fleur.relax``).

Two data sources:

* ``workchain_node`` (preferred) — the report CLI passes the finished
  plugin WorkChain node; lattice / volume / energy are derived from its
  raw outputs:
    - ABACUS: ``outputs.structure`` + ``outputs.misc['total_energy']`` (eV)
    - FLEUR:  ``outputs.optimized_structure`` +
      ``outputs.output_relax_wc_para['last_energy']`` (Hartree → eV)
* ``output_params`` (legacy) — the combined ``output_parameters`` dict
  produced by the former in-repo wrapper WorkChains (kept for reports
  against nodes created before the refactor).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ._common import (
    render_report_footer,
    render_report_header,
)

#: eV per Hartree (FLEUR energies are reported in Hartree).
HA_TO_EV = 27.211386245988


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


def _derive_from_node(node, workflow_type: str) -> Dict[str, Any]:
    """Derive lattice constants / volume / energy from a plugin relax node."""
    if workflow_type == "abacus":
        structure = node.outputs.structure
        try:
            energy = node.outputs.misc.get_dict().get("total_energy")
        except (AttributeError, KeyError):
            energy = None
    else:  # fleur
        structure = node.outputs.optimized_structure
        energy = None
        try:
            para = node.outputs.output_relax_wc_para.get_dict()
            # ``last_energy`` is the final total energy in **eV**
            # (aiida-fleur 2.0.0 already converted from Hartree; e.g.
            # bcc-U 2 atoms ≈ -1 528 354 eV, matching the EOS report).
            # The parallel ``energy`` field is the raw value / 27.2114
            # and must not be used.
            e_raw = para.get("last_energy")
            if isinstance(e_raw, (list, tuple)):
                e_raw = e_raw[0] if e_raw else None
            if e_raw is not None:
                energy = float(e_raw)  # already eV
        except (AttributeError, KeyError):
            pass

    cell = np.asarray(structure.cell)
    lengths = [float(x) for x in np.linalg.norm(cell, axis=1)]
    return {
        "lattice_constants": lengths,
        "lattice_constant": lengths[0],
        "volume": float(structure.get_cell_volume()),
        "energy": energy,
        "energy_units": "eV",
    }


def generate_summary_table(output_params: Dict[str, Any], workflow_type: str) -> str:
    """Render the relaxed lattice / volume / energy summary."""
    lines = ["| Property | Value |", "| --- | --- |"]
    lines.append(f"| backend | {workflow_type} |")

    if workflow_type == "fleur":
        lines.append("| mode | full relax (positions + cell) |")
    else:
        lines.append("| mode | volume-only relax |")
    lines.append(
        f"| relaxed lattice constants | "
        f"{_fmt(output_params.get('lattice_constants'), 6)} Å |"
    )
    lines.append(f"| relaxed volume | {_fmt(output_params.get('volume'))} Å³ |")
    lines.append(f"| relaxed energy | {_fmt(output_params.get('energy'))} eV |")
    return "\n".join(lines)


def generate_eos_scan_table(output_params: Dict[str, Any]) -> str:
    """Render the EOS volume-scan table (scaling / volume / energy)."""
    scaling = output_params.get("eos_scaling") or []
    volumes = output_params.get("eos_volumes") or []
    energies = output_params.get("eos_energies") or []

    if not volumes:
        return "No EOS scan data available."

    lines = ["| scale | volume (Å³) | energy (eV) |", "| --- | --- | --- |"]
    for i, vol in enumerate(volumes):
        scale = scaling[i] if i < len(scaling) else None
        energy = energies[i] if i < len(energies) else None
        lines.append(f"| {_fmt(scale, 4)} | {_fmt(vol, 4)} | {_fmt(energy, 6)} |")
    return "\n".join(lines)


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    workchain_node=None,
) -> str:
    """Generate a complete Markdown report for a relax WorkChain.

    ``workchain_node`` is the finished plugin relax WorkChain node; when
    given, the relaxed lattice / volume / energy are derived from its
    outputs instead of ``output_params``.
    """
    if workchain_node is not None:
        data = _derive_from_node(workchain_node, workflow_type)
        output_params = dict(output_params)
        output_params.update(data)

    lines: List[str] = [
        render_report_header(
            title="Relax WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Relaxed Structure",
        "",
        generate_summary_table(output_params, workflow_type),
        "",
    ]

    lines += [render_report_footer()]
    return "\n".join(lines)
