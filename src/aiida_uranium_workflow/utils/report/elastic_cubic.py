"""Cubic elastic-constant (energy–strain) report generation.

Renders the C11 / C12 / C44 result of the
``AbacusCubicElasticWorkChain``: the equilibrium data used (B / V0 / E0,
either supplied as inputs or from the EOS fallback), the D_o / D_m
distortion scans, the fitted δ² coefficients, the cubic
mechanical-stability (Born) checks and — when ``figure_dir`` is given —
a ΔE(δ) figure of both distortions with the fitted parabola (whose
curvature yields c′ and C44).

Where the numbers come from
---------------------------
* ``output_parameters`` (minimal layout)::

      {
        "workflow": "elastic_cubic", "backend": "abacus", "method": "energy",
        "constants_gpa": {"c11": …, "c12": …, "c44": …},
        "fits": {
          "orthorhombic": {"a2_ev": …, "deltas": [...],
                           "energies_ev": [...], "n_points": …, "rms_ev": …},
          "monoclinic":   {...},
        },
      }

  The legacy flat layout (``c11_gpa`` / ``deltas_o`` / …) is still
  accepted, so old nodes keep rendering.
* the WorkChain **inputs** (via ``workchain_node``) carry B / V0 / E0,
  ``crystal_system``, ``relax_internal`` and whether the EOS ran —
  these are intentionally not duplicated into the outputs.
* c′ = (C11 − C12)/2 and the 6×6 Voigt tensor are derived here.
"""

from __future__ import annotations

from ._common import (
    render_report_footer,
    render_report_header,
)
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _fmt(value: Any, prec: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        try:
            return "[" + ", ".join(f"{float(v):.{prec}f}" for v in value[:8]) + (
                "…]" if len(value) > 8 else "]"
            )
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def _input_value(workchain_node, name):
    """Read one WorkChain input (returns ``None`` when absent)."""
    if workchain_node is None:
        return None
    try:
        return getattr(workchain_node.inputs, name).value
    except Exception:  # noqa: BLE001 — missing port / non-Node value
        return None


def _normalise(
    output_params: Dict[str, Any],
    workchain_node=None,
) -> Dict[str, Any]:
    """Map either the minimal or the legacy layout onto common keys."""
    data = dict(output_params or {})

    constants = dict(data.get("constants_gpa") or {})
    c11 = constants.get("c11", data.get("c11_gpa"))
    c12 = constants.get("c12", data.get("c12_gpa"))
    c44 = constants.get("c44", data.get("c44_gpa"))
    c_prime = 0.5 * (c11 - c12) if (c11 is not None and c12 is not None) else None

    fits = data.get("fits")
    if not fits:
        # legacy layouts: the earliest one stored a single shared
        # ``deltas`` list, a later one split it into ``deltas_o`` /
        # ``deltas_m``.
        shared_deltas = data.get("deltas")
        fits = {
            "orthorhombic": {
                "a2_ev": data.get("a2_orthorhombic_ev"),
                "deltas": data.get("deltas_o", shared_deltas),
                "energies_ev": data.get("energies_orthorhombic_ev"),
                "n_points": data.get("n_points_o"),
            },
            "monoclinic": {
                "a2_ev": data.get("a2_monoclinic_ev"),
                "deltas": data.get("deltas_m", shared_deltas),
                "energies_ev": data.get("energies_monoclinic_ev"),
                "n_points": data.get("n_points_m"),
            },
        }

    # Equilibrium data / flags: prefer the WorkChain inputs (not echoed
    # into the outputs), fall back to legacy output keys.
    bulk = _input_value(workchain_node, "bulk_modulus")
    volume = _input_value(workchain_node, "volume_gs")
    energy = _input_value(workchain_node, "energy_gs_ev")
    relax = _input_value(workchain_node, "relax_internal")
    crystal_system = _input_value(workchain_node, "crystal_system")

    if workchain_node is not None:
        eos_used = not all(value is not None for value in (bulk, volume, energy))
    else:
        eos_used = data.get("eos_used")

    return {
        "backend": data.get("backend"),
        "method": data.get("method"),
        # this WorkChain is cubic-only, so default the label
        "crystal_system": crystal_system or data.get("crystal_system") or "cubic",
        "eos_used": eos_used,
        "relax_internal": (
            relax if relax is not None else data.get("relax_internal", True)
        ),
        "bulk_modulus_gpa": bulk if bulk is not None else data.get("bulk_modulus_gpa"),
        "volume_gs": volume if volume is not None else data.get("volume_gs"),
        "energy_gs_ev": energy if energy is not None else data.get("energy_gs_ev"),
        "c11_gpa": c11,
        "c12_gpa": c12,
        "c44_gpa": c44,
        "c_prime_gpa": c_prime,
        "elastic_tensor_gpa": (
            data.get("elastic_tensor_gpa")
            if data.get("elastic_tensor_gpa") is not None
            else _tensor_from_constants(c11, c12, c44)
        ),
        "fits": fits,
    }


def _tensor_from_constants(c11, c12, c44):
    """Build the cubic 6×6 Voigt tensor from C11 / C12 / C44 (GPa)."""
    if c11 is None or c12 is None or c44 is None:
        return None
    c11, c12, c44 = float(c11), float(c12), float(c44)
    return [
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c44],
    ]


def _cubic_stability(data: Dict[str, Any]) -> List[str]:
    """Cubic Born (mechanical-stability) criteria from the constants."""
    c11 = data.get("c11_gpa")
    c12 = data.get("c12_gpa")
    c44 = data.get("c44_gpa")
    if c11 is None or c12 is None or c44 is None:
        return []
    checks = [
        ("C11 + 2·C12 > 0", float(c11) + 2.0 * float(c12) > 0.0),
        ("C11 − C12 > 0 (c′ > 0)", float(c11) - float(c12) > 0.0),
        ("C44 > 0", float(c44) > 0.0),
    ]
    return [f"* {'OK' if ok else 'FAILED'} — {name}" for name, ok in checks]


def generate_summary_table(data: Dict[str, Any]) -> str:
    """Render the cubic elastic-constant summary table."""
    relax = data.get("relax_internal")
    rows = [
        ("backend", data.get("backend")),
        ("method", data.get("method")),
        ("crystal system", data.get("crystal_system")),
        (
            "internal relaxation",
            "on (relaxed-ion)"
            if relax
            else ("off (clamped-ion)" if relax is not None else None),
        ),
        (
            "equilibrium data",
            None
            if data.get("eos_used") is None
            else ("EOS fallback" if data.get("eos_used") else "supplied input"),
        ),
        ("bulk modulus B", f"{_fmt(data.get('bulk_modulus_gpa'))} GPa"),
        ("equilibrium volume V0", f"{_fmt(data.get('volume_gs'))} Å³"),
        ("equilibrium energy E0", f"{_fmt(data.get('energy_gs_ev'))} eV"),
        ("c′ = (C11−C12)/2", f"{_fmt(data.get('c_prime_gpa'))} GPa"),
        ("C11", f"{_fmt(data.get('c11_gpa'))} GPa"),
        ("C12", f"{_fmt(data.get('c12_gpa'))} GPa"),
        ("C44", f"{_fmt(data.get('c44_gpa'))} GPa"),
    ]
    lines = ["| Property | Value |", "| --- | --- |"]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


def generate_tensor_table(tensor) -> str:
    """Render the 6×6 Voigt elastic tensor as a Markdown table."""
    if tensor is None:
        return "*no elastic tensor (constants missing)*"
    array = np.asarray(tensor, dtype=float)
    if array.shape != (6, 6):
        return f"*elastic tensor has unexpected shape {array.shape}*"
    lines = ["| Cij (GPa) | 1 | 2 | 3 | 4 | 5 | 6 |", "| --- |" * 7]
    for index, row in enumerate(array, start=1):
        lines.append(
            f"| {index} | " + " | ".join(f"{value:.3f}" for value in row) + " |"
        )
    return "\n".join(lines)


def generate_scan_table(data: Dict[str, Any]) -> str:
    """E(δ) table of both distortions plus the fit summary."""
    fits = data.get("fits") or {}
    tables = {}
    for key in ("orthorhombic", "monoclinic"):
        fit = fits.get(key) or {}
        deltas = fit.get("deltas") or []
        energies = fit.get("energies_ev") or []
        tables[key] = {float(d): e for d, e in zip(deltas, energies)}
    lines = [
        "| δ | E(D_o) (eV) | E(D_m) (eV) |",
        "| --- | --- | --- |",
    ]
    for delta in sorted(set(tables["orthorhombic"]) | set(tables["monoclinic"])):
        lines.append(
            f"| {_fmt(delta)} | {_fmt(tables['orthorhombic'].get(delta))} "
            f"| {_fmt(tables['monoclinic'].get(delta))} |"
        )
    for key, title in (("orthorhombic", "D_o"), ("monoclinic", "D_m")):
        fit = fits.get(key) or {}
        lines.append(
            f"| **{title} fit: a₂ / points / rms** | **{_fmt(fit.get('a2_ev'))} eV** "
            f"| **{fit.get('n_points', '—')} pts / {_fmt(fit.get('rms_ev'), 6)} eV** |"
        )
    return "\n".join(lines)


def render_scan_figure(data: Dict[str, Any], pk: int, figure_dir) -> Optional[str]:
    """Render ΔE(δ) for D_o / D_m with the fitted parabola.

    The curvature of each fit gives the elastic constant (c′ from D_o,
    C44 from D_m); both are annotated on the panels. Returns the figure
    file name (relative to ``figure_dir``) or ``None`` when there is no
    scan data.
    """
    fits = data.get("fits") or {}
    if not any(
        (fits.get(key) or {}).get("deltas") for key in ("orthorhombic", "monoclinic")
    ):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    panels = (
        ("orthorhombic", "D_o (orthorhombic)", "c′", data.get("c_prime_gpa"), "tab:blue"),
        ("monoclinic", "D_m (monoclinic)", "C44", data.get("c44_gpa"), "tab:red"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, (key, title, symbol, constant, colour) in zip(axes, panels):
        fit = fits.get(key) or {}
        deltas = np.asarray(fit.get("deltas") or [], dtype=float)
        energies = np.asarray(fit.get("energies_ev") or [], dtype=float)
        if deltas.size == 0 or energies.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(title)
            continue
        delta_e = energies - energies.min()
        ax.plot(deltas, delta_e, "o", color=colour, ms=5, label="SCF points")
        design = np.vstack([np.ones_like(deltas), deltas**2, deltas**4]).T
        coefficients, *_ = np.linalg.lstsq(design, energies, rcond=None)
        dense = np.linspace(0.0, float(deltas.max()) * 1.05, 200)
        dense_design = np.vstack([np.ones_like(dense), dense**2, dense**4]).T
        ax.plot(
            dense,
            (dense_design @ coefficients) - energies.min(),
            "-",
            color=colour,
            lw=1.2,
            label="E₀ + a₂δ² + a₄δ⁴ fit",
        )
        annotation = f"$a_2$ = {float(coefficients[1]):.4g} eV"
        if constant is not None:
            annotation += f"\n{symbol} = {float(constant):.2f} GPa"
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("δ")
        ax.set_ylabel("ΔE (eV / cell)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Elastic constants δ-scan (pk={pk})")
    fig.tight_layout()

    name = f"elastic_cubic_delta_scan_pk{pk}.png"
    fig.savefig(figure_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    figure_dir=None,
    workchain_node=None,
) -> str:
    """Generate a complete Markdown report for an elastic_cubic WorkChain."""
    data = _normalise(output_params, workchain_node)

    lines: List[str] = [
        render_report_header(
            title="Cubic Elastic Constants Report (energy–strain)",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(data),
        "",
        "## Elastic Tensor (Voigt, GPa)",
        "",
        generate_tensor_table(data.get("elastic_tensor_gpa")),
        "",
        "> Voigt order: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy "
        "(shear strain components doubled).",
        "",
        "## D_o / D_m distortion scans",
        "",
        generate_scan_table(data),
        "",
    ]

    if figure_dir is not None:
        try:
            figure_name = render_scan_figure(data, pk, figure_dir)
        except Exception as exc:  # noqa: BLE001 — a figure must not kill the report
            figure_name = None
            lines += [f"*δ-scan figure failed: {exc}*", ""]
        if figure_name:
            lines += [
                "## Quadratic fits (ΔE vs δ)",
                "",
                f"![elastic constants δ-scan]({figure_name})",
                "",
            ]

    stability = _cubic_stability(data)
    if stability:
        lines += ["## Cubic stability (Born criteria)", "", *stability, ""]

    lines += [render_report_footer()]
    return "\n".join(lines)
