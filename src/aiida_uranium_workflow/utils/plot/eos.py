"""EOS (energy vs volume) curve figure rendering.

Given the EOS scan points and the fitted Birch-Murnaghan parameters,
render the energy-volume curve with the fitted equation of state
overlaid. Used by the eos report generator
(:mod:`aiida_uranium_workflow.utils.report.eos`).

The fit is performed on **per-atom** energy/volume (see
``AbacusEosWorkChain._fit_eos``); when ``natoms`` is given the figure
therefore plots per-atom quantities (identical to the totals for a
single-atom cell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

#: eV per Å³ → GPa (the BM energy formula needs B in eV/Å³).
EV_PER_A3_TO_GPA = 160.217733


def birch_murnaghan_energy(
    volumes: Sequence[float],
    volume0: float,
    energy0: float,
    bulk_modulus_gpa: float,
    bulk_deriv: float,
) -> np.ndarray:
    """Evaluate the 3rd-order Birch-Murnaghan E(V) at ``volumes``.

    ``volume0`` / ``energy0`` / ``volumes`` must share the same
    per-atom-or-total convention; ``bulk_modulus_gpa`` is converted to
    eV/Å³ internally (B is intensive, so the per-atom fit value applies
    unchanged).

    E(V) = E0 + (9·V0·B0)/16 · { [(V0/V)^(2/3) − 1]³·B′
           + [(V0/V)^(2/3) − 1]²·[6 − 4·(V0/V)^(2/3)] }
    """
    v = np.asarray(volumes, dtype=float)
    x = (volume0 / v) ** (2.0 / 3.0)
    bulk = bulk_modulus_gpa / EV_PER_A3_TO_GPA  # eV/Å³
    return energy0 + (9.0 * volume0 * bulk / 16.0) * (
        (x - 1.0) ** 3 * bulk_deriv + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


def render_eos_figure(
    output_path: Union[str, Path],
    volumes: Sequence[float],
    energies: Sequence[float],
    *,
    volume_gs: Optional[float] = None,
    energy_gs: Optional[float] = None,
    bulk_modulus_gpa: Optional[float] = None,
    bulk_deriv: Optional[float] = None,
    natoms: Optional[int] = None,
    fit_name: str = "birchmurnaghan",
    title: Optional[str] = None,
) -> Path:
    """Render the EOS figure (scan points + fitted curve) to ``output_path``.

    When the fit parameters are missing (e.g. a fit error) only the scan
    points are plotted. Returns the written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-atom convention when the number of atoms is known.
    per_atom = bool(natoms)
    factor = natoms if per_atom else 1
    x = np.asarray(volumes, dtype=float) / factor
    y = np.asarray(energies, dtype=float) / factor

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(x, y, s=42, c="#1f77b4", zorder=3, label="SCF points")

    has_fit = all(
        v is not None for v in (volume_gs, energy_gs, bulk_modulus_gpa, bulk_deriv)
    )
    if has_fit:
        v0 = volume_gs / factor
        e0 = energy_gs / factor
        lo, hi = float(x.min()), float(x.max())
        span = max(hi - lo, 1e-12)
        v_curve = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 200)
        e_curve = birch_murnaghan_energy(
            v_curve, v0, e0, bulk_modulus_gpa, bulk_deriv
        )
        ax.plot(v_curve, e_curve, "-", c="#d62728", lw=1.6,
                label=f"{fit_name} fit")
        ax.axvline(v0, color="0.5", ls="--", lw=1.0)
        ax.annotate(
            f"$V_0$ = {v0:.3f}",
            xy=(v0, e0),
            xytext=(0.02, 0.92),
            textcoords="axes fraction",
            fontsize=9,
        )

    ax.set_xlabel("Volume per atom (Å³/atom)" if per_atom else "Volume (Å³)")
    ax.set_ylabel("Energy per atom (eV/atom)" if per_atom else "Energy (eV)")
    ax.set_title(title or "EOS fit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
