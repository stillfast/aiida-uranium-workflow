"""Plot normalized band / DOS data.

This module is the *rendering* layer: it takes the backend-agnostic
:class:`BandData` / :class:`DosData` produced by :mod:`.extract` and
draws matplotlib figures. It does not import AiiDA.

The plotting rules mirror the standalone scripts in
``test_aiida_workchain/banddos/plot_abacus_band.py`` (energy axis
aligned to E_F, k-path labels, etc.) so end users can drop the new
utility in wherever the old scripts were used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: no DISPLAY required
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for utils.plot — install with "
        "`pip install matplotlib`."
    ) from exc

from .extract import BandData, DosData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kpath_axis(kpoints: np.ndarray) -> np.ndarray:
    """Cumulative k-point distance along the seekpath."""
    seg_len = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def plot_band(band: BandData, out_path: Path) -> Path:
    """Render the band structure with E_F at 0 and high-sym labels."""
    cumulative = _kpath_axis(band.kpoints)
    fig, ax = plt.subplots(figsize=(8, 5))
    for energies in band.energies:
        ax.plot(cumulative, energies - band.fermi_energy,
                color="C0", lw=0.8, alpha=0.85)

    if band.label_numbers:
        ax.set_xticks([cumulative[i] for i in band.label_numbers])
        ax.set_xticklabels(band.labels, fontsize=9)
        for k_idx in band.label_numbers:
            ax.axvline(cumulative[k_idx], color="grey",
                       lw=0.5, ls="--", alpha=0.6)

    ax.axhline(0.0, color="red", lw=0.6,
               label=f"E_F = {band.fermi_energy:.4f} eV")
    ax.set_xlabel("k-path")
    ax.set_ylabel("E - E_F [eV]")
    ax.set_title(f"Band structure — pk={band.workchain_pk}")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dos(dos: DosData, out_path: Path) -> Path:
    """Render total DOS with E_F at 0 (and any spin channels)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dos.energy - dos.fermi_energy, dos.total,
            color="black", lw=1.1, label="total")
    for name, arr in dos.spin.items():
        ax.plot(dos.energy - dos.fermi_energy, arr,
                lw=0.7, alpha=0.7, label=name)

    ax.axvline(0.0, color="red", lw=0.6,
               label=f"E_F = {dos.fermi_energy:.4f} eV")
    ax.set_xlabel("E - E_F [eV]")
    ax.set_ylabel("DOS [1/eV]")
    ax.set_title(f"DOS — pk={dos.workchain_pk}")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all(
    band: BandData,
    dos: DosData,
    out_dir: Optional[Path] = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Render both band and DOS, returning the produced paths.

    Parameters
    ----------
    band, dos : BandData, DosData
        Normalized containers (e.g. from :func:`extract_band_dos`).
    out_dir : Path, optional
        Output directory. Defaults to the current working directory.
    prefix : str
        File-name prefix; useful when batch-plotting multiple pk's.
    """
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pk = band.workchain_pk or dos.workchain_pk or "x"
    suffix = f"_pk{pk}" if prefix == "" else f"{prefix}_pk{pk}"
    paths = {
        "band": plot_band(band, out_dir / f"bands{suffix}.png"),
        "dos":  plot_dos(dos,  out_dir / f"dos{suffix}.png"),
    }
    return paths


__all__ = ["plot_band", "plot_dos", "plot_all"]