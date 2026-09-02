"""Render band / dos / pdos plots from a :class:`PlotSpec`.

Steps
-----
1. For every ``(backend, pk)`` in the spec, pull the data via the
   :func:`extract_band_dos` registry (handles ABACUS / FLEUR etc.).
2. According to ``is_combined``:

   * ``True``  — overlay every backend's data on a single figure.
   * ``False`` — write one PNG per ``(backend, pk)`` group.

The renderer is AiiDA-free: it only sees the normalized
:class:`BandData` / :class:`DosData` returned by :mod:`extract`.

Conventions
-----------
* Every individual ``(backend, label)`` series gets its own color so
  the legend stays unambiguous (e.g. ``pw_r`` red, ``lcao_r`` blue,
  ``fleur`` green).
* For band mode, the x-axis is the integer k-point index (0, 1, 2,
  …). High-symmetry labels are drawn as vertical guides at the
  indices recorded in ``data.label_numbers``.
* ``ylim`` in *band* mode is treated as an alias for the more
  explicit ``energy_range``; ``xlim`` / ``ylim`` are used for
  *dos* / *pdos* modes.
* ``zero_to_efermi`` shifts the energy axis so E_F = 0.

``band_compare`` mode (mode ``"band_compare"``) pulls every series the
same way, then computes the pairwise η_v / max η / ω matrices
(PRB 98, 085117) via :func:`compare_all` and writes Markdown tables +
heatmap PNGs (:func:`render_band_compare`). It is pure matplotlib
rendering on top of the AiiDA-free comparison functions in
:mod:`aiida_uranium_workflow.utils.plot.compare`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for the plot CLI — install with "
        "`pip install matplotlib`."
    ) from exc

from aiida_uranium_workflow.utils.plot import (
    BandData,
    DosData,
    extract_band_dos,
)
from aiida_uranium_workflow.utils.plot.compare import (
    compare_all,
    format_tables,
)
from aiida_uranium_workflow.cli.plot._loading import PlotSpec


# ---------------------------------------------------------------------------
# Color cycling: every (backend, label) gets its own color.
# ---------------------------------------------------------------------------


_PALETTE = [
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _ensure_profile_loaded() -> None:
    from aiida import load_profile
    load_profile()


def collect_series(
    spec: PlotSpec,
    data_key: str | None = None,
) -> List[Tuple[BandData | DosData, str, str]]:
    """Pull normalized data for every ``(backend, pk)`` pair in ``spec``.

    Returns a flat list of ``(data, backend, label)`` tuples — the
    order is stable and used downstream to assign colors.

    ``data_key`` selects which bundle key to return (defaults to
    ``spec.mode``, e.g. ``"band"`` for band mode); ``band_compare``
    mode passes ``"band"`` explicitly because its spec mode is not a
    bundle key.
    """
    _ensure_profile_loaded()

    flat: List[Tuple[Any, str, str]] = []
    for backend, series in spec.data.items():
        for idx, pk in enumerate(series.pks):
            try:
                bundle = extract_band_dos(pk)
            except Exception as exc:
                print(
                    f"[plot-{spec.mode}] WARNING: skipping {backend}/{pk} "
                    f"({exc})",
                    file=sys.stderr,
                )
                continue
            data = bundle.get(data_key or spec.mode)
            if data is None:
                continue
            label = (
                series.labels[idx]
                if idx < len(series.labels)
                else pk
            )
            flat.append((data, backend, label))
    return flat


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def _kpoint_x(kpoints: np.ndarray) -> np.ndarray:
    """Plain integer x-axis so the k-path ticks land on real k points."""
    return np.arange(len(kpoints))


# ---------------------------------------------------------------------------
# Per-mode renderers
# ---------------------------------------------------------------------------


def _render_band(ax, flat, spec: PlotSpec) -> None:
    """Render every (data, backend, label) onto ``ax`` with its own color."""
    # Walk the flat list and assign a sequential color per series so
    # ``pw_r`` and ``lcao_r`` (same backend) get different colors.
    for series_index, (data, backend, label) in enumerate(flat):
        series_color = _PALETTE[series_index % len(_PALETTE)]
        x = _kpoint_x(data.kpoints)
        for band in data.energies:
            y = (
                band - data.fermi_energy
                if spec.figure.zero_to_efermi
                else band
            )
            ax.plot(x, y, color=series_color, lw=0.8, alpha=0.85)
        # High-sym labels once per series (they're identical across
        # bands of the same workchain). Use the first series only —
        # subsequent series would just overwrite the same ticks.
        if data.label_numbers and not getattr(ax, "_kpath_labeled", False):
            # Merge labels that sit on (almost) the same k index — e.g.
            # the two sides of a seekpath "|" break (H|P) — so the tick
            # labels do not overlap.
            merged_numbers: list[int] = []
            merged_labels: list[str] = []
            for num, lbl in zip(data.label_numbers, data.labels):
                if merged_numbers and num - merged_numbers[-1] <= 1:
                    merged_labels[-1] = f"{merged_labels[-1]}|{lbl}"
                else:
                    merged_numbers.append(num)
                    merged_labels.append(lbl)
            ax.set_xticks(list(merged_numbers))
            ax.set_xticklabels(list(merged_labels), fontsize=9)
            for k_idx in merged_numbers:
                ax.axvline(
                    k_idx, color="grey",
                    lw=0.5, ls="--", alpha=0.6,
                )
            setattr(ax, "_kpath_labeled", True)
        # Legend proxy line.
        ax.plot([], [], color=series_color, lw=2.0,
                label=f"{backend}/{label}")

    if spec.figure.zero_to_efermi:
        ax.axhline(0.0, color="red", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel(spec.figure.xlabel or "k-point index")
    ax.set_ylabel(spec.figure.ylabel or "E - E_F (eV)")

    # Band mode: ylim / energy_range both control the y-axis.
    ylim = spec.figure.ylim or spec.figure.energy_range
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    # Band mode x-axis: an explicit ``figure.xlim`` from the JSON wins;
    # otherwise restrict the axis to the k-point index range (0 ..
    # nk-1) of the longest series, so the bands fill the plot without
    # matplotlib's default margins.
    if spec.figure.xlim:
        ax.set_xlim(spec.figure.xlim[0], spec.figure.xlim[1])
    else:
        nk_max = max((len(data.kpoints) for data, _, _ in flat), default=1)
        ax.set_xlim(0, max(nk_max - 1, 1))

    if spec.figure.title:
        ax.set_title(spec.figure.title)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc=spec.figure.legend_loc, fontsize=8)


def _render_dos(ax, flat, spec: PlotSpec) -> None:
    """Render every (data, backend, label) onto ``ax`` with its own color."""
    for series_index, (data, backend, label) in enumerate(flat):
        series_color = _PALETTE[series_index % len(_PALETTE)]
        x = (
            data.energy - data.fermi_energy
            if spec.figure.zero_to_efermi
            else data.energy
        )
        ax.plot(x, data.total, color=series_color, lw=1.1,
                label=f"{backend}/{label}")

    if spec.figure.zero_to_efermi:
        ax.axvline(0.0, color="red", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel(spec.figure.xlabel or "E - E_F (eV)")
    ax.set_ylabel(spec.figure.ylabel or "DOS (1/eV)")

    if spec.figure.xlim:
        ax.set_xlim(spec.figure.xlim[0], spec.figure.xlim[1])
    if spec.figure.ylim:
        ax.set_ylim(spec.figure.ylim[0], spec.figure.ylim[1])

    if spec.figure.title:
        ax.set_title(spec.figure.title)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc=spec.figure.legend_loc, fontsize=8)


def collect_pdos_series(
    spec: PlotSpec,
):
    """Pull normalized PDOS data for every ``(backend, pk)`` pair in ``spec``.

    Returns a flat list of ``(data, backend, label, orbitals)`` tuples;
    ``orbitals`` is the series' requested orbital subset (empty = all
    available channels). Series whose WorkChain has no ``dos_projected``
    output (run without ``run_proj_dos``) are skipped with a warning.
    """
    from aiida_uranium_workflow.utils.plot.extract import extract_pdos

    _ensure_profile_loaded()

    flat = []
    for backend, series in spec.data.items():
        for idx, pk in enumerate(series.pks):
            try:
                data = extract_pdos(pk)
            except Exception as exc:
                print(
                    f"[plot-{spec.mode}] WARNING: skipping {backend}/{pk} ({exc})",
                    file=sys.stderr,
                )
                continue
            label = (
                series.labels[idx]
                if idx < len(series.labels)
                else pk
            )
            flat.append((data, backend, label, list(series.orbitals)))
    return flat


#: Per-orbital linestyle — distinguishes s / p / d / f *within* one
#: series, while the series colour (below) distinguishes the backends.
_ORBITAL_LINESTYLES = {"s": "-", "p": "--", "d": "-.", "f": ":"}


def _render_pdos(ax, flat, spec: PlotSpec) -> None:
    """Render orbital-projected DOS curves for every series.

    Design: each series (backend) gets one base colour from
    :data:`_PALETTE`; its orbital channels are drawn in that colour with
    per-orbital linestyles (s solid, p dashed, d dash-dot, f dotted),
    and its ``total`` is the same colour as a thick solid line. Two
    series (e.g. ABACUS vs FLEUR) therefore stay clearly distinguishable
    even where their DOS look alike.
    """
    for series_index, (data, backend, label, orbitals) in enumerate(flat):
        series_color = _PALETTE[series_index % len(_PALETTE)]
        x = (
            data.energy - data.fermi_energy
            if spec.figure.zero_to_efermi
            else data.energy
        )
        chosen = [o for o in (orbitals or list(data.orbitals)) if o in data.orbitals]
        for orb in chosen:
            ax.plot(
                x, data.orbitals[orb],
                color=series_color,
                ls=_ORBITAL_LINESTYLES.get(orb, "-"),
                lw=1.0, alpha=0.85,
                label=f"{backend}/{label} {orb}",
            )
        ax.plot(x, data.total, color=series_color, lw=1.8, ls="-",
                label=f"{backend}/{label} total")

    if spec.figure.zero_to_efermi:
        ax.axvline(0.0, color="red", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel(spec.figure.xlabel or "E - E_F (eV)")
    ax.set_ylabel(spec.figure.ylabel or "PDOS (1/eV)")

    if spec.figure.xlim:
        ax.set_xlim(spec.figure.xlim[0], spec.figure.xlim[1])
    if spec.figure.ylim:
        ax.set_ylim(spec.figure.ylim[0], spec.figure.ylim[1])

    if spec.figure.title:
        ax.set_title(spec.figure.title)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc=spec.figure.legend_loc, fontsize=8)


_RENDERERS = {
    "band": _render_band,
    "dos":  _render_dos,
    "pdos": _render_pdos,
}


# ---------------------------------------------------------------------------
# band_compare mode
# ---------------------------------------------------------------------------


def _heatmap(
    labels,
    matrix: List[List[float]],
    *,
    metric: str,
    unit: str = "eV",
    out_path: Path,
) -> None:
    """Render a pairwise matrix as a labelled heatmap PNG.

    The colour scale auto-scales to each matrix's own range (so η_v and
    max η heatmaps are each readable on their own); the printed values
    make exact comparison across metrics possible.
    """
    import matplotlib.colors as mcolors

    data = np.ma.masked_invalid(np.asarray(matrix, dtype=float))
    finite = np.asarray(matrix, dtype=float)[np.isfinite(np.asarray(matrix, dtype=float))]
    mval = float(finite.max()) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 2.0,
                                     0.9 * len(labels) + 1.5))
    im = ax.imshow(
        data, cmap="viridis",
        norm=mcolors.Normalize(vmin=0.0, vmax=None),
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    threshold = mval * 0.6
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            val = matrix[i][j]
            if isinstance(val, float) and np.isnan(val):
                continue
            colour = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=colour)
    fig.colorbar(im, ax=ax, label=f"{metric} ({unit})", shrink=0.85)
    ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_band_compare(spec: PlotSpec, out_dir: Path) -> List[Path]:
    """Compare every band structure in ``spec`` pairwise and write
    Markdown tables + heatmap PNGs for η_v, max η and ω.

    Band structures are pulled through the same
    :func:`collect_series` path as band mode (so ABACUS and FLEUR
    WorkChain pks both work), then paired via
    :func:`compare_all` (PRB 98, 085117). Pairs with no common
    k-points / incompatible bands are marked ``—`` in the tables and
    masked in the heatmaps.

    Outputs (all in ``out_dir``):
      ``<stem>.md``        — the three Markdown tables
      ``<stem>_eta_v.png`` / ``<stem>_max_eta.png`` / ``<stem>_omega.png``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = collect_series(spec, data_key="band")
    if not flat:
        raise RuntimeError(
            f"No data could be extracted for spec {spec.figure.fig_name!r}."
        )

    # Labels include the backend so same-named presets stay distinct
    # (e.g. "abacus/pw_r" vs "fleur/lapw"). k-point weights ride along
    # when the backend stored them (needed for the weighted η_v sum).
    series = []
    for data, backend, label in flat:
        entry: list = [
            f"{backend}/{label}", data.energies, data.kpoints, data.fermi_energy,
        ]
        if data.weights is not None:
            entry.append(data.weights)
        series.append(tuple(entry))
    if len(series) < 2:
        raise RuntimeError(
            "band_compare needs at least two band structures in "
            f"spec {spec.figure.fig_name!r}; got {len(series)}."
        )

    align = (
        None if spec.figure.align in ("auto", "") else spec.figure.align
    )
    result = compare_all(
        series,
        sigma=spec.figure.sigma,
        e_min=spec.figure.e_min,
        e_max=spec.figure.e_max,
        align=align,
        occupied_only=spec.figure.occupied_only,
    )
    labels = list(result["labels"])
    stem = Path(spec.figure.fig_name).stem

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(format_tables(result), encoding="utf-8")

    out_paths: List[Path] = [md_path]
    for metric, matrix in (
        ("η_v", result["eta_v"]),
        ("max η", result["max_eta"]),
        ("ω (rigid shift)", result["omega"]),
    ):
        png = out_dir / f"{stem}_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('η', 'eta').replace('ω', 'omega')}.png"
        _heatmap(labels, matrix, metric=metric, out_path=png)
        out_paths.append(png)
    return out_paths


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def render_spec(spec: PlotSpec, out_dir: Path) -> List[Path]:
    """Render every curve in ``spec`` and return the PNG paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.mode == "pdos":
        flat = collect_pdos_series(spec)
    else:
        flat = collect_series(spec)
    if not flat:
        raise RuntimeError(
            f"No data could be extracted for spec {spec.figure.fig_name!r}."
        )

    renderer = _RENDERERS.get(spec.mode)
    if renderer is None:
        raise NotImplementedError(
            f"Rendering mode {spec.mode!r} is not implemented yet "
            f"(supported: {sorted(_RENDERERS)})."
        )

    out_paths: List[Path] = []

    if spec.is_combined:
        fig, ax = plt.subplots(figsize=(8, 5))
        renderer(ax, flat, spec)
        fig.tight_layout()
        out_path = out_dir / spec.figure.fig_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    else:
        for item in flat:
            if spec.mode == "pdos":
                data, backend, label, _orbitals = item
            else:
                data, backend, label = item
            fig, ax = plt.subplots(figsize=(8, 5))
            renderer(ax, [item], spec)
            fig.tight_layout()
            safe_label = label.replace("/", "_")
            stem = Path(spec.figure.fig_name).stem
            ext = Path(spec.figure.fig_name).suffix or ".png"
            out_path = out_dir / f"{stem}_{backend}_{safe_label}{ext}"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            out_paths.append(out_path)
    return out_paths


__all__ = ["render_spec", "render_band_compare", "collect_series"]