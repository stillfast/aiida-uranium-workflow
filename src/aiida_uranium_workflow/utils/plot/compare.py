"""Band-structure comparison metrics (η_v and max η).

Implements the band-distance metrics of
*Phys. Rev. B 98, 085117* (BFC band comparison):

.. math::

    \\eta_v(A, B) = \\min_\\omega
        \\sqrt{ \\frac{ \\sum_{n\\mathbf{k}} \\tilde{f}_{n\\mathbf{k}}
            (\\varepsilon_{n\\mathbf{k}}^A - \\varepsilon_{n\\mathbf{k}}^B + \\omega)^2 }
            { \\sum_{n\\mathbf{k}} \\tilde{f}_{n\\mathbf{k}} } }

    \\tilde{f}_{n\\mathbf{k}} =
        \\sqrt{ f_{n\\mathbf{k}}(\\varepsilon_F^A, \\sigma)
                f_{n\\mathbf{k}}(\\varepsilon_F^B, \\sigma) }

    \\mathrm{max}\\,\\eta = \\max |\\varepsilon_{n\\mathbf{k}}^A
        - \\varepsilon_{n\\mathbf{k}}^B + \\omega|

where ``f`` is the Fermi–Dirac distribution at temperature width
``sigma`` and ``omega`` is a rigid energy shift found by minimising
:math:`\\eta_v`.

Only **occupied** bands matter (the ``f`` weight suppresses empty
bands), so the metrics are dominated by the valence bands — this is
what makes them suitable for comparing band structures of the same
system computed with different codes / settings.

The two band structures must share the same k-points (the common
subset is used). How the states are paired is controlled by ``align``:

* ``"window"`` — per-k energy-sorted pairing inside an energy window
  (default when ``e_min`` / ``e_max`` is given); robust to codes that
  store a different number of deep core states.
* ``"index"`` — by band index (paper definition, default without a
  window).

``occupied_only`` restricts max η to states occupied in both structures
(Fermi–Dirac weight > 0.5), matching the SSSP ``max_diff`` definition.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _fermi_dirac(energy: np.ndarray, e_fermi: float, sigma: float) -> np.ndarray:
    """Fermi–Dirac occupation factor ``1 / (1 + exp((E - E_F)/sigma))``.

    ``sigma`` is the smearing width in the same units as ``energy``
    (eV). Guarded against overflow.
    """
    x = (energy - e_fermi) / sigma
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(x))


def _match_kpoints(
    kpoints_a: np.ndarray,
    kpoints_b: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """Return ``(keep_a, idx_b)`` pairing A's k-points to B's.

    k-points are matched by Cartesian coordinates (atol 1e-6); the
    result orders the common subset by A's k-points. Used to reorder
    band arrays *and* any per-k auxiliary arrays (weights) together.
    """
    if kpoints_a.shape != kpoints_b.shape:
        raise ValueError(
            "k-points have different shapes "
            f"({kpoints_a.shape} vs {kpoints_b.shape})"
        )
    idx_b: List[int] = []
    keep_a: List[int] = []
    for i, ka in enumerate(kpoints_a):
        for j, kb in enumerate(kpoints_b):
            if np.allclose(ka, kb, atol=1e-6):
                idx_b.append(j)
                keep_a.append(i)
                break
    if not keep_a:
        raise ValueError("No matching k-points between the two band structures.")
    return keep_a, idx_b


def _common_subset(
    e_a: np.ndarray,
    e_b: np.ndarray,
    kpoints_a: np.ndarray,
    kpoints_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict both band structures to their common k-points.

    ``e_a`` / ``e_b`` have shape (nbands, nk). k-points are matched by
    Cartesian coordinates (atol 1e-6). Returns the matched subsets in
    the order of ``A``'s k-points.
    """
    keep_a, idx_b = _match_kpoints(kpoints_a, kpoints_b)
    return (
        e_a[:, keep_a],
        e_b[:, idx_b],
        kpoints_a[keep_a],
        kpoints_b[idx_b],
    )


def compare_bands(
    energies_a: np.ndarray,
    energies_b: np.ndarray,
    *,
    kpoints_a: np.ndarray | None = None,
    kpoints_b: np.ndarray | None = None,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    fermi_a: float = 0.0,
    fermi_b: float = 0.0,
    sigma: float = 0.1,
    e_min: float | None = None,
    e_max: float | None = None,
    align: str | None = None,
    occupied_only: bool = False,
) -> dict:
    """Compute ``(eta_v, max_eta, omega)`` between two band structures.

    Parameters
    ----------
    energies_a / energies_b : (nbands, nk) arrays in eV.
    kpoints_a / kpoints_b : optional (nk, 3) arrays; when given, the
        two structures are restricted to their common k-points.
    weights_a / weights_b : optional (nk,) k-point weights (degeneracy
        / path segment lengths); the geometric mean enters the η_v sum.
    fermi_a / fermi_b : Fermi energies (eV) of A and B.
    sigma : Fermi–Dirac smearing width (eV).
    e_min / e_max : optional energy window (eV, relative to each
        structure's own E_F) restricting which states enter the
        comparison.
    align : how the two band structures are paired:

        * ``"window"`` — per k-point, pair the windowed states sorted by
          energy. This aligns physically corresponding states even when
          the two codes store a different number of deep core states
          (which would otherwise shift every band index).
        * ``"index"`` — by band index (paper definition; requires the
          same band count, truncated to the common number).

        Defaults to ``"window"`` when ``e_min`` / ``e_max`` is given,
        otherwise ``"index"``.
    occupied_only : when True, max η is restricted to states where both
        structures are occupied (Fermi–Dirac weight > 0.5), matching the
        SSSP ``max_diff`` semantics.

    Returns
    -------
    dict with ``eta_v`` (eV), ``max_eta`` (eV), ``omega`` (eV), the
    number of (band, k) points used and the resolved ``align``.
    """
    e_a = np.asarray(energies_a, dtype=float)
    e_b = np.asarray(energies_b, dtype=float)
    w_a = None if weights_a is None else np.asarray(weights_a, dtype=float).ravel()
    w_b = None if weights_b is None else np.asarray(weights_b, dtype=float).ravel()

    if kpoints_a is not None and kpoints_b is not None:
        keep_a, idx_b = _match_kpoints(
            np.asarray(kpoints_a), np.asarray(kpoints_b)
        )
        e_a, e_b = e_a[:, keep_a], e_b[:, idx_b]
        if w_a is not None and w_a.size == len(keep_a):
            w_a = w_a[keep_a]
        if w_b is not None and w_b.size == len(idx_b):
            w_b = w_b[idx_b]

    if align is None:
        align = "window" if (e_min is not None or e_max is not None) else "index"
    if align not in ("index", "window"):
        raise ValueError(
            f"align={align!r} not in ('index', 'window')"
        )

    # ------------------------------------------------------------------
    # State selection + pairing
    # ------------------------------------------------------------------
    if align == "index":
        nbands = min(e_a.shape[0], e_b.shape[0])
        e_a, e_b = e_a[:nbands], e_b[:nbands]
        if e_a.shape != e_b.shape:
            raise ValueError(
                f"Band arrays have incompatible shapes: {e_a.shape} vs {e_b.shape}"
            )
        # Fermi–Dirac weights (geometric mean), evaluated at each band's
        # own energy relative to its own E_F; multiplied by the
        # geometric mean of the k-point weights when both are given.
        fa = _fermi_dirac(e_a, fermi_a, sigma)
        fb = _fermi_dirac(e_b, fermi_b, sigma)
        occ_raw = np.sqrt(fa * fb)
        f_tilde = occ_raw.copy()
        if w_a is not None and w_b is not None and w_a.size == e_a.shape[1]:
            f_tilde = f_tilde * np.sqrt(w_a * w_b)[None, :]
        elif w_a is not None and w_a.size == e_a.shape[1]:
            f_tilde = f_tilde * w_a[None, :]
        in_window = np.ones(f_tilde.shape, dtype=bool)
        # ``delta`` uses energies relative to each structure's own E_F,
        # so the fitted ω is the rigid shift between the band shapes,
        # not a reference-frame offset (ABACUS stores absolute energies
        # with E_F ≈ 27.7 eV while the FLEUR recipe already shifts to
        # E_F ≈ 0).
        delta = (e_a - fermi_a) - (e_b - fermi_b)
    else:  # align == "window"
        lo = e_min if e_min is not None else -np.inf
        hi = e_max if e_max is not None else np.inf
        ra = e_a - fermi_a   # relative to own E_F
        rb = e_b - fermi_b
        pair_a: List[np.ndarray] = []
        pair_b: List[np.ndarray] = []
        pair_ft: List[np.ndarray] = []
        pair_occ: List[np.ndarray] = []
        pair_w: List[np.ndarray] = []
        for k in range(ra.shape[1]):
            wa = np.sort(ra[(ra[:, k] >= lo) & (ra[:, k] <= hi), k])
            wb = np.sort(rb[(rb[:, k] >= lo) & (rb[:, k] <= hi), k])
            m = min(wa.size, wb.size)
            if m == 0:
                continue
            wa, wb = wa[:m], wb[:m]
            fa_k = _fermi_dirac(wa, 0.0, sigma)
            fb_k = _fermi_dirac(wb, 0.0, sigma)
            pair_a.append(wa)
            pair_b.append(wb)
            pair_ft.append(np.sqrt(fa_k * fb_k))
            pair_occ.append(np.sqrt(fa_k * fb_k))
            if w_a is not None and w_b is not None and k < w_a.size and k < w_b.size:
                pair_w.append(np.full(m, np.sqrt(w_a[k] * w_b[k])))
            elif w_a is not None and k < w_a.size:
                pair_w.append(np.full(m, w_a[k]))
            else:
                pair_w.append(np.ones(m))
        if not pair_a:
            raise ValueError("No (band, k) states inside the energy window.")
        e_a = np.concatenate(pair_a)
        e_b = np.concatenate(pair_b)
        occ_raw = np.concatenate(pair_occ)
        f_tilde = np.concatenate(pair_ft) * np.concatenate(pair_w)
        in_window = np.ones(f_tilde.shape, dtype=bool)
        delta = e_a - e_b

    # Optional energy window on top of index alignment.
    if align == "index" and (e_min is not None or e_max is not None):
        lo = e_min if e_min is not None else -np.inf
        hi = e_max if e_max is not None else np.inf
        in_window = (
            ((e_a - fermi_a) >= lo) & ((e_a - fermi_a) <= hi)
            & ((e_b - fermi_b) >= lo) & ((e_b - fermi_b) <= hi)
        )
        f_tilde = f_tilde * in_window

    wsum = f_tilde.sum()
    if wsum <= 0:
        raise ValueError("No (band, k) points with non-zero weight after filtering.")

    # Optimal rigid shift: minimise Σ f̃ (Δ + ω)²  →  ω = -Σ f̃ Δ / Σ f̃.
    omega = -np.sum(f_tilde * delta) / wsum

    diff = delta + omega
    eta_v = float(np.sqrt(np.sum(f_tilde * diff**2) / wsum))

    # max η: over the selected states; optionally only where both
    # structures are occupied (SSSP ``max_diff`` semantics — empty /
    # weakly-occupied bands no longer pollute the max). No states in
    # the mask → 0 (nothing to compare).
    max_mask = in_window
    if occupied_only:
        max_mask = max_mask & (occ_raw > 0.5)
    if max_mask.any():
        max_eta = float(np.max(np.abs(diff[max_mask])))
    else:
        max_eta = 0.0

    return {
        "eta_v": eta_v,
        "max_eta": max_eta,
        "omega": float(omega),
        "npoints": int(in_window.sum()),
        "align": align,
    }


def compare_all(
    series,
    *,
    sigma: float = 0.1,
    e_min: float | None = None,
    e_max: float | None = None,
    align: str | None = None,
    occupied_only: bool = False,
) -> dict:
    """Pairwise comparison of every band structure.

    ``series`` is a list of tuples::

        (label, energies, kpoints, fermi_energy)
        (label, energies, kpoints, fermi_energy, weights)

    with ``energies`` (nbands, nk) in eV and ``weights`` (nk,) k-point
    weights (optional; the shorter form defaults to uniform weights).
    Returns::

        {
            "labels": [...],
            "eta_v":     [[...] pairwise η_v in eV],
            "max_eta":   [[...] pairwise max η in eV],
            "omega":     [[...] pairwise rigid shift in eV],
        }
    """
    n = len(series)
    eta = np.full((n, n), np.nan)
    mx = np.full((n, n), np.nan)
    omega = np.full((n, n), np.nan)
    labels = [s[0] for s in series]

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = series[i], series[j]
            ea, ka, fa = si[1], si[2], si[3]
            eb, kb, fb = sj[1], sj[2], sj[3]
            wa = si[4] if len(si) > 4 else None
            wb = sj[4] if len(sj) > 4 else None

            try:
                res = compare_bands(
                    ea, eb, kpoints_a=ka, kpoints_b=kb,
                    weights_a=wa, weights_b=wb,
                    fermi_a=fa, fermi_b=fb,
                    sigma=sigma, e_min=e_min, e_max=e_max,
                    align=align,
                    occupied_only=occupied_only,
                )
                eta[i, j] = eta[j, i] = res["eta_v"]
                mx[i, j] = mx[j, i] = res["max_eta"]
                omega[i, j] = omega[j, i] = res["omega"]
            except (ValueError, IndexError):
                # Incompatible k-points / bands — leave NaN.
                pass

    return {
        "labels": labels,
        "eta_v": eta.tolist(),
        "max_eta": mx.tolist(),
        "omega": omega.tolist(),
    }


# ---------------------------------------------------------------------------
# 2D table rendering (Markdown — pure text, no matplotlib needed)
# ---------------------------------------------------------------------------


def _fmt(value: float, ndigits: int = 4) -> str:
    """Format one table cell; ``NaN`` (incompatible pair) renders as ``—``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{ndigits}f}"


def format_table(
    labels: List[str],
    matrix: List[List[float]],
    *,
    metric: str = "η_v",
    unit: str = "eV",
    ndigits: int = 4,
) -> str:
    """Render a pairwise matrix as a Markdown table.

    Row / column labels are the series labels; the diagonal is left
    blank (a band structure compared with itself is trivially 0).
    ``metric`` is only used in the caption line (not the table body),
    so the same formatter serves η_v, max η and ω.
    """
    header = [""] + labels
    lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for i, row_label in enumerate(labels):
        cells = [str(row_label)]
        for j, _ in enumerate(labels):
            if i == j:
                cells.append("—")
            else:
                cells.append(_fmt(matrix[i][j], ndigits))
        lines.append(f"| {' | '.join(cells)} |")
    caption = f"Pairwise {metric} ({unit})"
    return "\n".join([caption, *lines])


def format_tables(result: dict, *, ndigits: int = 4) -> str:
    """Render the full pairwise report (η_v, max η, ω) as Markdown.

    ``result`` is what :func:`compare_all` returns::

        {"labels": [...], "eta_v": [[...]], "max_eta": [[...]], "omega": [[...]]}

    Returns one Markdown document with three tables. This is a pure
    text function — it never touches matplotlib or AiiDA — so it is
    directly unit-testable.
    """
    labels = list(result["labels"])
    sections = [
        format_table(labels, result["eta_v"], metric="η_v", ndigits=ndigits),
        "",
        format_table(labels, result["max_eta"], metric="max η", ndigits=ndigits),
        "",
        format_table(labels, result["omega"], metric="ω (rigid shift)", ndigits=ndigits),
    ]
    return "\n".join(sections)
