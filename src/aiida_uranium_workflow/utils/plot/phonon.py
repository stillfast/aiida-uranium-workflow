"""Phonon band-structure + DOS figure rendering.

Shared by the phonopy report generator (:mod:`utils.report.phonopy`) and
the standalone ``aiida-uranium-plot-phonon`` CLI entry point: given any
node exposing ``phonon_bands`` (and optionally ``total_phonon_dos``)
outputs — the ``AbacusPhonopyWorkChain`` itself or the underlying
``PhonopyCalculation`` — render the standard phonon figure.

Label sources, in priority order:

1. explicit ``band_labels`` (list of high-symmetry labels; positions are
   the segment boundaries of the band path);
2. the ``labels`` attribute of the BandsData, when the aiida-phonopy
   parser stored usable ``(index, label)`` pairs;
3. no labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _load_node(node: Union[int, str, Any]):
    """Accept a node object or anything :func:`aiida.orm.load_node` accepts."""
    if hasattr(node, "pk"):
        return node
    from aiida.orm import load_node

    return load_node(node)


def _get_outputs(node) -> dict:
    # CalcJob outputs use CREATE links; WorkChain outputs use RETURN links.
    return {
        link.link_label: link.node
        for link in node.base.links.get_outgoing().all()
        if link.link_type.value in ("create", "return")
    }


def _path_length(kpoints: np.ndarray) -> np.ndarray:
    """Cumulative reciprocal-space path length (in units of |2π/a|)."""
    diff = np.diff(kpoints, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(diff, axis=1))])


def _path_breaks(kpoints: np.ndarray) -> List[int]:
    """Indices where a band path jumps between disconnected segments.

    seekpath-style explicit paths concatenate disjoint high-symmetry
    arcs (e.g. bcc ``Γ-H-N-Γ-P-H | P-N``) with a *connector* that has no
    interior q-points — one giant step far larger than the sampling step.
    Drawing straight through it would show a chord across the BZ (the
    "H→P 直线"). Returns the point index where each new chunk starts.
    """
    if len(kpoints) < 3:
        return []
    steps = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    median_step = float(np.median(steps))
    if median_step <= 0.0:
        return []
    return [int(index) + 1 for index in np.where(steps > 3.0 * median_step)[0]]


def _chunked_x(kpoints: np.ndarray, breaks: List[int]):
    """Return ``(x, chunks)``: cumulative q-distance along each contiguous
    arc, with the arcs concatenated head-to-tail (no gap, no connector
    chord). ``chunks`` are the inclusive point ranges to draw as separate
    polylines so no fake straight line crosses the BZ at a break."""
    chunks: List[tuple] = []
    start = 0
    for break_at in sorted(breaks):
        chunks.append((start, break_at - 1))
        start = break_at
    chunks.append((start, len(kpoints) - 1))

    x = np.zeros(len(kpoints))
    offset = 0.0
    for chunk_start, chunk_end in chunks:
        segment = _path_length(kpoints[chunk_start:chunk_end + 1])
        x[chunk_start:chunk_end + 1] = segment + offset
        offset = x[chunk_end]  # next arc starts right where this one ended
    return x, chunks


def _label_ticks(x: np.ndarray, label_pos, label_text):
    """Return ``(tick_x, tick_text)`` for the high-symmetry labels.

    Two consecutive labels with no interior q-points between them (the
    "connector" of two disconnected arcs, e.g. H→P of the bcc path) share
    the same x and are merged into one tick written ``H|P`` — the
    standard band-structure notation for a path break.
    """
    if label_pos is None:
        return None, None
    positions = [min(int(i), len(x) - 1) for i in label_pos]
    texts = [str(t) for t in label_text]

    tick_x: List[float] = []
    tick_text: List[str] = []
    index = 0
    while index < len(positions):
        merged = [texts[index]]
        while (
            index + 1 < len(positions)
            and positions[index + 1] - positions[index] <= 1
        ):
            index += 1
            merged.append(texts[index])
        tick_x.append(float(x[positions[index]]))
        tick_text.append("|".join(merged))
        index += 1
    return tick_x, tick_text


def _segment_boundaries(n_kpoints: int, n_labels: int) -> List[int]:
    """Indices of the segment boundaries for ``n_labels`` evenly split labels."""
    if n_labels <= 1:
        return [0]
    pts_per_seg = max((n_kpoints - 1) // (n_labels - 1), 1)
    return [min(i * pts_per_seg, n_kpoints - 1) for i in range(n_labels)]


def _geometric_boundaries(kpoints: np.ndarray) -> List[int]:
    """Locate the high-symmetry points of a band path geometrically.

    At every high-symmetry point the q-path either turns or folds back on
    itself (a reversal, dot ≈ -1 — common in the bcc path), and the
    connector jump between two disconnected arcs is an extreme case. The
    boundaries are the point indices where the step direction stops being
    (nearly) collinear with the previous step. Zero-length steps
    (duplicate consecutive q-points) are skipped.
    """
    nk = len(kpoints)
    if nk < 3:
        return [0, max(nk - 1, 0)]
    steps = np.diff(kpoints, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    valid = np.where(norms > 0.0)[0]
    if len(valid) < 2:
        return [0, nk - 1]
    unit = steps[valid] / norms[valid][:, None]
    dots = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    changed = np.where(dots < 0.9)[0]  # between condensed steps c and c+1
    boundaries = {0, nk - 1}
    for condensed in changed:
        # the turn sits at the point where the *later* step starts
        boundaries.add(int(valid[condensed + 1]))
    return sorted(boundaries)


def _resolve_labels(bands_node, band_labels: Optional[Sequence[str]]):
    """Return ``(positions, labels)`` or ``(None, None)`` when no usable labels."""
    if band_labels:
        labels = [str(lb) for lb in band_labels if str(lb) not in ("", "?")]
        if len(labels) >= 2:
            n_kpts = len(np.asarray(bands_node.get_kpoints()))
            return _segment_boundaries(n_kpts, len(labels)), labels

    raw = bands_node.base.attributes.get("labels")
    numbers = bands_node.base.attributes.get("label_numbers")
    if isinstance(raw, list) and len(raw) >= 2:
        # aiida-phonopy stores plain label strings *plus* the q-point index
        # of every label in ``label_numbers`` (or, in some paths, explicit
        # [qpoint_index, label] pairs).
        if all(isinstance(item, str) for item in raw):
            if isinstance(numbers, (list, tuple)) and len(numbers) == len(raw):
                # Use the recorded positions: the geometric/heuristic guesses
                # put the ticks at the wrong q-points for paths whose
                # high-symmetry points do not fall on a direction change
                # (e.g. the Cmcm 15-label path: one label per arc corner plus
                # the "?" placeholder of the connector start).
                explicit_pos: List[int] = []
                explicit_labels: List[str] = []
                valid = True
                for label, number in zip(raw, numbers):
                    symbol = str(label)
                    if symbol in ("", "?"):
                        continue
                    try:
                        explicit_pos.append(int(number))
                    except (TypeError, ValueError):
                        valid = False
                        break
                    explicit_labels.append(symbol)
                if valid and len(explicit_labels) >= 2:
                    return explicit_pos, explicit_labels
            labels = [lb for lb in raw if lb not in ("", "?")]
            if len(labels) >= 2:
                kpts = np.asarray(bands_node.get_kpoints())
                positions = _geometric_boundaries(kpts)
                if len(positions) != len(labels):
                    # Geometric detection failed (e.g. straight-through
                    # labels) — fall back to the evenly-split guess.
                    positions = _segment_boundaries(len(kpts), len(labels))
                return positions, labels
        positions: List[int] = []
        labels: List[str] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            symbol = str(item[1])
            if symbol in ("", "?"):
                continue
            try:
                positions.append(int(item[0]))
            except (TypeError, ValueError):
                continue
            labels.append(symbol)
        if len(labels) >= 2 and len(positions) == len(labels):
            return positions, labels
    return None, None


def extract_phonon_data(node) -> Dict[str, Any]:
    """Extract the phonon band (+DOS) data of one node.

    Returns ``{"freqs", "kpoints", "label_pos", "label_text", "dos_x",
    "dos_y"}`` — the raw arrays a renderer needs (freqs shape
    ``(nk, nbands)``, DOS in THz). Raises ``ValueError`` when the node
    has no ``phonon_bands`` output.
    """
    node = _load_node(node)
    outputs = _get_outputs(node)
    bands = outputs.get("phonon_bands")
    if bands is None:
        raise ValueError(
            f"node {node.pk} has no 'phonon_bands' output; "
            f"available: {sorted(outputs)}"
        )

    freqs = np.asarray(bands.get_bands())
    kpts = np.asarray(bands.get_kpoints())
    label_pos, label_text = _resolve_labels(bands, None)

    dos = outputs.get("total_phonon_dos")
    dos_x = dos_y = None
    if dos is not None:
        dos_x = np.asarray(dos.get_x()[1])
        dos_y = np.asarray(dos.get_y()[0][1])

    return {
        "freqs": freqs,
        "kpoints": kpts,
        "label_pos": label_pos,
        "label_text": label_text,
        "dos_x": dos_x,
        "dos_y": dos_y,
    }


def render_phonon_spec(spec, out_dir: Union[str, Path]) -> List[Path]:
    """Render a JSON-driven phonon figure (band + DOS) for several nodes.

    ``spec`` is a :class:`aiida_uranium_workflow.cli.plot._loading.PlotSpec`
    with ``mode == "phonon"``; ``spec.data`` maps backend → pks + labels
    (e.g. abacus pw/lcao, fleur lapw). Every series is drawn on the same
    band axis (each in its own colour, legend ``backend/label``) and its
    DOS on the shared right axis.

    ``spec.is_combined``: ``True`` → one figure per spec (``fig_name``);
    ``False`` → one figure per series (``<label>_<fig_name>``).

    Returns the written PNG paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys

    from aiida_uranium_workflow.cli.plot._loading import PlotSpec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat: List[tuple] = []
    for backend, series in spec.data.items():
        for idx, pk in enumerate(series.pks):
            label = series.labels[idx] if idx < len(series.labels) else pk
            try:
                data = extract_phonon_data(pk)
            except Exception as exc:
                print(
                    f"[plot-phonon] WARNING: skipping {backend}/{pk} ({exc})",
                    file=sys.stderr,
                )
                continue
            flat.append((data, backend, label))
    if not flat:
        raise RuntimeError(
            f"No phonon data could be extracted for spec {spec.figure.fig_name!r}."
        )

    out_paths: List[Path] = []

    def _render(flat_series, output_path) -> None:
        fig, (ax_band, ax_dos) = plt.subplots(
            1, 2, figsize=(10, 6), sharey=True,
            gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.05},
        )
        _draw_phonon_series(ax_band, ax_dos, flat_series, spec)
        if spec.figure.title:
            fig.suptitle(spec.figure.title)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if spec.is_combined:
        output_path = out_dir / spec.figure.fig_name
        _render(flat, output_path)
        out_paths.append(output_path)
    else:
        for data, backend, label in flat:
            safe_label = str(label).replace("/", "_")
            output_path = out_dir / f"{safe_label}_{spec.figure.fig_name}"
            _render([(data, backend, label)], output_path)
            out_paths.append(output_path)

    return out_paths


def _draw_phonon_series(ax_band, ax_dos, flat, spec) -> None:
    """Draw all series onto the shared band / DOS axes."""
    palette = [
        "tab:blue", "tab:red", "tab:green", "tab:orange",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan",
    ]
    max_x = 0.0
    for series_index, (data, backend, label) in enumerate(flat):
        color = palette[series_index % len(palette)]
        freqs = np.asarray(data["freqs"])
        kpts = np.asarray(data["kpoints"])
        # Physical x-axis: cumulative distance along the q-path (not the
        # per-series point index, which would stretch runs with different
        # path-point densities relative to each other). Disconnected
        # high-symmetry arcs (the seekpath/phonopy "connector" jump, e.g.
        # H→P of the bcc path) are split and drawn as separate polylines
        # with a gap, so no fake straight line crosses the BZ.
        breaks = _path_breaks(kpts)
        x, chunks = _chunked_x(kpts, breaks)
        max_x = max(max_x, float(x[-1]) if len(x) else 0.0)

        for band in freqs.T:
            for start, end in chunks:
                ax_band.plot(
                    x[start:end + 1], band[start:end + 1],
                    color=color, lw=1.0, alpha=0.9,
                )

        # High-symmetry labels of the first series only (they are the
        # same k-path for all nodes of the same structure). Adjacent
        # labels without interior points (the H→P connector) are merged
        # into a single "H|P" tick.
        if (
            data.get("label_pos") is not None
            and not getattr(ax_band, "_ph_labeled", False)
        ):
            tick_x, tick_text = _label_ticks(x, data["label_pos"], data["label_text"])
            for xp in tick_x[1:-1]:
                ax_band.axvline(x=xp, color="grey", lw=0.6, ls="--", alpha=0.6)
            ax_band.set_xticks(tick_x)
            ax_band.set_xticklabels(tick_text, fontsize=9)
            setattr(ax_band, "_ph_labeled", True)

        ax_band.plot([], [], color=color, lw=2.0, label=f"{backend}/{label}")

        dos_x = data.get("dos_x")
        dos_y = data.get("dos_y")
        if dos_x is not None and dos_y is not None:
            ax_dos.plot(dos_y, dos_x, color=color, lw=1.0)
            ax_dos.fill_betweenx(dos_x, dos_y, color=color, alpha=0.2)

    ax_band.set_xlabel(spec.figure.xlabel or "q-path distance (2π/a)")
    ax_band.set_ylabel(spec.figure.ylabel or "Frequency (THz)")
    if spec.figure.xlim:
        ax_band.set_xlim(spec.figure.xlim[0], spec.figure.xlim[1])
    else:
        if max_x <= 0.0:  # degenerate k-points (tests) — fall back to index
            max_x = max(
                (len(np.asarray(d["kpoints"])) for d, _, _ in flat), default=1
            ) - 1
        ax_band.set_xlim(0, max_x)
    # ``ylim`` is the frequency window (THz) for phonon mode.
    if spec.figure.ylim:
        ax_band.set_ylim(spec.figure.ylim[0], spec.figure.ylim[1])
    ax_dos.set_xlabel("DOS (1/THz)")
    ax_dos.set_ylim(ax_band.get_ylim())
    ax_band.legend(loc=spec.figure.legend_loc, fontsize=8)


def render_phonon_figure(
    node: Union[int, str, Any],
    output_path: Union[str, Path],
    *,
    band_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    freq_range: Optional[Sequence[float]] = None,
) -> Path:
    """Render the phonon band structure (+ DOS) figure and save it to ``output_path``.

    :param node: a node object, pk or UUID exposing ``phonon_bands``
        (and optionally ``total_phonon_dos``) outputs.
    :param output_path: destination PNG path.
    :param band_labels: optional explicit high-symmetry labels for the
        x-axis (positions are the band-path segment boundaries).
    :param title: optional figure title (defaults to a node-derived one).
    :param freq_range: optional ``(fmin, fmax)`` frequency window (THz).
    :returns: the written ``output_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node = _load_node(node)
    outputs = _get_outputs(node)
    bands = outputs.get("phonon_bands")
    if bands is None:
        raise ValueError(
            f"node {node.pk} has no 'phonon_bands' output; "
            f"available: {sorted(outputs)}"
        )

    freqs = np.asarray(bands.get_bands())
    kpts = np.asarray(bands.get_kpoints())
    x = _path_length(kpts)
    label_pos, label_text = _resolve_labels(bands, band_labels)

    dos = outputs.get("total_phonon_dos")
    dos_x = dos_y = None
    if dos is not None:
        dos_x = np.asarray(dos.get_x()[1])
        dos_y = np.asarray(dos.get_y()[0][1])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.05},
    )

    # ---- band structure ------------------------------------------------
    # Split disconnected high-symmetry arcs (seekpath/phonopy connector
    # jumps) so no fake straight line crosses the BZ (e.g. H→P of bcc).
    breaks = _path_breaks(kpts)
    x, chunks = _chunked_x(kpts, breaks)
    for band in freqs.T:
        for start, end in chunks:
            ax1.plot(x[start:end + 1], band[start:end + 1], color="tab:blue", lw=1.2)
    if label_pos is not None:
        # Merge connector labels (H|P) and map them onto the chunked
        # q-distance x-axis.
        tick_x, tick_text = _label_ticks(x, label_pos, label_text)
        for xp in tick_x[1:-1]:
            ax1.axvline(x=xp, color="grey", lw=0.8, ls="--")
        ax1.set_xticks(tick_x)
        ax1.set_xticklabels(tick_text)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylabel("Frequency (THz)")
    ax1.set_title(title or f"Phonon bands (pk={node.pk})")

    # ---- DOS -----------------------------------------------------------
    if dos_y is not None:
        ax2.plot(dos_y, dos_x, color="tab:red", lw=1.2)
        ax2.fill_betweenx(dos_x, dos_y, color="tab:red", alpha=0.3)
        ax2.set_xlabel("DOS (1/THz)")
        ax2.set_title("Total DOS")
    ax2.set_ylim(ax1.get_ylim())

    if freq_range is not None:
        ax1.set_ylim(float(freq_range[0]), float(freq_range[1]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
