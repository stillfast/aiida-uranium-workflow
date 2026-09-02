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


def _segment_boundaries(n_kpoints: int, n_labels: int) -> List[int]:
    """Indices of the segment boundaries for ``n_labels`` evenly split labels."""
    if n_labels <= 1:
        return [0]
    pts_per_seg = max((n_kpoints - 1) // (n_labels - 1), 1)
    return [min(i * pts_per_seg, n_kpoints - 1) for i in range(n_labels)]


def _resolve_labels(bands_node, band_labels: Optional[Sequence[str]]):
    """Return ``(positions, labels)`` or ``(None, None)`` when no usable labels."""
    if band_labels:
        labels = [str(lb) for lb in band_labels if str(lb) not in ("", "?")]
        if len(labels) >= 2:
            n_kpts = len(np.asarray(bands_node.get_kpoints()))
            return _segment_boundaries(n_kpts, len(labels)), labels

    raw = bands_node.base.attributes.get("labels")
    if isinstance(raw, list) and len(raw) >= 2:
        # aiida-phonopy stores either plain label strings (positions =
        # evenly split segment boundaries) or [qpoint_index, label] pairs.
        if all(isinstance(item, str) for item in raw):
            labels = [lb for lb in raw if lb not in ("", "?")]
            if len(labels) >= 2:
                n_kpts = len(np.asarray(bands_node.get_kpoints()))
                return _segment_boundaries(n_kpts, len(labels)), labels
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
    for series_index, (data, backend, label) in enumerate(flat):
        color = palette[series_index % len(palette)]
        freqs = np.asarray(data["freqs"])
        kpts = np.asarray(data["kpoints"])
        x = np.arange(len(kpts))

        for band in freqs.T:
            ax_band.plot(x, band, color=color, lw=1.0, alpha=0.9)

        # High-symmetry labels of the first series only (they are the
        # same k-path for all nodes of the same structure).
        if (
            data.get("label_pos") is not None
            and not getattr(ax_band, "_ph_labeled", False)
        ):
            xpos = [x[min(int(i), len(x) - 1)] for i in data["label_pos"]]
            for xp in xpos[1:-1]:
                ax_band.axvline(x=xp, color="grey", lw=0.6, ls="--", alpha=0.6)
            ax_band.set_xticks(xpos)
            ax_band.set_xticklabels(data["label_text"], fontsize=9)
            setattr(ax_band, "_ph_labeled", True)

        ax_band.plot([], [], color=color, lw=2.0, label=f"{backend}/{label}")

        dos_x = data.get("dos_x")
        dos_y = data.get("dos_y")
        if dos_x is not None and dos_y is not None:
            ax_dos.plot(dos_y, dos_x, color=color, lw=1.0)
            ax_dos.fill_betweenx(dos_x, dos_y, color=color, alpha=0.2)

    ax_band.set_xlabel(spec.figure.xlabel or "k-point index")
    ax_band.set_ylabel(spec.figure.ylabel or "Frequency (THz)")
    if spec.figure.xlim:
        ax_band.set_xlim(spec.figure.xlim[0], spec.figure.xlim[1])
    else:
        nk_max = max((len(np.asarray(d["kpoints"])) for d, _, _ in flat), default=1)
        ax_band.set_xlim(0, max(nk_max - 1, 1))
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
    for band in freqs.T:
        ax1.plot(x, band, color="tab:blue", lw=1.2)
    if label_pos is not None:
        # ``label_pos`` are k-point *indices*; map them to path-length
        # x-coordinates (the x-axis is the cumulative reciprocal distance).
        xpos = [x[min(int(i), len(x) - 1)] for i in label_pos]
        for xp in xpos[1:-1]:
            ax1.axvline(x=xp, color="grey", lw=0.8, ls="--")
        ax1.set_xticks(xpos)
        ax1.set_xticklabels(label_text)
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
