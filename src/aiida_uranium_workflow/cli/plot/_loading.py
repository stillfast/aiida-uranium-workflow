"""Load JSON-driven band / DOS / PDOS plot specifications.

The expected JSON shape (one of ``band.json`` / ``dos.json`` /
``pdos.json``)::

    {
      "mode": "band" | "dos" | "pdos" | "band_compare",
      "is_combined": true,
      "data": {
        "<backend>": {"pks": [uuid|int, ...], "labels": [str, ...],
                      "orbitals": ["s", "p", "d", "f"]},   # pdos mode only
        ...
      },
      "figure": {
        "title": "...",
        "xlabel": "...", "ylabel": "...",
        "energy_range": [emin, emax],        # band mode (y-axis)
        "xlim": [0, nk-1],                   # band mode (k-point index
                                             # range; optional, defaults
                                             # to the data range) and
                                             # dos / pdos modes
        "ylim": [...],                       # band (y) / dos / pdos
        "zero_to_efermi": true,
        "legend_loc": "best",
        "fig_name": "band.png",
        "sigma": 0.1,                        # band_compare only (eV)
        "e_min": -5.0,                       # band_compare only (eV,
        "e_max": 5.0                         # relative to each E_F)
      }
    }

This module is purely the *spec parser* + helper to look up preset
names from the project's ``output.json`` when the spec only has pks.
The actual plotting lives in :mod:`.cli.plot._rendering`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Normalized spec data classes
# ---------------------------------------------------------------------------


@dataclass
class BackendSeries:
    """One backend's contribution to a single figure."""

    backend: str
    pks: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    #: pdos mode: which orbital channels to draw (e.g. ["s", "p", "d", "f"]).
    #: Empty means "every channel available in the data".
    orbitals: List[str] = field(default_factory=list)


@dataclass
class FigureSpec:
    """Plotting parameters (axis labels, ranges, output file)."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    # ``ylim`` / ``energy_range`` control the band y-axis; ``xlim``
    # controls the band x-axis (k-point index range; defaults to the
    # k-points range of the plotted data) and the dos/pdos x-axis.
    energy_range: Optional[List[float]] = None
    xlim: Optional[List[float]] = None
    ylim: Optional[List[float]] = None
    zero_to_efermi: bool = True
    legend_loc: str = "best"
    fig_name: str = "band.png"
    # band_compare mode: Fermi–Dirac smearing width (eV) and optional
    # energy window (eV, relative to each E_F) restricting which bands
    # enter the η_v sum.
    sigma: float = 0.1
    e_min: Optional[float] = None
    e_max: Optional[float] = None
    # band_compare mode: how states are paired ("window" — per-k
    # energy-sorted pairing inside the energy window; "index" — by band
    # index; default: window when e_min/e_max given, else index).
    align: str = "auto"
    # band_compare mode: restrict max η to states occupied in both
    # structures (Fermi–Dirac weight > 0.5), SSSP max_diff semantics.
    occupied_only: bool = False


@dataclass
class PlotSpec:
    """One full plot specification file."""

    mode: str                              # "band" | "dos" | "pdos" | "band_compare"
    is_combined: bool = True
    data: Dict[str, BackendSeries] = field(default_factory=dict)
    figure: FigureSpec = field(default_factory=FigureSpec)

    @property
    def backends(self) -> List[str]:
        return list(self.data.keys())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Plot spec JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_pk(pk: Any) -> str:
    """Accept and stringify int / str / uuid."""
    return str(pk)


def load_spec(path: Path) -> PlotSpec:
    """Parse a band.json / dos.json / pdos.json / phonon.json into a
    :class:`PlotSpec`."""
    raw = _read_json(path)

    mode = str(raw.get("mode", "")).strip().lower()
    if mode not in ("band", "dos", "pdos", "band_compare", "phonon"):
        raise ValueError(
            f"Invalid mode {mode!r} in {path}; expected "
            f"'band' / 'dos' / 'pdos' / 'band_compare' / 'phonon'."
        )

    is_combined = bool(raw.get("is_combined", True))

    data: Dict[str, BackendSeries] = {}
    for backend, body in (raw.get("data") or {}).items():
        if not isinstance(body, dict):
            continue
        series = BackendSeries(
            backend=str(backend),
            pks=[_coerce_pk(p) for p in (body.get("pks") or [])],
            labels=[str(l) for l in (body.get("labels") or [])],
            orbitals=[str(o) for o in (body.get("orbitals") or [])],
        )
        if series.pks:
            data[str(backend)] = series

    fig_raw = raw.get("figure") or {}
    figure = FigureSpec(
        title=str(fig_raw.get("title", "")),
        xlabel=str(fig_raw.get("xlabel", "")),
        ylabel=str(fig_raw.get("ylabel", "")),
        energy_range=list(fig_raw["energy_range"]) if "energy_range" in fig_raw else None,
        xlim=list(fig_raw["xlim"]) if "xlim" in fig_raw else None,
        ylim=list(fig_raw["ylim"]) if "ylim" in fig_raw else None,
        zero_to_efermi=bool(fig_raw.get("zero_to_efermi", True)),
        legend_loc=str(fig_raw.get("legend_loc", "best")),
        fig_name=str(fig_raw.get("fig_name", "band.png")),
        sigma=float(fig_raw.get("sigma", 0.1)),
        e_min=(
            float(fig_raw["e_min"])
            if "e_min" in fig_raw and fig_raw["e_min"] is not None else None
        ),
        e_max=(
            float(fig_raw["e_max"])
            if "e_max" in fig_raw and fig_raw["e_max"] is not None else None
        ),
        align=str(fig_raw.get("align", "auto")).strip().lower(),
        occupied_only=bool(fig_raw.get("occupied_only", False)),
    )

    return PlotSpec(
        mode=mode,
        is_combined=is_combined,
        data=data,
        figure=figure,
    )


# ---------------------------------------------------------------------------
# Convenience: read output.json to resolve preset names → pks
# ---------------------------------------------------------------------------


def load_output_json(path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Read the nested ``output.json`` produced by ``aiida-uranium run``.

    Returns ``{backend: {key: {preset_name: pk|uuid}}}``. The ``key``
    here is the second-level dict name (``"banddos"`` for our banddos
    workflow).
    """
    return _read_json(path)


def resolve_pks_from_output(
    output: Dict[str, Any],
    backend: str,
    key: str,
    preset: str,
) -> Optional[str]:
    """Return the pk/uuid string for ``(backend, key, preset)`` or None."""
    try:
        return output[backend][key][preset]
    except (KeyError, TypeError):
        return None


__all__ = [
    "BackendSeries",
    "FigureSpec",
    "PlotSpec",
    "load_spec",
    "load_output_json",
    "resolve_pks_from_output",
]