"""Plot band / DOS data from any supported AiiDA WorkChain.

Two-stage API:

1. :func:`extract_band_dos(pk_or_uuid)` — pulls the data from the
   AiiDA WorkChain and normalizes it into :class:`BandData` /
   :class:`DosData` (backend-agnostic, no AiiDA types).
2. :func:`plot_band` / :func:`plot_dos` / :func:`plot_all` — render
   the standardized data to matplotlib PNGs.

Example
-------
::

    from aiida_uranium_workflow.utils.plot import (
        extract_band_dos, plot_all,
    )

    data = extract_band_dos(339125)               # ABACUS
    paths = plot_all(data["band"], data["dos"],
                     out_dir="/tmp/banddos",
                     prefix="abacus")

    # Same call works for FleurBandAndDosWorkChain (band + dos in one wc).
    data = extract_band_dos(<fleur_wc_pk>)
    plot_all(data["band"], data["dos"], out_dir="/tmp/banddos")

Adding a new backend means writing one extractor in :mod:`extract`
and registering it in :data:`EXTRACTORS` — the plotting code does
not need to change.
"""
from ._constants import HA_TO_EV
from .compare import compare_all, compare_bands, format_tables
from .extract import (
    BandData,
    DosData,
    EXTRACTORS,
    extract_abacus,
    extract_band_dos,
    extract_fleur,
)
from .plot import plot_all, plot_band, plot_dos

__all__ = [
    # Constants
    "HA_TO_EV",
    # Data containers
    "BandData",
    "DosData",
    # Extractors
    "EXTRACTORS",
    "extract_band_dos",
    "extract_abacus",
    "extract_fleur",
    # Comparison (PRB 98, 085117 + SSSP alignment modes)
    "compare_bands",
    "compare_all",
    "format_tables",
    # Plotters
    "plot_band",
    "plot_dos",
    "plot_all",
]