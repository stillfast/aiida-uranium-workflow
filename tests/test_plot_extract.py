"""Tests for the band / DOS extractors with a missing ``dos`` output.

Regression: ``extract_abacus`` / ``extract_fleur`` unconditionally read
``wc.outputs.dos``, which is a ``required=False`` port — band-only runs
(``run_dos=False``) carry no DOS output, and the whole node was skipped
by the plot CLI ("Node<...> does not have an output with link label
'dos'") even though the band data was perfectly plottable.
"""

from __future__ import annotations

import numpy as np

from aiida_uranium_workflow.utils.plot.extract import (
    extract_abacus,
    extract_fleur,
)


class _Outputs:
    """Mimic a WorkChain's ``outputs``: ``dos`` raises like AiiDA when absent."""

    def __init__(self, band, dos=None):
        self.band_structure = band
        self._dos = dos
        self.output_parameters = _Params()

    @property
    def dos(self):
        if self._dos is None:
            raise AttributeError(
                "does not have an output with link label 'dos'"
            )
        return self._dos


class _Params:
    def get_dict(self):
        return {"band": {}, "dos": {}}


class _Band:
    """A minimal BandsData stand-in (arrays + attributes)."""

    def __init__(self, nk=3, nbands=4):
        self.base = _Base()
        self._bands = np.zeros((1, nk, nbands))  # (1, nk, nbands)
        self._kpoints = np.zeros((nk, 3))

    def get_array(self, name):
        if name == "bands":
            return self._bands
        if name == "kpoints":
            return self._kpoints
        raise KeyError(name)


class _Base:
    def __init__(self):
        self.attributes = {
            "labels": [],
            "label_numbers": [],
            "fermi_level": 0.0,
            "cell": np.eye(3),
        }


class _Dos:
    """Minimal DOS stand-in (abacus style: energy / tdos arrays)."""

    def __init__(self):
        self.base = _Base()
        self._energy = np.linspace(-10, 10, 100)
        self._tdos = np.zeros(100)

    def get_array(self, name):
        if name == "energy":
            return self._energy
        if name == "tdos":
            return self._tdos
        raise KeyError(name)


class _FleurDos:
    """Minimal FLEUR XyData stand-in (x_array / y_array_0)."""

    def __init__(self):
        self.base = _Base()
        self.base.attributes["y_names"] = ["Total"]
        self._x = np.linspace(-10, 10, 100)
        self._y0 = np.zeros(100)

    def get_array(self, name):
        if name == "x_array":
            return self._x
        if name == "y_array_0":
            return self._y0
        raise KeyError(name)


def test_extract_abacus_band_only():
    """Band-only run: band is extracted, dos is None — no crash."""
    wc = type("Wc", (), {"pk": 1, "outputs": _Outputs(_Band())})()
    bundle = extract_abacus(wc)
    assert bundle["band"] is not None
    assert bundle["band"].energies.shape[0] == 4  # nbands
    assert bundle["dos"] is None


def test_extract_abacus_with_dos():
    """Full band+dos run: both are extracted."""
    wc = type("Wc", (), {"pk": 1, "outputs": _Outputs(_Band(), _Dos())})()
    bundle = extract_abacus(wc)
    assert bundle["band"] is not None
    assert bundle["dos"] is not None
    assert bundle["dos"].total.shape[0] == 100


def test_extract_fleur_band_only():
    """FLEUR band-only run: band extracted, dos None — no crash."""
    wc = type("Wc", (), {"pk": 1, "outputs": _Outputs(_Band())})()
    bundle = extract_fleur(wc)
    assert bundle["band"] is not None
    assert bundle["dos"] is None


def test_extract_fleur_with_dos():
    """FLEUR full run: both extracted."""
    wc = type("Wc", (), {"pk": 1, "outputs": _Outputs(_Band(), _FleurDos())})()
    bundle = extract_fleur(wc)
    assert bundle["band"] is not None
    assert bundle["dos"] is not None
    assert bundle["dos"].total.shape[0] == 100


def test_collect_series_band_mode_keeps_band_only_node(monkeypatch):
    """``collect_series`` in band mode must keep a band-only node (the
    regression: it was skipped because extract raised on the missing
    ``dos`` output)."""
    from aiida_uranium_workflow.cli.plot import _rendering
    from aiida_uranium_workflow.cli.plot._loading import BackendSeries, FigureSpec, PlotSpec
    from aiida_uranium_workflow.utils.plot.extract import BandData

    monkeypatch.setattr(_rendering, "_ensure_profile_loaded", lambda: None)

    band = BandData(
        energies=np.zeros((4, 3)),
        kpoints=np.zeros((3, 3)),
        labels=[],
        label_numbers=[],
        fermi_energy=0.0,
        workchain_pk=1,
    )
    monkeypatch.setattr(
        _rendering,
        "extract_band_dos",
        lambda pk: {"band": band, "dos": None},
    )

    spec = PlotSpec(
        mode="band",
        is_combined=True,
        data={"abacus": BackendSeries(backend="abacus", pks=["1"])},
        figure=FigureSpec(),
    )
    flat = _rendering.collect_series(spec)
    assert len(flat) == 1
    assert flat[0][0] is band
    assert flat[0][1] == "abacus"


def test_collect_series_dos_mode_skips_band_only_node(monkeypatch):
    """DOS mode still skips a node without DOS data."""
    from aiida_uranium_workflow.cli.plot import _rendering
    from aiida_uranium_workflow.cli.plot._loading import BackendSeries, FigureSpec, PlotSpec
    from aiida_uranium_workflow.utils.plot.extract import BandData

    monkeypatch.setattr(_rendering, "_ensure_profile_loaded", lambda: None)

    band = BandData(
        energies=np.zeros((4, 3)),
        kpoints=np.zeros((3, 3)),
        labels=[],
        label_numbers=[],
        fermi_energy=0.0,
        workchain_pk=1,
    )
    monkeypatch.setattr(
        _rendering,
        "extract_band_dos",
        lambda pk: {"band": band, "dos": None},
    )

    spec = PlotSpec(
        mode="dos",
        is_combined=True,
        data={"abacus": BackendSeries(backend="abacus", pks=["1"])},
        figure=FigureSpec(),
    )
    flat = _rendering.collect_series(spec)
    assert flat == []
