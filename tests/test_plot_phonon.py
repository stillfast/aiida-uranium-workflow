"""Tests for the JSON-driven phonon plotting (band-style spec)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aiida_uranium_workflow.cli.plot._loading import (
    BackendSeries,
    FigureSpec,
    PlotSpec,
    load_spec,
)
from aiida_uranium_workflow.utils.plot import phonon


def _fake_phonon_node(nk=20, nbands=5):
    """A stand-in for a node exposing ``phonon_bands`` + DOS."""
    freqs = np.tile(np.linspace(-1.0, 3.0, nk), (nbands, 1)).T  # (nk, nbands)
    kpts = np.zeros((nk, 3))

    class _Bands:
        def get_bands(self):
            return freqs

        def get_kpoints(self):
            return kpts

        base = SimpleNamespace(attributes={})

    class _Dos:
        def get_x(self):
            return ("x", np.linspace(-2.0, 4.0, 100))

        def get_y(self):
            # XyData.get_y() -> [(y_name, y_array, y_units), ...]
            return [("dos", np.linspace(0.0, 1.0, 100), "1/THz")]

    return SimpleNamespace(
        pk=1,
        base=SimpleNamespace(attributes={}),
        outputs={"phonon_bands": _Bands(), "total_phonon_dos": _Dos()},
    )


def _patch_node_loading(monkeypatch):
    node = _fake_phonon_node()
    monkeypatch.setattr(phonon, "_load_node", lambda _n: node)
    monkeypatch.setattr(
        phonon, "_get_outputs", lambda _node: node.outputs
    )
    return node


def _spec(**kwargs):
    fig = kwargs.pop("figure", {})
    return PlotSpec(
        mode="phonon",
        is_combined=kwargs.pop("is_combined", True),
        data={
            "abacus": BackendSeries(backend="abacus", pks=["1"], labels=["pw"]),
            "fleur": BackendSeries(backend="fleur", pks=["2"], labels=["lapw"]),
        },
        figure=FigureSpec(fig_name="phonon.png", **fig),
    )


class TestLoadSpec:
    def test_phonon_mode_accepted(self, tmp_path):
        spec_file = tmp_path / "phonon.json"
        spec_file.write_text(
            '{"mode": "phonon", "is_combined": true, '
            '"data": {"abacus": {"pks": ["abc"], "labels": ["pw"]}}, '
            '"figure": {"ylim": [-2, 2], "fig_name": "phonon.png"}}'
        )
        spec = load_spec(spec_file)
        assert spec.mode == "phonon"
        assert spec.figure.ylim == [-2, 2]
        assert spec.figure.fig_name == "phonon.png"

    def test_invalid_mode_rejected(self, tmp_path):
        spec_file = tmp_path / "bad.json"
        spec_file.write_text('{"mode": "quantum", "data": {}}')
        with pytest.raises(ValueError, match="Invalid mode"):
            load_spec(spec_file)


class TestExtractPhononData:
    def test_extracts_bands_and_dos(self, monkeypatch):
        _patch_node_loading(monkeypatch)
        data = phonon.extract_phonon_data("1")
        assert data["freqs"].shape == (20, 5)
        assert data["kpoints"].shape == (20, 3)
        assert data["dos_x"] is not None
        assert data["dos_y"] is not None


class TestRenderPhononSpec:
    def test_combined_figure(self, monkeypatch, tmp_path):
        _patch_node_loading(monkeypatch)
        paths = phonon.render_phonon_spec(_spec(), tmp_path)
        assert len(paths) == 1
        assert (tmp_path / "phonon.png").exists()
        assert paths[0].stat().st_size > 0

    def test_per_series_figures(self, monkeypatch, tmp_path):
        _patch_node_loading(monkeypatch)
        paths = phonon.render_phonon_spec(
            _spec(is_combined=False), tmp_path
        )
        assert len(paths) == 2
        assert (tmp_path / "pw_phonon.png").exists()
        assert (tmp_path / "lapw_phonon.png").exists()

    def test_no_data_raises(self, monkeypatch, tmp_path):
        def _fail_load(_n):
            raise ValueError("no phonon_bands output")

        monkeypatch.setattr(phonon, "_load_node", _fail_load)
        with pytest.raises(RuntimeError, match="No phonon data"):
            phonon.render_phonon_spec(_spec(), tmp_path)

    def test_skips_bad_nodes(self, monkeypatch, tmp_path, capsys):
        node = _fake_phonon_node()
        monkeypatch.setattr(phonon, "_load_node", lambda _n: node)
        outputs = dict(node.outputs)
        # Only the first pk extracts fine; the second raises.
        calls = {"n": 0}

        def _fake_outputs(_node):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ValueError("no phonon_bands output")
            return outputs

        monkeypatch.setattr(phonon, "_get_outputs", _fake_outputs)
        paths = phonon.render_phonon_spec(_spec(), tmp_path)
        assert len(paths) == 1  # only the good series made it
        assert "skipping" in capsys.readouterr().err
