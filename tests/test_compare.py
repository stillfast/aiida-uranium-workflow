"""Unit tests for the band-comparison metrics (PRB 98, 085117).

Covers :mod:`aiida_uranium_workflow.utils.plot.compare`:

* Fermi–Dirac weighting (:func:`_fermi_dirac`)
* common k-point subsetting (:func:`_common_subset`)
* pairwise η_v / max η / ω (:func:`compare_bands`)
* all-pairs matrix (:func:`compare_all`)
* Markdown table rendering (:func:`format_tables`)
"""

import numpy as np
import pytest

from aiida_uranium_workflow.utils.plot.compare import (
    _common_subset,
    _fermi_dirac,
    compare_all,
    compare_bands,
    format_tables,
)

NK, NB = 5, 4
KPOINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.6, 0.0, 0.0],
    [0.8, 0.0, 0.0],
])


def _base_bands() -> np.ndarray:
    """A small fake band structure: 4 bands, 5 k-points, in eV."""
    return np.array([
        [-5.0 + j * 0.1 for j in range(NK)],
        [-4.0 + j * 0.1 for j in range(NK)],
        [-2.0 + j * 0.1 for j in range(NK)],
        [-1.0 + j * 0.1 for j in range(NK)],
    ])


class TestFermiDirac:
    def test_occupancy_limits(self):
        f = _fermi_dirac(np.array([-1.0, 0.0, 1.0]), e_fermi=0.0, sigma=0.1)
        assert f[0] == pytest.approx(1.0, abs=1e-4)   # exp(-10) → 0.99995
        assert f[1] == pytest.approx(0.5)
        assert f[2] == pytest.approx(0.0, abs=1e-4)

    def test_sigma_broadens(self):
        f_wide = _fermi_dirac(np.array([0.3]), e_fermi=0.0, sigma=0.5)
        f_narrow = _fermi_dirac(np.array([0.3]), e_fermi=0.0, sigma=0.1)
        assert f_wide[0] > f_narrow[0]


class TestCommonSubset:
    def test_matches_by_coordinate(self):
        shuffled = KPOINTS[::-1]
        e_a = _base_bands()
        e_b = _base_bands()[:, ::-1]
        ea, eb, ka, kb = _common_subset(e_a, e_b, KPOINTS, shuffled)
        # Both restricted to the common (all) k-points, ordered by A.
        assert ea.shape == e_b.shape == (NB, NK)
        assert np.allclose(ka, KPOINTS)
        assert np.allclose(kb, KPOINTS)

    def test_no_common_raises(self):
        other = KPOINTS + 10.0
        with pytest.raises(ValueError):
            _common_subset(_base_bands(), _base_bands(), KPOINTS, other)


class TestCompareBands:
    def test_identical(self):
        e = _base_bands()
        res = compare_bands(e, e, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0)
        assert res["eta_v"] == pytest.approx(0.0, abs=1e-10)
        assert res["max_eta"] == pytest.approx(0.0, abs=1e-10)
        assert res["omega"] == pytest.approx(0.0, abs=1e-10)

    def test_rigid_shift_absorbed_by_omega(self):
        # A rigid +0.05 eV shift of every band must be fully absorbed
        # by the optimal omega → η_v = 0 and max η = 0, ω = +0.05.
        e = _base_bands()
        res = compare_bands(e, e + 0.05, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0)
        assert res["eta_v"] == pytest.approx(0.0, abs=1e-8)
        assert res["max_eta"] == pytest.approx(0.0, abs=1e-8)
        assert res["omega"] == pytest.approx(0.05, abs=1e-6)

    def test_single_band_shift(self):
        # One band shifted by +0.1 eV: η_v is a weighted RMS of the
        # residual after the *global* omega (so strictly < 0.1 and > 0),
        # and max η = max|Δ + ω| > 0.
        e = _base_bands()
        e_b = e.copy()
        e_b[1] += 0.1
        res = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0)
        assert 0.0 < res["eta_v"] < 0.1
        assert 0.0 < res["max_eta"] <= 0.1
        # The weighted sum is minimised → the first-order condition
        # Σ f̃ (Δ + ω) = 0 holds.
        delta = e[:NB] - e_b[:NB]
        f = np.sqrt(_fermi_dirac(e, 0.0, 0.1) * _fermi_dirac(e_b, 0.0, 0.1))
        assert np.sum(f * (delta + res["omega"])) == pytest.approx(0.0, abs=1e-9)

    def test_energy_window_excludes_bands(self):
        e = _base_bands()
        e_b = e.copy()
        e_b[0] += 0.5  # deepest band only
        full = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                             fermi_a=0.0, fermi_b=0.0)
        # A window above the deepest band removes the shifted band:
        # only bands in [-3, 3] eV (relative to E_F) enter.
        win = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0,
                            e_min=-3.0, e_max=3.0)
        assert win["eta_v"] < full["eta_v"]
        # The window gates max η too: the shifted deep band is outside
        # the window, so max η no longer sees it.
        assert win["max_eta"] < full["max_eta"]

    def test_window_restricts_max_eta(self):
        # A deep band shifted by 100 eV would dominate the unweighted
        # max η; with a window above it, max η reflects only the
        # windowed bands.
        e = _base_bands()
        e_b = e.copy()
        e_b[0] += 100.0
        e_b[2] += 0.05
        win = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0,
                            e_min=-3.0, e_max=3.0)
        # Band 2 (≈-2 eV, inside the window) shifted by 0.05 dominates;
        # the global ω partially absorbs the shift, so the residual
        # max η lies between 0 and 0.05.
        assert 0.0 < win["max_eta"] <= 0.05
        assert win["max_eta"] < 1.0

    def test_shape_mismatch_raises(self):
        # Different k-point counts with no k-point arrays → the arrays
        # cannot be truncated to a common shape.
        with pytest.raises(ValueError):
            compare_bands(np.zeros((4, 5)), np.zeros((4, 4)))

    def test_different_energy_frames_aligned(self):
        # The same band structure stored in two different reference
        # frames — absolute (E_F = 27.7 eV) vs already shifted to
        # E_F = 0 — must compare as identical after each is shifted by
        # its own E_F: η_v = max η = 0 and ω = 0.
        e_abs = _base_bands() + 27.7
        e_rel = _base_bands()
        res = compare_bands(e_abs, e_rel, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=27.7, fermi_b=0.0)
        assert res["eta_v"] == pytest.approx(0.0, abs=1e-8)
        assert res["max_eta"] == pytest.approx(0.0, abs=1e-8)
        assert res["omega"] == pytest.approx(0.0, abs=1e-6)

    def test_band_index_misalignment_window(self):
        # Two codes storing a different number of deep core states:
        # A has 2 extra deep bands (shifting every band index by 2),
        # B has 1. Same-band-index pairing would compare unrelated
        # states; with a window, the per-k energy-sorted pairing aligns
        # the physically corresponding shallow bands, so η_v and ω stay
        # small.
        e_a = _base_bands()
        e_b = _base_bands() + 0.05          # same shallow bands, +0.05 eV
        e_a = np.vstack([np.full((2, NK), -200.0), e_a])  # 2 deep states
        e_b = np.vstack([np.full((1, NK), -200.0), e_b])  # 1 deep state
        res = compare_bands(e_a, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0,
                            e_min=-10.0, e_max=10.0)
        # 0.05 eV rigid shift fully absorbed by ω → η_v ≈ 0, ω = +0.05
        # (B's shallow bands are 0.05 eV above A's; ω is added to Δ).
        assert res["eta_v"] == pytest.approx(0.0, abs=1e-6)
        assert res["max_eta"] == pytest.approx(0.0, abs=1e-6)
        assert res["omega"] == pytest.approx(0.05, abs=1e-6)


class TestCompareAll:
    def test_symmetric_and_diagonal(self):
        e = _base_bands()
        tbl = compare_all([
            ("A", e, KPOINTS, 0.0),
            ("B", e + 0.05, KPOINTS, 0.0),
        ])
        eta = np.asarray(tbl["eta_v"])
        # Off-diagonal is symmetric; diagonal is NaN (self-compare).
        assert eta[0, 1] == eta[1, 0]
        assert np.isnan(np.diag(eta)).all()
        # Rigid shift pair → η_v = 0.
        assert eta[0, 1] == pytest.approx(0.0, abs=1e-8)

    def test_incompatible_pair_is_nan(self):
        e = _base_bands()
        other_k = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tbl = compare_all([
            ("A", e, KPOINTS, 0.0),
            ("X", e, other_k, 0.0),
        ])
        assert np.isnan(tbl["eta_v"][0][1])

    def test_three_way_rigid_shifts(self):
        e = _base_bands()
        tbl = compare_all([
            ("A", e, KPOINTS, 0.0),
            ("B", e + 0.1, KPOINTS, 0.0),
            ("C", e - 0.2, KPOINTS, 0.0),
        ])
        eta = np.asarray(tbl["eta_v"])
        mx = np.asarray(tbl["max_eta"])
        # Every pair differs only by a rigid shift → η_v = 0 for all.
        assert np.allclose(eta[np.isfinite(eta)], 0.0, atol=1e-8)
        assert np.allclose(mx[np.isfinite(mx)], 0.0, atol=1e-8)
        om = np.asarray(tbl["omega"])
        # ω = -Σf̃(εA-εB)/Σf̃: for A vs B with B = A + 0.1 the rigid
        # shift that aligns B to A is +0.1.
        assert om[0, 1] == pytest.approx(0.1, abs=1e-6)
        assert om[0, 2] == pytest.approx(-0.2, abs=1e-6)


class TestWeights:
    def test_uniform_weights_match_no_weights(self):
        e = _base_bands()
        e_b = e.copy()
        e_b[1] += 0.1
        base = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                             fermi_a=0.0, fermi_b=0.0)
        w = np.ones(NK)
        weighted = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                                 weights_a=w, weights_b=w,
                                 fermi_a=0.0, fermi_b=0.0)
        assert weighted["eta_v"] == pytest.approx(base["eta_v"], abs=1e-10)
        assert weighted["omega"] == pytest.approx(base["omega"], abs=1e-10)

    def test_nonuniform_weights_change_result(self):
        # Concentrating weight on the k-point where band 1 differs most
        # must change η_v vs uniform weights.
        e = _base_bands()
        e_b = e.copy()
        e_b[1] += 0.1 * np.linspace(0.0, 1.0, NK)  # grows along k
        uniform = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                                fermi_a=0.0, fermi_b=0.0)
        w = np.zeros(NK)
        w[-1] = 1.0   # only the last k-point (largest shift) counts
        peaked = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                               weights_a=w, weights_b=w,
                               fermi_a=0.0, fermi_b=0.0)
        assert peaked["eta_v"] > uniform["eta_v"]

    def test_weights_reordered_with_kpoints(self):
        # Weights must follow their k-point through the common-subset
        # matching (shuffled B k-points carry shuffled weights).
        e = _base_bands()
        e_b = e[:, ::-1]
        shuffled_k = KPOINTS[::-1]
        w = np.arange(1.0, NK + 1.0)
        res = compare_bands(e, e_b,
                            kpoints_a=KPOINTS, kpoints_b=shuffled_k,
                            weights_a=np.ones(NK), weights_b=w[::-1],
                            fermi_a=0.0, fermi_b=0.0)
        assert res["eta_v"] == pytest.approx(0.0, abs=1e-10)


class TestOccupiedOnly:
    def test_excludes_unoccupied_states_from_max(self):
        # A deep occupied band shifted by 100 eV dominates the
        # unweighted max η; with occupied_only + a window above the deep
        # states, max η reflects only the occupied states in the window.
        e = _base_bands()
        e_b = e.copy()
        e_b[3] += 0.05            # occupied band (≈-1 eV) shifted 0.05
        e_b[0] += 100.0           # occupied deep band shifted 100 eV
        full = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                             fermi_a=0.0, fermi_b=0.0)
        occ = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0, occupied_only=True,
                            e_min=-4.5, e_max=10.0)
        # Without a window the 100 eV shift dominates max η.
        assert full["max_eta"] > 90.0
        # Window (above the deep band at ≈-5 eV) + occupied mask: the
        # deep shifted band is excluded, so max η only sees the 0.05
        # shift of the occupied band (partially absorbed by ω).
        assert 0.0 < occ["max_eta"] <= 0.05

    def test_no_occupied_states_gives_zero_max(self):
        # All states above E_F (unoccupied) → occupied mask is empty →
        # max η = 0.
        e = _base_bands() + 8.0
        e_b = e.copy()
        e_b[0] += 100.0
        res = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                            fermi_a=0.0, fermi_b=0.0, occupied_only=True)
        assert res["max_eta"] == 0.0


class TestAlignIndex:
    def test_index_with_window_masks_deep_states(self):
        # align="index" pairs by band index; without a window a shifted
        # deep band dominates η_v, with a window it is masked out.
        e = _base_bands()
        e_b = e.copy()
        e_b[0] += 100.0            # deep band (≈-5 eV) shifted 100 eV
        no_window = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                                  fermi_a=0.0, fermi_b=0.0, align="index")
        with_window = compare_bands(e, e_b, kpoints_a=KPOINTS, kpoints_b=KPOINTS,
                                    fermi_a=0.0, fermi_b=0.0, align="index",
                                    e_min=-4.5, e_max=10.0)
        assert no_window["eta_v"] > with_window["eta_v"]
        # The window excludes the shifted deep band; the remaining
        # shallow bands are unshifted → ω ≈ 0.
        assert with_window["omega"] == pytest.approx(0.0, abs=1e-6)

    def test_invalid_align_raises(self):
        with pytest.raises(ValueError):
            compare_bands(_base_bands(), _base_bands(), align="electrons")


class TestFormatTables:
    def test_markdown_structure(self):
        e = _base_bands()
        tbl = compare_all([("A", e, KPOINTS, 0.0), ("B", e + 0.05, KPOINTS, 0.0)])
        md = format_tables(tbl)
        assert "η_v" in md and "max η" in md and "ω (rigid shift)" in md
        # 3 tables × 1 separator line each (2 data columns + label col).
        assert md.count("| --- |") == 6
        # Values render with 4 decimals; NaN cells render as "—".
        assert "0.0000" in md
        assert "—" in md

    def test_roundtrip_number_format(self):
        labels = ["x", "y"]
        matrix = [[0.0, 0.123456], [0.123456, 0.0]]
        out = format_tables({"labels": labels, "eta_v": matrix,
                             "max_eta": matrix, "omega": matrix})
        assert "0.1235" in out  # 4 decimal places
