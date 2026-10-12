"""Invariants of the PEXSI presets and the solver-comparison scan.

PEXSI (pole expansion + selected inversion) has a set of hard
constraints that silently produce a *wrong but converged* number when
violated, so they are pinned by tests rather than only by comments:

* LCAO + Γ-only (``gamma_only 1`` with a 1×1×1 mesh);
* ``pexsi_temp`` is the temperature of the pole expansion's
  Fermi–Dirac occupation and is expressed in Ry, so it must equal
  ``smearing_sigma`` and the ABACUS-side smearing must be ``fd``;
* the pole count is bounded (npole 150 aborts with ``getPole error.``
  at σ = 0.1 eV) and must divide the MPI rank count via
  ``pexsi_nproc_pole``;
* ``lcao_1q_fdref`` is the *matched* genelpa reference for
  ``lcao_1q_pexsi``: identical numbers, only ``ks_solver`` differs —
  otherwise an energy/timing difference would not be the solver's.
"""

from __future__ import annotations

import yaml

from aiida_uranium_workflow.utils.common import PARAMETERS_DIR

SCF_YML = PARAMETERS_DIR / "abacus" / "scf.yml"
SUPERCELL_YML = PARAMETERS_DIR / "supercell.yml"

#: Number of MPI ranks the yeesuan jobs request (56 = one node).
MPI_RANKS = 56

#: PEXSI input keys the reference deliberately does NOT share.
PEXSI_ONLY_KEYS = (
    "ks_solver",
    "pexsi_temp",
    "pexsi_npole",
    "pexsi_nproc_pole",
    "pexsi_elec_thr",
    "pexsi_mu_thr",
    # pinned in the baseline so the verification steps (1_method, 3_delta_e)
    # are explicit variations on a documented baseline
    "pexsi_method",
    "pexsi_delta_e",
)


def _input(preset: str) -> dict:
    data = yaml.safe_load(SCF_YML.read_text())
    return data[preset]["abacus"]["parameters"]["input"]


def _kpoints(preset: str) -> list:
    data = yaml.safe_load(SCF_YML.read_text())
    return data[preset]["kpoints_mesh"]


def test_pexsi_preset_exists_and_is_gamma_only_lcao() -> None:
    inp = _input("lcao_1q_pexsi")
    assert inp["ks_solver"] == "pexsi"
    assert inp["basis_type"] == "lcao"
    assert inp["gamma_only"] == 1
    assert _kpoints("lcao_1q_pexsi") == [1, 1, 1]


def test_pexsi_temperature_matches_smearing() -> None:
    """PEXSI integrates a Fermi–Dirac occupation at ``pexsi_temp``; the
    ABACUS-side smearing must use the same scheme and the same width
    (both in Ry) or the density is built with a different occupation."""
    inp = _input("lcao_1q_pexsi")
    assert inp["smearing_method"] == "fd"
    assert inp["pexsi_temp"] == inp["smearing_sigma"]


def test_pexsi_pole_count_is_safe_and_rank_compatible() -> None:
    inp = _input("lcao_1q_pexsi")
    # npole 150 at σ = 0.1 eV aborts with "getPole error." on 3.9.0.25.
    assert 40 <= inp["pexsi_npole"] <= 100
    assert inp["pexsi_nproc_pole"] >= 1
    assert MPI_RANKS % inp["pexsi_nproc_pole"] == 0


def test_genelpa_reference_is_matched_except_for_the_solver() -> None:
    """``lcao_1q_fdref`` must differ from ``lcao_1q_pexsi`` only in
    ``ks_solver``, ``mixing_beta`` and the PEXSI-only keys, so the scan
    compares solvers and nothing else (same binary, same thresholds, same
    Γ mesh).

    ``mixing_beta`` is the one documented exception: the PEXSI SCF goes
    into a limit cycle at 0.5 (the 16-atom cell never converges) while
    genelpa converges at 0.5 in 12 steps. The converged energy is
    insensitive to it — measured 0.022 meV/atom between β 0.2 and β 0.5
    on the 16-atom cell.
    """
    pexsi, ref = _input("lcao_1q_pexsi"), _input("lcao_1q_fdref")
    assert ref["ks_solver"] == "genelpa"
    assert pexsi["mixing_beta"] == 0.2, "PEXSI limit-cycles at 0.5"
    assert ref["mixing_beta"] == 0.5, "genelpa's validated reference value"

    ignored = set(PEXSI_ONLY_KEYS) | {"ks_solver", "mixing_beta"}
    stripped = {k: v for k, v in pexsi.items() if k not in ignored}
    assert stripped == {k: v for k, v in ref.items() if k not in ignored}
    assert _kpoints("lcao_1q_fdref") == _kpoints("lcao_1q_pexsi") == [1, 1, 1]


def test_pecsi_scan_protocol_is_identical_across_cells() -> None:
    """All cells of the scan share one k-mesh and carry no per-cell SCF
    override: the size scaling must not be contaminated by a settings
    change between cells."""
    data = yaml.safe_load(SUPERCELL_YML.read_text())
    cells = data["pexsi_scan"]["abacus"]["supercells"]
    assert [cell["label"] for cell in cells] == [
        "1x1x1_2at",
        "2x2x2_16at",
        "3x3x3_54at",
        "4x4x4_128at",
    ]
    for index, cell in enumerate(cells, start=1):
        assert cell["matrix"] == [
            [index if row == col else 0 for col in range(3)] for row in range(3)
        ]
        assert cell["kpoints_mesh"] == [1, 1, 1]
        overrides = set(cell) - {"matrix", "label", "kpoints_mesh"}
        assert not overrides, f"unexpected per-cell override: {overrides}"


def test_pecsi_scan_small_omits_the_cells_pexsi_cannot_do() -> None:
    """PEXSI's density is already inaccurate at 54 atoms with this build,
    so the executable scan stops at 2×2×2."""
    data = yaml.safe_load(SUPERCELL_YML.read_text())
    cells = data["pexsi_scan_small"]["abacus"]["supercells"]
    assert [cell["label"] for cell in cells] == ["1x1x1_2at", "2x2x2_16at"]
    assert all(cell["kpoints_mesh"] == [1, 1, 1] for cell in cells)
    # the full ladder stays available for a rebuilt PEXSI
    assert len(data["pexsi_scan"]["abacus"]["supercells"]) == 4
