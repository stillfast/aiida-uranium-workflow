"""Tests for the three independent tables (Status / Energy / Time) in
the smear / convergence / magmom reports.

Each report's ``generate_report`` now emits three separate Markdown
tables, one per signal:

* **Calculation Status** — AiiDA exit codes (``0`` = OK, ``-`` = missing)
* **Total Energy [unit]** — total energy per cell, unit-aware
* **Wall Time [s]** — wall-clock seconds per cell

The tests below build representative ``output_params`` dicts and
verify each table is rendered correctly.
"""

from __future__ import annotations

import pytest

from aiida_uranium_workflow.utils.report.convergence import (
    generate_energy_table as generate_convergence_energy_table,
)
from aiida_uranium_workflow.utils.report.convergence import (
    generate_report as generate_convergence_report,
)
from aiida_uranium_workflow.utils.report.convergence import (
    generate_status_table as generate_convergence_status_table,
)
from aiida_uranium_workflow.utils.report.convergence import (
    generate_wall_time_table as generate_convergence_wall_time_table,
)
from aiida_uranium_workflow.utils.report.magmom import (
    generate_energy_table as generate_magmom_energy_table,
)
from aiida_uranium_workflow.utils.report.magmom import (
    generate_report as generate_magmom_report,
)
from aiida_uranium_workflow.utils.report.magmom import (
    generate_status_table as generate_magmom_status_table,
)
from aiida_uranium_workflow.utils.report.magmom import (
    generate_wall_time_table as generate_magmom_wall_time_table,
)
from aiida_uranium_workflow.utils.report.smear import (
    generate_energy_table as generate_smear_energy_table,
)
from aiida_uranium_workflow.utils.report.smear import (
    generate_report as generate_smear_report,
)
from aiida_uranium_workflow.utils.report.smear import (
    generate_status_table as generate_smear_status_table,
)
from aiida_uranium_workflow.utils.report.smear import (
    generate_wall_time_table as generate_smear_wall_time_table,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


SMALL_STATUS = {
    "smear_gauss_sigma_0_05": 0,
    "smear_mp_sigma_0_02": 0,
}
SMALL_ENERGY = {
    "smear_gauss_sigma_0_05": -1234.5,
    "smear_mp_sigma_0_02": -1234.4,
}
SMALL_TIME = {
    "smear_gauss_sigma_0_05": 12.3,
    "smear_mp_sigma_0_02": 30.0,
}


# ---------------------------------------------------------------------------
# Smear — three independent table generators
# ---------------------------------------------------------------------------


class TestSmearStatusTable:
    """``generate_status_table`` is the original exit-code table."""

    def test_basic_grid(self) -> None:
        out = generate_smear_status_table(SMALL_STATUS, sigma_unit="ev")
        # Columns are sorted by sigma ascending: 0.02, 0.05.
        # gauss row: 0.02 (mp) = -, 0.05 (gauss) = 0
        # mp row: 0.02 (mp) = 0, 0.05 (gauss) = -
        assert "| gauss | - | 0 |" in out
        assert "| mp | 0 | - |" in out

    def test_missing_label_renders_dash(self) -> None:
        out = generate_smear_status_table(
            {"smear_gauss_sigma_0_05": 0}, sigma_unit="ev"
        )
        # The mp row should have a ``-`` cell.
        assert "| mp | - |" in out or "| -" in out


class TestSmearEnergyTable:
    def test_unit_annotation_eV(self) -> None:
        out = generate_smear_energy_table(SMALL_ENERGY, sigma_unit="ev")
        assert "| smearing_method\\total_energy [eV] |" in out

    def test_unit_annotation_Ry(self) -> None:
        out = generate_smear_energy_table(SMALL_ENERGY, sigma_unit="ry")
        assert "| smearing_method\\total_energy [Ry] |" in out

    def test_cells_render_six_decimals(self) -> None:
        out = generate_smear_energy_table(SMALL_ENERGY, sigma_unit="ev")
        # Columns sorted ascending: sigma 0.02 (mp) first, 0.05 (gauss) last.
        # gauss row: 0.02 column = — (no gauss at 0.02), 0.05 column = -1234.500000
        # mp row:    0.02 column = -1234.400000, 0.05 column = — (no mp at 0.05)
        assert "| gauss | — | -1234.500000 |" in out
        assert "| mp | -1234.400000 | — |" in out

    def test_none_value_renders_dash(self) -> None:
        out = generate_smear_energy_table(
            {"smear_gauss_sigma_0_05": None, "smear_mp_sigma_0_02": -1.0},
            sigma_unit="ev",
        )
        assert "— |" in out
        assert "-1.000000" in out

    def test_empty_dict_returns_placeholder(self) -> None:
        assert "No data" in generate_smear_energy_table({}, sigma_unit="ev")


class TestSmearWallTimeTable:
    def test_header(self) -> None:
        out = generate_smear_wall_time_table(SMALL_TIME)
        assert "| smearing_method\\wall_time [s] |" in out

    def test_cells_render_three_decimals(self) -> None:
        out = generate_smear_wall_time_table(SMALL_TIME)
        # gauss row: 0.02 column = — (no gauss at 0.02), 0.05 column = 12.300
        # mp row:    0.02 column = 30.000, 0.05 column = — (no mp at 0.05)
        assert "| gauss | — | 12.300 |" in out
        assert "| mp | 30.000 | — |" in out

    def test_none_value_renders_dash(self) -> None:
        out = generate_smear_wall_time_table({"smear_gauss_sigma_0_05": None})
        assert "— |" in out

    def test_empty_dict_returns_placeholder(self) -> None:
        assert "No data" in generate_smear_wall_time_table({})


# ---------------------------------------------------------------------------
# Smear report — integration
# ---------------------------------------------------------------------------


class TestSmearReportHasThreeTables:
    def test_all_three_sections_present(self) -> None:
        params = {
            "status": SMALL_STATUS,
            "total_energy": SMALL_ENERGY,
            "wall_time_seconds": SMALL_TIME,
        }
        report = generate_smear_report(params, pk=1, workflow_type="vasp")
        assert "## Calculation Status" in report
        assert "## Total Energy [eV]" in report
        assert "## Wall Time [s]" in report
        # Each table renders the values once, in its own grid.
        assert "0 |" in report  # status row
        assert "-1234.500000" in report  # energy row
        assert "12.300" in report  # time row

    def test_ry_unit_in_abacus(self) -> None:
        report = generate_smear_report(
            {"status": SMALL_STATUS, "total_energy": SMALL_ENERGY},
            pk=1,
            workflow_type="abacus",
        )
        assert "## Total Energy [Ry]" in report

    def test_status_only_when_other_keys_missing(self) -> None:
        report = generate_smear_report(
            {"status": SMALL_STATUS}, pk=1, workflow_type="vasp"
        )
        assert "## Calculation Status" in report
        assert "## Total Energy" not in report
        assert "## Wall Time" not in report


# ---------------------------------------------------------------------------
# Convergence — three independent table generators
# ---------------------------------------------------------------------------


class TestConvergenceStatusTable:
    def test_basic_grid(self) -> None:
        out = generate_convergence_status_table(
            {"ecutwfc_80_kpoints_distance_0_1": 0},
            workflow_type="abacus",
        )
        # The header includes ecutwfc (Ry) for ABACUS.
        assert "ecutwfc (Ry)" in out
        # Cell value 0 is rendered.
        assert "| 0 |" in out


class TestConvergenceEnergyTable:
    def test_header_includes_unit(self) -> None:
        out = generate_convergence_energy_table(
            {"ecutwfc_80_kpoints_distance_0_1": -123.45},
            workflow_type="abacus",
        )
        assert "ecutwfc (Ry)" in out
        assert "-123.450000" in out

    def test_vasp_uses_eV(self) -> None:
        out = generate_convergence_energy_table(
            {"encut_300_kpoints_spacing_0_1": -42.0}, workflow_type="vasp"
        )
        assert "encut (eV)" in out
        assert "-42.000000" in out

    def test_none_renders_dash(self) -> None:
        out = generate_convergence_energy_table(
            {"ecutwfc_80_kpoints_distance_0_1": None}, workflow_type="abacus"
        )
        assert "—" in out

    def test_empty(self) -> None:
        assert "No data" in generate_convergence_energy_table(
            {}, workflow_type="abacus"
        )


class TestConvergenceWallTimeTable:
    def test_basic(self) -> None:
        out = generate_convergence_wall_time_table(
            {"ecutwfc_80_kpoints_distance_0_1": 60.0}, workflow_type="abacus"
        )
        assert "ecutwfc (Ry)" in out
        assert "| 60.000 |" in out

    def test_none_renders_dash(self) -> None:
        out = generate_convergence_wall_time_table(
            {"ecutwfc_80_kpoints_distance_0_1": None}, workflow_type="abacus"
        )
        assert "—" in out

    def test_empty(self) -> None:
        assert "No data" in generate_convergence_wall_time_table(
            {}, workflow_type="abacus"
        )


class TestConvergenceReportHasThreeTables:
    def test_all_three_sections_present(self) -> None:
        params = {
            "status": {"ecutwfc_80_kpoints_distance_0_1": 0},
            "total_energy": {"ecutwfc_80_kpoints_distance_0_1": -123.45},
            "wall_time_seconds": {"ecutwfc_80_kpoints_distance_0_1": 60.0},
            "total_energy_per_atom": {"ecutwfc_80_kpoints_distance_0_1": -12.345},
            "num_atoms": {"ecutwfc_80_kpoints_distance_0_1": 1},
        }
        report = generate_convergence_report(params, pk=1, workflow_type="abacus")
        assert "## Calculation Status" in report
        assert "## Total Energy [Ry]" in report
        assert "## Wall Time [s]" in report

    def test_status_value_appears_in_table(self) -> None:
        """The numeric exit code (0 / -1 / 300) ends up in the Markdown cell."""
        params = {
            "status": {"ecutwfc_80_kpoints_distance_0_1": 0},
            "total_energy": {"ecutwfc_80_kpoints_distance_0_1": -123.45},
        }
        report = generate_convergence_report(params, pk=1, workflow_type="abacus")
        # The status table renders the exit code as a plain integer.
        assert "| 0 |" in report


# ---------------------------------------------------------------------------
# Magmom — three independent table generators
# ---------------------------------------------------------------------------


MAGMOM_STATUS = {1: 0, 2: 0}
MAGMOM_ENERGY = {1: -123.45, 2: -123.50}
MAGMOM_TIME = {1: 12.5, 2: 14.0}


class TestMagmomStatusTable:
    def test_basic(self) -> None:
        out = generate_magmom_status_table(MAGMOM_STATUS)
        assert "| child pk | exit_status |" in out
        assert "| 1 | 0 |" in out
        assert "| 2 | 0 |" in out

    def test_empty(self) -> None:
        assert "No status data" in generate_magmom_status_table({})


class TestMagmomEnergyTable:
    def test_header_and_cells(self) -> None:
        out = generate_magmom_energy_table(MAGMOM_ENERGY)
        assert "| child pk | final_energy |" in out
        assert "| 1 | -123.450000 |" in out
        assert "| 2 | -123.500000 |" in out

    def test_none_renders_dash(self) -> None:
        out = generate_magmom_energy_table({1: None, 2: -1.0})
        assert "|" in out
        assert "-1.000000" in out
        assert "—" in out

    def test_empty(self) -> None:
        assert "No data" in generate_magmom_energy_table({})


class TestMagmomWallTimeTable:
    def test_header_and_cells(self) -> None:
        out = generate_magmom_wall_time_table(MAGMOM_TIME)
        assert "| child pk | wall_time [s] |" in out
        assert "| 1 | 12.500 |" in out
        assert "| 2 | 14.000 |" in out

    def test_none_renders_dash(self) -> None:
        out = generate_magmom_wall_time_table({1: None})
        assert "—" in out

    def test_empty(self) -> None:
        assert "No data" in generate_magmom_wall_time_table({})


class TestMagmomReportHasThreeTables:
    def test_all_three_sections_present(self) -> None:
        params = {
            "status": MAGMOM_STATUS,
            "final_energy": MAGMOM_ENERGY,
            "wall_time_seconds": MAGMOM_TIME,
        }
        report = generate_magmom_report(params, pk=1, workflow_type="abacus")
        assert "## Calculation Status" in report
        assert "## Final Energy" in report
        assert "## Wall Time [s]" in report
        # Each table renders the right values.
        assert "| 1 | 0 |" in report  # status row
        assert "-123.450000" in report  # energy row
        assert "12.500" in report  # time row


# ---------------------------------------------------------------------------
# Magmom — FLEUR integration (per-atom 3-vectors, Hartree energies)
# ---------------------------------------------------------------------------


FLEUR_MAGMOM = {
    "magnetization": {
        101: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # NM
        102: [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]],  # FM
        103: [[0.0, 0.0, 4.0], [0.0, 0.0, -4.0]],  # AFM
    },
    "total_energy_hartree": {101: -56165.979, 102: -56165.999, 103: -56165.955},
    "final_energy": {101: -1528536.0, 102: -1528536.5, 103: -1528535.8},
    "wall_time_seconds": {101: 900.0, 102: 950.0, 103: 920.0},
    "status": {101: 0, 102: 0, 103: 0},
    "config_labels": {101: "NM", 102: "FM", 103: "AFM"},
    "magmom_list": [
        {"label": "NM", "bmu": 0.0},
        {
            "label": "FM",
            "bmu": 4.0,
            "inpxml_changes": [["set_species", {"species_name": "all"}]],
        },
        {
            "label": "AFM",
            "bmu": 4.0,
            "inpxml_changes": [["set_species", {}], ["set_atomgroup", {"position": 2}]],
        },
    ],
}


class TestFleurMagmomReport:
    def test_summary_shows_fleur_backend(self) -> None:
        report = generate_magmom_report(FLEUR_MAGMOM, pk=7, workflow_type="fleur")
        assert "| backend | fleur |" in report
        assert "| magnetization entries | 3 |" in report
        assert "| total_energy_hartree entries | 3 |" in report

    def test_magnetism_matrix_per_atom_and_total(self) -> None:
        report = generate_magmom_report(FLEUR_MAGMOM, pk=7, workflow_type="fleur")
        assert "## Magnetism Matrix" in report
        # FM row: per-atom +z, cell total +z 8.0 (matrix table).
        assert "[+0.00, +0.00, +4.00]" in report
        assert "[+0.00, +0.00, +8.00]" in report
        # AFM row: moments cancel -> cell total 0, per-atom +4 / -4.
        assert "[+0.00, +0.00, -4.00]" in report

    def test_delta_energy_mev_column(self) -> None:
        report = generate_magmom_report(FLEUR_MAGMOM, pk=7, workflow_type="fleur")
        # FM is 0.020 Ha below the NM reference -> -544 meV (Ha -> meV).
        assert "| -544.228 |" in report
        # AFM is 0.024 Ha above -> +653 meV.
        assert "| +653.073 |" in report

    def test_initial_magmom_bmu(self) -> None:
        report = generate_magmom_report(FLEUR_MAGMOM, pk=7, workflow_type="fleur")
        # FLEUR bmu seeds render in the M_start columns (scalar +0.00 / +4.00).
        assert "| +0.00 | +0.00 |" in report  # NM row M_start_U1/U2
        assert "| +4.00 | +4.00 |" in report  # FM row M_start_U1/U2
        # The inpxml_changes payload must not leak into the report.
        assert "set_species" not in report

    def test_matrix_table_fleur_cell_total(self) -> None:
        report = generate_magmom_report(FLEUR_MAGMOM, pk=7, workflow_type="fleur")
        # Matrix table: FM cell total [0, 0, 8].
        assert "[+0.00, +0.00, +8.00]" in report


# ---------------------------------------------------------------------------
# Relax — FLEUR EOS + ABACUS volume-only reports
# ---------------------------------------------------------------------------


class TestRelaxReport:
    def test_fleur_full_relax_report(self) -> None:
        from aiida_uranium_workflow.utils.report.relax import generate_report

        out = {
            "workflow": "relax",
            "backend": "fleur",
            "mode": "full-relax",
            "lattice_constants": [3.4581, 3.4581, 3.4581],
            "lattice_constant": 3.4581,
            "volume": 25.83,
            "energy": -1528353.919388,
            "energy_units": "eV",
            "relax_energy_hartree": -56165.97,
        }
        report = generate_report(out, pk=7, workflow_type="fleur")
        assert "mode | full relax (positions + cell)" in report
        assert "[3.458100, 3.458100, 3.458100] Å" in report
        assert "-1528353.919388 eV" in report
        assert "## EOS Volume Scan" not in report

    def test_abacus_volume_report(self) -> None:
        from aiida_uranium_workflow.utils.report.relax import generate_report

        out = {
            "workflow": "relax",
            "backend": "abacus",
            "mode": "volume",
            "lattice_constants": [3.4581, 3.4581, 3.4581],
            "lattice_constant": 3.4581,
            "volume": 25.83,
            "energy": -1528353.919388,
            "energy_units": "eV",
        }
        report = generate_report(out, pk=8, workflow_type="abacus")
        assert "mode | volume-only relax" in report
        assert "[3.458100, 3.458100, 3.458100] Å" in report
        assert "-1528353.919388 eV" in report
        assert "## EOS Volume Scan" not in report


class TestRelaxDeriveFromNode:
    """_derive_from_node must respect the plugin's ``energy_units``."""

    @staticmethod
    def _node(outputs):
        class _Outs:
            pass

        outs = _Outs()
        for k, v in outputs.items():
            setattr(outs, k, v)
        node = type("Node", (), {"outputs": outs})()
        return node

    @staticmethod
    def _structure():
        import numpy as np
        from ase.build import bulk

        cell = np.asarray(bulk("U", "bcc", a=3.45).cell)

        def _vol(self):
            return float(np.linalg.det(cell))

        return type("Struct", (), {
            "cell": cell,
            "get_cell_volume": _vol,
        })()

    def _fleur_node(self, energy, units):
        def _get_dict(self):
            return {"last_energy": energy, "energy_units": units}

        para = type("P", (), {"get_dict": _get_dict})()
        return self._node({"optimized_structure": self._structure(),
                           "output_relax_wc_para": para})

    def test_fleur_eV_not_rescaled(self):
        from aiida_uranium_workflow.utils.report.relax import _derive_from_node
        data = _derive_from_node(self._fleur_node(-1528354.014, "eV"), "fleur")
        # aiida-fleur already converted Ha → eV; do NOT multiply again.
        assert data["energy"] == pytest.approx(-1528354.014, abs=1e-6)

    def test_fleur_last_energy_used_as_ev_regardless_of_units(self):
        # aiida-fleur 2.0.0's ``last_energy`` is already an eV numerical
        # value (bcc-U 2 atoms ≈ -1 528 354 eV, matching the EOS report);
        # the ``energy_units`` label is unreliable — always use as eV.
        from aiida_uranium_workflow.utils.report.relax import _derive_from_node
        data = _derive_from_node(self._fleur_node(-1528354.014, "Htr"), "fleur")
        assert data["energy"] == pytest.approx(-1528354.014, abs=1e-6)

    def test_lattice_and_volume_derived(self):
        import numpy as np
        from ase.build import bulk
        from aiida_uranium_workflow.utils.report.relax import _derive_from_node

        cell = np.asarray(bulk("U", "bcc", a=3.45).cell)
        data = _derive_from_node(self._fleur_node(-1.0, "eV"), "fleur")
        assert data["lattice_constant"] == pytest.approx(
            float(np.linalg.norm(cell[0])), abs=1e-6)
        assert data["volume"] == pytest.approx(
            float(np.linalg.det(cell)), abs=1e-3)


# ---------------------------------------------------------------------------
# Magmom — Magnetism Matrix table (compact per-configuration format)
# ---------------------------------------------------------------------------


class TestMagmomMatrixTable:
    """The compact pk-keyed magnetism matrix table."""

    def test_vasp_matrix(self) -> None:
        from aiida_uranium_workflow.utils.report.magmom import (
            generate_magmom_matrix_table,
        )

        # VASP: magnetization is the cell total (vector for SOC / scalar
        # for nosoc); site_magnetization holds per-site moments when parsed.
        params = {
            "final_energy": {342714: -27.26117914, 342720: -27.26441465},
            "magnetization": {342714: [0.0, 0.0, 0.0], 342720: [0.0, 0.0, 2.09678]},
            "site_magnetization": {
                342714: {"sphere": {"x": {"site_moment": {}}}, "full_cell": [0.0]},
                342720: {"sphere": {"x": {"site_moment": {}}}, "full_cell": [2.1]},
            },
            "magmom_list": [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]],
            ],
        }
        table = generate_magmom_matrix_table(params, workflow_type="vasp")
        # E in eV; cell total from magnetization.
        assert "| 342714 | -27.26117914 | +0.000 |" in table
        assert "[+0.00, +0.00, +2.10]" in table
        # No per-site moments parsed -> M_U final columns hidden entirely.
        assert "M_U1 (µ_B)" not in table
        assert "| — | — |" not in table
        # Per-atom initial magmom (magmom_per_atom_list injected as magmom_list).
        assert "| [+0.00, +0.00, +0.00] | [+0.00, +0.00, +0.00] |" in table
        assert "| [+0.00, +0.00, +4.00] | [+0.00, +0.00, +4.00] |" in table

    def test_vasp_matrix_with_site_moments(self) -> None:
        from aiida_uranium_workflow.utils.report.magmom import (
            generate_magmom_matrix_table,
        )

        # When VASP per-site moments are parsed, the M_U columns appear.
        params = {
            "final_energy": {1: -100.0, 2: -99.5},
            "magnetization": {1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 4.0]},
            "site_magnetization": {
                1: {"sphere": {"x": {"site_moment": {"0": 0.0, "1": 0.0}}}},
                2: {"sphere": {"x": {"site_moment": {"0": 2.0, "1": 2.0}}}},
            },
            "magmom_list": [[[0.0], [0.0]], [[4.0], [4.0]]],
        }
        table = generate_magmom_matrix_table(params, workflow_type="vasp")
        assert "M_U1 (µ_B) | M_U2 (µ_B)" in table
        assert (
            "| 2 | -99.50000000 | +500.000 | [+0.00, +0.00, +4.00] | +2.00 | +2.00 |"
            in table
        )

    def test_abacus_nosoc_scalar_cell(self) -> None:
        from aiida_uranium_workflow.utils.report.magmom import (
            generate_magmom_matrix_table,
        )

        # nosoc (collinear) ABACUS: cell total is a scalar, not a vector.
        params = {
            "final_energy": {1: -100.0, 2: -99.5},
            "final_magnetism": {1: 0.0, 2: 2.65},
            "magmom_list": [[[0.0], [0.0]], [[4.0], [4.0]]],
        }
        table = generate_magmom_matrix_table(params, workflow_type="abacus")
        assert "| 1 | -100.00000000 | +0.000 | +0.00 |" in table
        assert "| 2 | -99.50000000 | +500.000 | +2.65 |" in table

    def test_abacus_matrix(self) -> None:
        from aiida_uranium_workflow.utils.report.magmom import (
            generate_magmom_matrix_table,
        )

        params = {
            "final_energy": {342670: -27051.032119701, 342678: -27051.031759607},
            "final_magnetism": {
                342670: {"total_magnetism": [0.0, 0.0, 5.49e-06]},
                342678: {"total_magnetism": [0.0, 0.0, 0.543068531506]},
            },
            "magmom_list": [[[0.0], [0.0]], [[4.0], [4.0]]],
        }
        table = generate_magmom_matrix_table(params, workflow_type="abacus")
        # Row keys are the pks; NM (first entry, zero magmom) is the reference.
        assert "| 342670 | -27051.03211970 | +0.000 |" in table
        assert "| 342678 | -27051.03175961 | +0.360 |" in table
        # ABACUS has no per-atom final moments -> M_U final columns are
        # hidden entirely (M_cell is the only magnetism output).
        assert "M_U1 (µ_B)" not in table
        assert "| — | — |" not in table
        # M_start columns remain (from the injected magmom_list).
        assert "M_start_U1 (µ_B) | M_start_U2 (µ_B)" in table

    def test_fleur_matrix(self) -> None:
        from aiida_uranium_workflow.utils.report.magmom import (
            generate_magmom_matrix_table,
        )

        table = generate_magmom_matrix_table(FLEUR_MAGMOM, workflow_type="fleur")
        # E in Hartree; per-atom final vectors present; cell total is the sum.
        assert "| 101 | -56165.97900000 | +0.000 |" in table
        assert "[+0.00, +0.00, +4.00]" in table  # M_U1 FM
        assert "[+0.00, +0.00, +8.00]" in table  # M_cell FM
        assert "[+0.00, +0.00, -4.00]" in table  # M_U2 AFM
        assert "M_start_U1 (µ_B)" in table
        assert "| +4.00 | +4.00 |" in table  # bmu seeds


# ---------------------------------------------------------------------------
# Overview table (improve.md Phase B) — shared per-label summary
# ---------------------------------------------------------------------------


class TestRenderOverviewTable:
    """The canonical energy / time / scf-steps per-label table."""

    def test_basic(self):
        from aiida_uranium_workflow.utils.report._common import (
            render_overview_table,
        )

        out = {
            "total_energy": {"a": -10.5, "b": -9.0},
            "wall_time_seconds": {"a": 60.0, "b": None},
            "scf_steps": {"a": 12, "b": 30},
            "status": {"a": 0, "b": 0},
        }
        table = render_overview_table(out)
        assert "| label | energy (eV) | time (s) | scf steps | exit |" in table
        assert "| a | -10.500000 | 60.000000 | 12.000000 | 0 |" in table
        assert "| b | -9.000000 | — | 30.000000 | 0 |" in table

    def test_empty(self):
        from aiida_uranium_workflow.utils.report._common import (
            render_overview_table,
        )

        assert render_overview_table({"eentropy": {}}) == ""

    def test_final_energy_fallback(self):
        from aiida_uranium_workflow.utils.report._common import (
            render_overview_table,
        )

        out = {
            "final_energy": {"x": -1.0},
            "wall_time_seconds": {"x": 5.0},
        }
        table = render_overview_table(out)
        assert "| x | -1.000000 | 5.000000 | — | — |" in table

    def test_pk_keyed_magmom(self):
        from aiida_uranium_workflow.utils.report._common import (
            render_overview_table,
        )

        out = {
            "final_energy": {342670: -27066.5},
            "wall_time_seconds": {342670: 75.0},
            "scf_steps": {342670: 22},
        }
        table = render_overview_table(out)
        assert "| 342670 | -27066.500000 | 75.000000 | 22.000000 | — |" in table
