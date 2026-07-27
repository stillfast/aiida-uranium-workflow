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
