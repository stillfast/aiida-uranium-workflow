"""Tests for the defect workflow (structure generation, adapters, workchain)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import numpy as np
import pytest

from aiida_uranium_workflow.input_builders.defects.abacus import AbacusDefectsAdapter
from aiida_uranium_workflow.input_builders.defects.fleur import FleurDefectsAdapter
from aiida_uranium_workflow.schedulers.defects import parse_defects_protocol
from aiida_uranium_workflow.utils.defects import (
    analyse_defect_species,
    create_interstitial,
    create_vacancy,
    formation_energy,
    make_supercell,
)
from aiida_uranium_workflow.utils.report.defects import generate_report


def _bcc_uranium():
    from aiida import orm
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


class TestProtocolParser:
    def test_splits_backends(self):
        proto = {
            "abacus": {"defect": {"type": "vacancy"}},
            "fleur": {"defect": {"type": "interstitial"}},
        }
        out = parse_defects_protocol(proto)
        assert out["abacus"]["defect"]["type"] == "vacancy"
        assert out["fleur"]["defect"]["type"] == "interstitial"

    def test_empty(self):
        assert parse_defects_protocol(None) == {}
        assert parse_defects_protocol({}) == {}


class TestStructureGeneration:
    def test_make_supercell(self):
        s = _bcc_uranium()
        sup = make_supercell(s, [2, 2, 2])
        assert len(sup.sites) == 8 * len(s.sites)
        # Cell volume ×8.
        assert sup.get_cell_volume() == pytest.approx(8 * s.get_cell_volume())

    def test_vacancy(self):
        s = _bcc_uranium()
        sup = make_supercell(s, [2, 2, 2])
        vac, removed = create_vacancy(sup, 0)
        assert len(vac.sites) == len(sup.sites) - 1
        assert removed == "U"
        # Cell unchanged.
        assert vac.get_cell_volume() == pytest.approx(sup.get_cell_volume())

    def test_vacancy_out_of_range(self):
        sup = make_supercell(_bcc_uranium(), [2, 2, 2])
        with pytest.raises(ValueError):
            create_vacancy(sup, 10 ** 6)

    def test_interstitial(self):
        s = _bcc_uranium()
        sup = make_supercell(s, [2, 2, 2])
        inter = create_interstitial(sup, "U", [0.25, 0.25, 0.25])
        assert len(inter.sites) == len(sup.sites) + 1
        assert inter.get_cell_volume() == pytest.approx(sup.get_cell_volume())

    def test_analyse_defect_species(self):
        s = _bcc_uranium()
        sup = make_supercell(s, [2, 2, 2])
        vac, _ = create_vacancy(sup, 0)
        inter = create_interstitial(sup, "U", [0.25, 0.25, 0.25])
        assert analyse_defect_species(sup, vac) == {"removed": {"U": 1}, "inserted": {}}
        assert analyse_defect_species(sup, inter) == {"removed": {}, "inserted": {"U": 1}}


class TestFormationEnergy:
    def test_vacancy_atom_scaled(self):
        """E_f = E_defect − E_host × (N_defect / N_host), no μ.

        Vacancy: host 8 atoms (-110 eV), defect 7 atoms (-100 eV):
        E_f = -100 - (-110)·7/8 = -3.75 eV.
        """
        res = formation_energy(
            defect_energy_ev=-100.0,
            host_energy_ev=-110.0,
            defect_natoms=7,
            host_natoms=8,
        )
        assert res["formation_energy_ev"] == pytest.approx(-3.75)
        assert res["energy_difference_ev"] == pytest.approx(10.0)
        assert res["host_natoms"] == 8
        assert res["defect_natoms"] == 7
        assert "chemical_potential" not in res

    def test_interstitial_atom_scaled(self):
        """Interstitial: host 8 atoms (-110 eV), defect 9 atoms (-115 eV):
        E_f = -115 - (-110)·9/8 = +8.75 eV.
        """
        res = formation_energy(
            defect_energy_ev=-115.0,
            host_energy_ev=-110.0,
            defect_natoms=9,
            host_natoms=8,
        )
        assert res["formation_energy_ev"] == pytest.approx(8.75)
        assert res["energy_difference_ev"] == pytest.approx(-5.0)


class TestAdapters:
    def _abacus_adapter(self, code_label):
        return AbacusDefectsAdapter(
            code_label=code_label,
            software_params={
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints_mesh": [4, 4, 4],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "abacus": {
                    "supercell_matrix": [2, 2, 2],
                    "defect": {
                        "type": "vacancy", "site_index": 0,
                        "element": "U", "label": "vac_U_0",
                    },
                    "wf_parameters": {"mode": "scf"},
                }
            },
            extra_codes={},
        )

    def _fleur_adapter(self, code_label, inpgen_label):
        return FleurDefectsAdapter(
            code_label=code_label,
            software_params={
                "wf_parameters": {"mode": "density", "itmax_per_run": 100},
                "calc_parameters": {"comp": {"kmax": 7.0}},
            },
            metadata={},
            workflow_data={
                "fleur": {
                    "supercell_matrix": [2, 2, 2],
                    "defect": {
                        "type": "interstitial", "element": "U",
                        "position": [0.25, 0.25, 0.25], "label": "int_U",
                    },
                    "wf_parameters": {"mode": "relax"},
                }
            },
            extra_codes={"inpgen": inpgen_label},
        )

    def test_abacus_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm

        computer = aiida_localhost
        orm.InstalledCode(
            label="defects_test_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        adapter = self._abacus_adapter(f"defects_test_abacus@{computer.label}")
        inputs = adapter._build_workchain_inputs(_bcc_uranium())

        assert inputs["defect"].get_dict()["type"] == "vacancy"
        assert inputs["supercell_matrix"].get_list() == [2, 2, 2]
        assert inputs["wf_parameters"].get_dict() == {"mode": "scf"}
        assert "abacus" in inputs["base"]
        assert "code" in inputs["base"]["abacus"]

    def test_fleur_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        computer = aiida_localhost
        orm.InstalledCode(
            label="defects_test_fleur",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="defects_test_inpgen",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        adapter = self._fleur_adapter(
            f"defects_test_fleur@{computer.label}",
            f"defects_test_inpgen@{computer.label}",
        )
        inputs = adapter._build_workchain_inputs(_bcc_uranium())

        assert inputs["defect"].get_dict()["type"] == "interstitial"
        assert inputs["wf_parameters"].get_dict() == {"mode": "relax"}
        assert "fleur" in inputs["base"]
        assert "inpgen" in inputs["base"]
        err = WorkflowFactory("uranium.defects.fleur").spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"


class TestReport:
    def test_generate_report(self):
        md = generate_report(
            {
                "defect": {"type": "vacancy", "label": "vac_U_0"},
                "mode": "scf",
                "host_natoms": 8,
                "defect_natoms": 7,
                "host_energy_ev": -110.0,
                "defect_energy_ev": -100.0,
                "energy_difference_ev": 10.0,
                "formation_energy_ev": -3.75,
                "formula": "E_defect − E_host × N_defect/N_host",
            },
            7,
            "abacus",
        )
        assert "Defect Formation Energy Report" in md
        assert "formation energy | -3.750000 eV" in md
        assert "formula | E_defect − E_host × N_defect/N_host" in md
        assert "vacancy" in md


class TestSupercellMatrixSpec:
    """Normalisation of the supercell_matrix / supercell_matrices preset keys."""

    def test_single_diagonal(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        assert normalize_supercell_matrices([2, 2, 2]) == [[2, 2, 2]]
        assert normalize_supercell_matrices(None) == []

    def test_single_3x3(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        m = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        assert normalize_supercell_matrices(m) == [m]

    def test_list_of_diagonals(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        assert normalize_supercell_matrices([[2, 2, 2], [3, 3, 3]]) == [
            [2, 2, 2],
            [3, 3, 3],
        ]

    def test_three_diagonal_rows_are_three_cells(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        assert normalize_supercell_matrices([[2, 2, 2], [3, 3, 3], [4, 4, 4]]) == [
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]

    def test_mixed_list(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        m = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert normalize_supercell_matrices([m, [3, 3, 3]]) == [m, [3, 3, 3]]

    def test_labels(self):
        from aiida_uranium_workflow.utils.defects import supercell_label

        assert supercell_label([2, 2, 2]) == "2x2x2"
        assert supercell_label([[2, 0, 0], [0, 3, 0], [0, 0, 4]]) == "2x3x4"
        assert supercell_label([[2, 1, 0], [0, 2, 0], [0, 0, 2]]) == "m3x3"

    def test_bad_spec_raises(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_matrices

        with pytest.raises(ValueError):
            normalize_supercell_matrices([2, 2])
        with pytest.raises(TypeError):
            normalize_supercell_matrices("2x2x2")


class TestMultiCellReport:
    def test_cells_table_rendered(self):
        from aiida_uranium_workflow.utils.report.defects import generate_report

        params = {
            "workflow": "defects",
            "mode": "scf",
            "defect": {"type": "vacancy", "label": "vac_U_0"},
            "n_cells": 2,
            "cells": [
                {"label": "2x2x2", "matrix": [2, 2, 2], "host_natoms": 8,
                 "defect_natoms": 7, "host_energy_ev": -110.0,
                 "defect_energy_ev": -100.0, "formation_energy_ev": -3.75},
                {"label": "3x3x3", "matrix": [3, 3, 3], "host_natoms": 27,
                 "defect_natoms": 26, "host_energy_ev": -371.0,
                 "defect_energy_ev": -360.0, "formation_energy_ev": -2.85},
            ],
        }
        report = generate_report(params, 42, "abacus")
        assert "Supercell convergence (2 cells)" in report
        assert "| 2x2x2 |" in report and "| 3x3x3 |" in report
        assert "-3.7500" in report

    def test_single_cell_has_no_table(self):
        from aiida_uranium_workflow.utils.report.defects import generate_report

        params = {
            "defect": {"type": "vacancy"},
            "n_cells": 1,
            "cells": [{"label": "2x2x2", "formation_energy_ev": -3.75}],
        }
        report = generate_report(params, 42, "abacus")
        assert "Supercell convergence" not in report


class TestPerCellKpointsSpec:
    """supercell_cells: per-cell matrix + k-mesh specs."""

    def test_cells_with_meshes(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

        spec = [
            {"matrix": [2, 2, 2], "kpoints_mesh": [7, 7, 7]},
            {"matrix": [3, 3, 3], "kpoints_mesh": [5, 5, 5], "label": "big"},
        ]
        assert normalize_supercell_cells(spec) == [
            {"matrix": [2, 2, 2], "kpoints_mesh": [7, 7, 7]},
            {"matrix": [3, 3, 3], "kpoints_mesh": [5, 5, 5], "label": "big"},
        ]

    def test_distance_and_bare_matrix_mixed(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

        spec = [{"matrix": [2, 2, 2], "kpoints_distance": 0.1}, [3, 3, 3]]
        assert normalize_supercell_cells(spec) == [
            {"matrix": [2, 2, 2], "kpoints_distance": 0.1},
            {"matrix": [3, 3, 3]},
        ]

    def test_unknown_cell_key_raises(self):
        from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

        with pytest.raises(ValueError, match="unknown per-cell keys"):
            normalize_supercell_cells([{"matrix": [2, 2, 2], "nonsense": 1}])

    def test_per_cell_scf_override_is_idempotent(self):
        """Normalising twice must not fail (the CLI adapter and the WorkChain
        both normalise the protocol).

        Regression: the adapter stores cells carrying ``{'scf': {...}}`` in
        the ``supercell_cells`` List; the WorkChain normalises them again, so
        a second pass has to accept the already-collected ``scf`` mapping
        instead of raising "unknown per-cell keys ['scf']".
        """
        from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

        spec = [
            {"matrix": [5, 5, 5], "mixing_ndim": 16},
            {"matrix": [6, 6, 6], "mixing_ndim": 24, "scf_nmax": 250},
        ]
        once = normalize_supercell_cells(spec)
        twice = normalize_supercell_cells(once)
        assert once == twice
        assert once[0]["scf"] == {"mixing_ndim": 16}
        assert once[1]["scf"] == {"mixing_ndim": 24, "scf_nmax": 250}

    def test_per_cell_scf_unknown_key_raises(self):
        """An unknown key inside the collected ``scf`` mapping is rejected."""
        from aiida_uranium_workflow.utils.defects import normalize_supercell_cells

        with pytest.raises(ValueError, match="unknown SCF override keys"):
            normalize_supercell_cells([{"matrix": [2, 2, 2], "scf": {"nope": 1}}])


class TestPerCellKpointsReport:
    def test_kmesh_column(self):
        from aiida_uranium_workflow.utils.report.defects import generate_cells_table

        table = generate_cells_table(
            {
                "cells": [
                    {"label": "2x2x2", "matrix": [2, 2, 2], "kpoints_mesh": [7, 7, 7],
                     "host_natoms": 8, "defect_natoms": 7,
                     "host_energy_ev": -110.0, "defect_energy_ev": -100.0,
                     "formation_energy_ev": -3.75},
                    {"label": "3x3x3", "matrix": [3, 3, 3], "kpoints_distance": 0.05,
                     "host_natoms": 27, "defect_natoms": 26,
                     "host_energy_ev": -371.0, "defect_energy_ev": -360.0,
                     "formation_energy_ev": -2.85},
                ]
            }
        )
        assert "| k-mesh |" in table
        # integer matrices / k-meshes render without decimals
        assert "[7, 7, 7]" in table
        assert "0.050000 Å⁻¹" in table


class TestRuntimeColumns:
    def test_cells_table_has_time_and_steps(self):
        """The per-cell table reports wall time + SCF steps per leg."""
        from aiida_uranium_workflow.utils.report.defects import generate_cells_table

        table = generate_cells_table(
            {
                "cells": [
                    {
                        "label": "5x5x5",
                        "matrix": [5, 5, 5],
                        "kpoints_mesh": [1, 1, 1],
                        "host_natoms": 250,
                        "defect_natoms": 249,
                        "host_energy_ev": -3383177.0,
                        "defect_energy_ev": -3369642.0,
                        "formation_energy_ev": 2.077,
                        "time_host_s": 745.0,
                        "time_defect_s": 759.0,
                        "scf_steps_host": 11,
                        "scf_steps_defect": 60,
                    },
                    {  # legacy cell without the new fields → dashes, no crash
                        "label": "2x2x2",
                        "matrix": [2, 2, 2],
                        "host_natoms": 16,
                        "defect_natoms": 15,
                        "host_energy_ev": -1.0,
                        "defect_energy_ev": -1.0,
                        "formation_energy_ev": 0.0,
                    },
                ]
            }
        )
        header = table.splitlines()[0]
        for col in ("t_host (s)", "t_defect (s)", "steps host", "steps defect"):
            assert col in header
        assert "| 745 | 759 | 11 | 60 |" in table
        assert table.count("| — | — | — | — |") == 1
