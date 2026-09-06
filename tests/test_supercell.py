"""Tests for the supercell workflow (protocol parsing, adapters, report,
supercell structure generation)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import numpy as np
import pytest

from aiida_uranium_workflow.input_builders.supercell.abacus import AbacusSupercellAdapter
from aiida_uranium_workflow.schedulers.supercell import parse_supercell_protocol
from aiida_uranium_workflow.utils.report.supercell import generate_report
from aiida_uranium_workflow.workflows.supercell.abacus import (
    SupercellScfWorkChain,
    make_supercell_structure,
)


class TestProtocolParser:
    def test_splits_backends(self):
        proto = {"abacus": {"supercells": [{"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}}
        out = parse_supercell_protocol(proto)
        assert out == proto

    def test_empty(self):
        assert parse_supercell_protocol(None) == {}
        assert parse_supercell_protocol({}) == {}


class TestMakeSupercell:
    def test_identity_keeps_cell(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert len(sc.sites) == len(cell.sites)
        assert sc.get_cell_volume() == pytest.approx(cell.get_cell_volume(), rel=1e-6)

    def test_2x2x2_octuples_volume(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        assert len(sc.sites) == 8 * len(cell.sites)
        assert sc.get_cell_volume() == pytest.approx(8 * cell.get_cell_volume(), rel=1e-6)

    def test_coordinates_cartesian_not_fractional(self, aiida_profile):
        """Regression: pymatgen_to_structure must write CARTESIAN site
        coordinates — fractional coords piled atoms into the cell corner
        and ABACUS aborted with "atoms too close"."""
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        positions = np.array([s.position for s in sc.sites])
        # The bcc primitive basis has negative components, but the sites
        # must still span the whole 2× supercell (≈6.9 Å), not be piled
        # into the cell corner (~0–1 Å).
        span = positions.max() - positions.min()
        assert span.max() > 3.0
        # Minimum periodic nearest-neighbour distance ≈ bcc NN (a√3/2).
        frac = positions @ np.linalg.inv(np.asarray(sc.cell))
        dmin = 1e9
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                d = frac[j] - frac[i]
                d -= np.round(d)
                dmin = min(dmin, np.linalg.norm(d @ np.asarray(sc.cell)))
        assert dmin == pytest.approx(3.45 * np.sqrt(3) / 2, rel=0.05)

    def test_non_diagonal_matrix(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 1, 0], [1, 2, 0], [0, 0, 2]])        # det([[2,1,0],[1,2,0],[0,0,2]]) = 6 primitive cells.
        assert sc.get_cell_volume() == pytest.approx(6 * cell.get_cell_volume(), rel=1e-6)


class TestAdapter:
    def test_abacus_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="supercell_test_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusSupercellAdapter(
            code_label=f"supercell_test_abacus@{computer.label}",
            software_params={
                "parameters": {"input": {"nspin": 1}},
                "kpoints_mesh": [11, 11, 11],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "abacus": {
                    "supercells": [
                        {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "label": "1x1x1", "kpoints_mesh": [11, 11, 11]},
                        {"matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                         "label": "2x2x2", "kpoints_mesh": [7, 7, 7],
                         "scf_thr": 1e-6, "mixing_beta": 0.4},
                    ],
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"] == "scf"
        cells = inputs["supercell_parameters"].get_dict()["supercells"]
        assert len(cells) == 2
        assert cells[1]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert cells[1]["scf_thr"] == 1e-6
        err = SupercellScfWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"

    def test_missing_supercells_raises(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="supercell_test_empty",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusSupercellAdapter(
            code_label=f"supercell_test_empty@{computer.label}",
            software_params={"parameters": {"input": {"nspin": 1}}},
            metadata={},
            workflow_data={"abacus": {}},
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        with pytest.raises(ValueError):
            adapter._build_workchain_inputs(structure)


class TestReport:
    def test_table_renders_rows(self):
        out = {
            "workflow": "supercell",
            "cells": [
                {"label": "1x1x1", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "natoms": 2, "volume": 41.086, "energy": -27065.73,
                 "time_s": 120.5, "scf_steps": 21, "scf_pk": 1},
                {"label": "2x2x2", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                 "natoms": 16, "volume": 328.69, "energy": -27065.73 * 8,
                 "time_s": 3600.0, "scf_steps": 35, "scf_pk": 2},
            ],
        }
        report = generate_report(out, pk=7, workflow_type="abacus")
        assert "Supercell SCF WorkChain Report" in report
        assert "| 1x1x1 |" in report
        assert "| 2x2x2 |" in report
        assert "41.086" in report
        # time + scf steps columns present; scf pk column dropped.
        assert "| time (s) | scf steps |" in report
        assert "| 120.500 | 21 |" in report
        assert "| 3600.000 | 35 |" in report
        assert "scf pk" not in report

    def test_time_and_steps_default_to_dash(self):
        """Rows without time / scf steps render dashes (legacy data)."""
        out = {
            "workflow": "supercell",
            "cells": [
                {"label": "1x1x1", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "natoms": 2, "volume": 41.086, "energy": -27065.73},
            ],
        }
        report = generate_report(out, pk=7, workflow_type="abacus")
        assert "| — | — |" in report

    def test_empty(self):
        report = generate_report({"cells": []}, pk=1, workflow_type="abacus")
        assert "No supercell data available." in report
