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
