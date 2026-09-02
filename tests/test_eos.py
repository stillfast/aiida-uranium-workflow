"""Tests for the eos workflow (protocol parsing, adapters, fit, report)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.eos.abacus import AbacusEosAdapter
from aiida_uranium_workflow.input_builders.eos.fleur import FleurEosAdapter
from aiida_uranium_workflow.schedulers.eos import parse_eos_protocol
from aiida_uranium_workflow.utils.plot.eos import (
    birch_murnaghan_energy,
    render_eos_figure,
)
from aiida_uranium_workflow.utils.report.eos import (
    generate_eos_table,
    generate_report,
    generate_summary_table,
)


class TestProtocolParser:
    def test_splits_backends(self):
        proto = {"abacus": {"points": 7}, "fleur": {"step": 0.01}}
        out = parse_eos_protocol(proto)
        assert out == {"abacus": {"points": 7}, "fleur": {"step": 0.01}}

    def test_empty(self):
        assert parse_eos_protocol(None) == {}
        assert parse_eos_protocol({}) == {}


class TestReport:
    def test_summary_table(self):
        data = {
            "workflow": "eos",
            "backend": "abacus",
            "fit": "birchmurnaghan",
            "volume_gs": 41.45,
            "lattice_constant_gs": 3.4609,
            "bulk_modulus": 138.3,
            "bulk_deriv": 4.2,
            "energy_gs_ev": -27065.7,
            "scales": [0.98, 1.0, 1.02],
            "volumes": [40.0, 41.45, 43.0],
            "energies_ev": [-27065.5, -27065.7, -27065.4],
            "n_points": 3,
        }
        table = generate_summary_table(data)
        assert "volume_gs | 41.4500 Å³" in table
        assert "bulk_modulus | 138.3000 GPa" in table
        assert "lattice_constant_gs | 3.4609 Å" in table

    def test_generate_report_no_node(self):
        md = generate_report({"volume_gs": 41.0}, 7, "abacus")
        assert "Equation of State Report" in md
        assert "## EOS scan" in md

    def test_summary_table_fleur_keys(self):
        """FLEUR plugin keys: energy_gs derived from the scan minimum,
        scaling_gs row present."""
        table = generate_summary_table(
            {
                "fit": "birchmurnaghan",
                "volume_gs": 41.45,
                "scaling_gs": 0.9967,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
                "scaling": [0.98, 1.0, 1.02],
                "volumes": [40.0, 41.45, 43.0],
                "total_energy": [-27065.5, -27065.7, -27065.4],
                "total_energy_units": "eV",
            }
        )
        # energy_gs = min(total_energy)
        assert "energy_gs | -27065.7000 eV" in table
        assert "scaling_gs | 0.9967" in table

    def test_eos_table_fleur_keys(self):
        """The scan table must render an energy column from FLEUR's
        ``total_energy`` key."""
        table = generate_eos_table(
            {
                "scaling": [0.98, 1.0, 1.02],
                "volumes": [40.0, 41.45, 43.0],
                "total_energy": [-27065.5, -27065.7, -27065.4],
            }
        )
        assert "| scale | volume (Å³) | energy (eV) |" in table
        assert "| 0.9800 | 40.0000 | -27065.500000 |" in table
        assert "| 1.0000 | 41.4500 | -27065.700000 |" in table

    def test_generate_report_fleur_node(self):
        """Lattice constant derived from the GS structure when the fit
        dict (FLEUR plugin) does not carry it."""
        from types import SimpleNamespace

        import numpy as np

        para = SimpleNamespace(
            get_dict=lambda: {
                "volume_gs": 41.45,
                "scaling_gs": 0.9967,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
                "scaling": [0.98, 1.0, 1.02],
                "volumes": [40.0, 41.45, 43.0],
                "total_energy": [-27065.5, -27065.7, -27065.4],
            }
        )
        gs_structure = SimpleNamespace(cell=np.eye(3) * 3.4609)

        class _Outputs(dict):
            def __getattr__(self, name):
                if name in self:
                    return self[name]
                raise AttributeError(name)

        node = SimpleNamespace(
            outputs=_Outputs(
                output_eos_wc_para=para, output_eos_wc_structure=gs_structure
            )
        )
        md = generate_report({}, 7, "fleur", workchain_node=node)
        assert "lattice_constant_gs | 3.4609 Å" in md
        assert "energy_gs | -27065.7000 eV" in md
        assert "| 1.0000 | 41.4500 | -27065.700000 |" in md

    def test_summary_table_per_atom(self):
        """Per-atom ground-state quantities: explicit keys (ABACUS) or
        derived from the cell totals (FLEUR)."""
        # ABACUS output carries the per-atom keys.
        table = generate_summary_table(
            {
                "natoms": 2,
                "volume_gs": 86.0,
                "volume_gs_per_atom": 43.0,
                "energy_gs_ev": -27065.7,
                "energy_gs_per_atom_ev": -13532.85,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
            }
        )
        assert "volume_gs | 86.0000 Å³" in table
        assert "volume_gs_per_atom | 43.0000 Å³/atom" in table
        assert "energy_gs_per_atom | -13532.8500 eV/atom" in table

        # FLEUR output: per-atom values are derived as total / natoms.
        table = generate_summary_table(
            {
                "natoms": 2,
                "volume_gs": 86.0,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
                "scaling": [1.0],
                "volumes": [86.0],
                "total_energy": [-27065.7],
            }
        )
        assert "volume_gs_per_atom | 43.0000 Å³/atom" in table
        assert "energy_gs_per_atom | -13532.8500 eV/atom" in table
        assert "natoms | 2" in table

    def test_eos_table_per_atom_columns(self):
        """With natoms known the scan table gains per-atom columns."""
        table = generate_eos_table(
            {
                "natoms": 2,
                "scaling": [0.98, 1.0],
                "volumes": [40.0, 41.45],
                "total_energy": [-27065.5, -27065.7],
            }
        )
        assert (
            "| scale | volume (Å³) | volume/atom (Å³) | energy (eV) | "
            "energy/atom (eV) |" in table
        )
        assert "| 0.9800 | 40.0000 | 20.0000 | -27065.500000 | -13532.750000 |" in table
        assert "| 1.0000 | 41.4500 | 20.7250 | -27065.700000 | -13532.850000 |" in table

    def test_eos_table_three_columns_without_natoms(self):
        """Without natoms the scan table stays at three columns."""
        table = generate_eos_table(
            {
                "scaling": [1.0],
                "volumes": [41.45],
                "total_energy": [-27065.7],
            }
        )
        assert "| scale | volume (Å³) | energy (eV) |" in table
        assert "volume/atom" not in table


class TestEosFigure:
    """EOS curve figure: BM evaluation + PNG rendering + report embedding."""

    def test_birch_murnaghan_energy_at_v0(self):
        """The BM function returns E0 at V0 and has its minimum near V0."""
        v0, e0, b_gpa, b_deriv = 45.0, -27065.7, 138.3, 4.2
        e_at_v0 = birch_murnaghan_energy([v0], v0, e0, b_gpa, b_deriv)
        assert abs(e_at_v0[0] - e0) < 1e-6
        # The minimum of the curve lies within a small neighbourhood of V0.
        v = [v0 * s for s in (0.99, 0.999, 1.0, 1.001, 1.01)]
        e = birch_murnaghan_energy(v, v0, e0, b_gpa, b_deriv)
        assert float(e[v.index(v0)]) <= min(float(x) for x in e) + 1e-6

    def test_render_eos_figure(self, tmp_path):
        out = tmp_path / "eos_curve.png"
        render_eos_figure(
            out,
            volumes=[40.0, 42.0, 44.0, 46.0, 48.0],
            energies=[-27065.5, -27066.0, -27066.2, -27066.0, -27065.5],
            volume_gs=44.0,
            energy_gs=-27066.2,
            bulk_modulus_gpa=138.3,
            bulk_deriv=4.2,
            natoms=2,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_eos_figure_no_fit(self, tmp_path):
        """Without fit parameters only the scan points are plotted."""
        out = tmp_path / "eos_curve_points.png"
        render_eos_figure(
            out,
            volumes=[40.0, 42.0, 44.0],
            energies=[-27065.5, -27066.0, -27066.2],
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_generate_report_with_figure(self, tmp_path):
        md = generate_report(
            {
                "natoms": 2,
                "fit": "birchmurnaghan",
                "scaling": [0.98, 1.0, 1.02],
                "volumes": [40.0, 41.45, 43.0],
                "energies_ev": [-27065.5, -27065.7, -27065.4],
                "volume_gs": 41.45,
                "volume_gs_per_atom": 20.725,
                "energy_gs_ev": -27065.7,
                "energy_gs_per_atom_ev": -13532.85,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
            },
            7,
            "abacus",
            figure_dir=tmp_path,
        )
        assert "## EOS curve" in md
        assert "![EOS curve](eos_curve_pk7.png)" in md
        assert (tmp_path / "eos_curve_pk7.png").exists()

    def test_generate_report_figure_named_after_report(self, tmp_path):
        """With ``report_stem`` the figure shares the report's name."""
        md = generate_report(
            {
                "scaling": [1.0],
                "volumes": [41.45],
                "energies_ev": [-27065.7],
                "volume_gs": 41.45,
                "energy_gs_ev": -27065.7,
                "bulk_modulus": 138.3,
                "bulk_deriv": 4.2,
            },
            "e1a9fcac-4841-458d-857d-9967bc79d013",
            "fleur",
            figure_dir=tmp_path,
            report_stem="report_nosoc_e1a9fcac",
        )
        assert "![EOS curve](report_nosoc_e1a9fcac_eos_curve.png)" in md
        assert (tmp_path / "report_nosoc_e1a9fcac_eos_curve.png").exists()

    def test_generate_report_figure_skipped_without_data(self, tmp_path):
        """No scan data → a note instead of a figure (no crash)."""
        md = generate_report({"volume_gs": 41.0}, 7, "abacus", figure_dir=tmp_path)
        assert "no EOS scan data" in md
        assert not list(tmp_path.glob("*.png"))


class TestAdapters:
    def test_abacus_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from aiida_uranium_workflow.workflows.eos.abacus import AbacusEosWorkChain
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="eos_test_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusEosAdapter(
            code_label=f"eos_test_abacus@{computer.label}",
            software_params={
                "parameters": {"input": {"nspin": 1, "scf_thr": 1e-7}},
                "kpoints_distance": 0.1,
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "abacus": {
                    "points": 7,
                    "step": 0.01,
                    "guess": 1.0,
                    "scf_thr": 1e-6,
                    "mixing_beta": 0.4,
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert (
            inputs["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"]
            == "scf"
        )
        # EOS protocol overrides the shared scf.yml threshold locally.
        assert (
            inputs["base"]["abacus"]["parameters"].get_dict()["input"]["scf_thr"]
            == 1e-6
        )
        assert (
            inputs["base"]["abacus"]["parameters"].get_dict()["input"]["mixing_beta"]
            == 0.4
        )
        # Scan settings only carry points / step / guess.
        assert inputs["eos_parameters"].get_dict() == {
            "points": 7,
            "step": 0.01,
            "guess": 1.0,
        }
        err = AbacusEosWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"

    def test_fleur_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from aiida.plugins import WorkflowFactory
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="eos_test_fleur",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="eos_test_inpgen",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = FleurEosAdapter(
            code_label=f"eos_test_fleur@{computer.label}",
            software_params={
                "wf_parameters": {"mode": "density", "itmax_per_run": 100},
                "calc_parameters": {"comp": {"kmax": 7.0}},
            },
            metadata={"options": {"resources": {"num_machines": 1}}},
            workflow_data={"fleur": {"points": 9, "step": 0.005}},
            extra_codes={"inpgen": f"eos_test_inpgen@{computer.label}"},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert sorted(inputs["scf"].keys()) == [
            "calc_parameters",
            "fleur",
            "inpgen",
            "options",
            "wf_parameters",
        ]
        assert inputs["wf_parameters"].get_dict()["points"] == 9
        err = WorkflowFactory("fleur.eos").spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"
