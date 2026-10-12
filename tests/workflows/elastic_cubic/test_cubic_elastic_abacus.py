"""Tests for the ABACUS cubic elastic-constant (energy–strain) workflow.

Covers the textbook distortion matrices (volume conserving), the
D_o/D_m → c′/C44 fits, the (6.34)+(6.35) solve for C11/C12, and the
WorkChain wiring (equilibrium supplied → EOS skipped; 12 distorted-cell
SCF submissions; gather recovers known constants from synthetic
energies).
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import numpy as np
import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.elastic_cubic.abacus import (
    AbacusCubicElasticWorkChain,
    EV_PER_A3_TO_GPA,
    _monoclinic_matrix,
    _orthorhombic_matrix,
    _solve_cubic,
    distort_structure,
    is_cubic,
    scale_to_volume,
)

DEFAULT_DELTAS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]


def _cubic_structure(a=3.431):
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=a, cubic=True))


class TestDistortions:
    def test_matrices_volume_conserving(self):
        for delta in (0.01, 0.03, 0.05):
            assert np.isclose(np.linalg.det(_orthorhombic_matrix(delta)), 1.0, atol=1e-6)
            assert np.isclose(np.linalg.det(_monoclinic_matrix(delta)), 1.0, atol=1e-6)

    def test_orthorhombic_shape(self):
        matrix = _orthorhombic_matrix(0.02)
        assert np.allclose(
            matrix,
            np.diag([1.02, 0.98, 1.0 / (1.0 - 0.02**2)]),
        )

    def test_monoclinic_shape(self):
        matrix = _monoclinic_matrix(0.02)
        assert np.isclose(matrix[0, 1], 0.02)
        assert np.isclose(matrix[1, 0], 0.02)
        assert np.isclose(matrix[2, 2], 1.0 / (1.0 - 0.02**2))

    def test_distort_keeps_volume_and_atoms(self):
        structure = _cubic_structure()
        volume0 = structure.get_cell_volume()
        for matrix in (_orthorhombic_matrix(0.03), _monoclinic_matrix(0.03)):
            deformed = distort_structure(structure, matrix)
            assert len(deformed.sites) == len(structure.sites)
            assert np.isclose(
                deformed.get_cell_volume(), volume0, rtol=1e-6
            )


class TestCubicCheck:
    def test_accepts_cubic(self):
        assert is_cubic(_cubic_structure())

    def test_rejects_orthorhombic(self):
        from ase.build import bulk

        atoms = bulk("U", "bcc", a=3.431, cubic=True)
        atoms.cell[0, 0] *= 1.05  # stretch one axis
        assert not is_cubic(orm.StructureData(ase=atoms))


class TestSolve:
    def test_recovers_constants(self, aiida_profile_clean):
        """Synthetic E(δ) = E0 + a2·δ² recovers c′ / C44 / C11 / C12."""
        bulk_gpa = 138.0
        c_prime = 40.0
        c44 = 30.0
        volume = 40.0
        energy0 = -27000.0
        # a2 = 2·V·(constant in eV/Å³)
        a2_o = 2.0 * volume * (c_prime / EV_PER_A3_TO_GPA)
        a2_m = 2.0 * volume * (c44 / EV_PER_A3_TO_GPA)

        energies_o = [energy0 + a2_o * d * d for d in DEFAULT_DELTAS]
        energies_m = [energy0 + a2_m * d * d for d in DEFAULT_DELTAS]

        out = _solve_cubic(
            deltas_o=orm.List(list=DEFAULT_DELTAS),
            energies_o=orm.List(list=energies_o),
            deltas_m=orm.List(list=DEFAULT_DELTAS),
            energies_m=orm.List(list=energies_m),
            bulk_modulus=orm.Float(bulk_gpa),
            volume_gs=orm.Float(volume),
        )
        result = out.get_dict()
        constants = result["constants_gpa"]
        assert constants["c11"] == pytest.approx(
            bulk_gpa + 4.0 / 3.0 * c_prime, rel=1e-6
        )
        assert constants["c12"] == pytest.approx(
            bulk_gpa - 2.0 / 3.0 * c_prime, rel=1e-6
        )
        assert constants["c44"] == pytest.approx(c44, rel=1e-6)
        assert result["method"] == "energy"
        # minimal layout: no input echoes, no derived c′/tensor keys
        for key in (
            "c11_gpa", "c_prime_gpa", "bulk_modulus_gpa",
            "volume_gs", "energy_gs_ev", "elastic_tensor_gpa",
            "deltas_o", "n_points_o", "relax_internal", "eos_used",
        ):
            assert key not in result, key
        fits = result["fits"]
        assert fits["orthorhombic"]["n_points"] == 6
        assert fits["monoclinic"]["n_points"] == 6
        assert fits["orthorhombic"]["a2_ev"] == pytest.approx(a2_o, rel=1e-6)
        assert fits["monoclinic"]["a2_ev"] == pytest.approx(a2_m, rel=1e-6)
        assert fits["orthorhombic"]["rms_ev"] < 1e-9  # exact quadratic data

    def test_quartic_term_does_not_bias_a2(self, aiida_profile_clean):
        """E(δ) = E₀ + a₂δ² + a₄δ⁴ : the δ² coefficient is recovered even
        with a real anharmonic δ⁴ term (a pure a₂δ² fit would be biased)."""
        bulk_gpa = 100.0
        c_prime = 52.0
        c44 = 110.0
        volume = 164.5
        a2_o = 2.0 * volume * (c_prime / EV_PER_A3_TO_GPA)
        a2_m = 2.0 * volume * (c44 / EV_PER_A3_TO_GPA)
        a4 = 2.0e4  # deliberately strong anharmonicity: makes the δ²-only bias visible
        energy0 = -856.5

        deltas = DEFAULT_DELTAS
        e_o = [energy0 + a2_o * d * d + a4 * d**4 for d in deltas]
        e_m = [energy0 + a2_m * d * d + a4 * d**4 for d in deltas]

        out = _solve_cubic(
            deltas_o=orm.List(list=deltas),
            energies_o=orm.List(list=e_o),
            deltas_m=orm.List(list=deltas),
            energies_m=orm.List(list=e_m),
            bulk_modulus=orm.Float(bulk_gpa),
            volume_gs=orm.Float(volume),
        )
        result = out.get_dict()
        assert result["fits"]["orthorhombic"]["a2_ev"] == pytest.approx(a2_o, rel=1e-9)
        assert result["fits"]["monoclinic"]["a2_ev"] == pytest.approx(a2_m, rel=1e-9)
        assert result["constants_gpa"]["c44"] == pytest.approx(c44, rel=1e-9)

        # ... whereas a δ²-only fit absorbs the a₄δ⁴ term and is biased
        # (here a4 > 0 → biased high; a negative a₄ biases it low, as in
        # the real Si data).
        a2_only = np.linalg.lstsq(
            np.vstack([np.ones(len(deltas)), np.asarray(deltas) ** 2]).T,
            np.asarray(e_m),
            rcond=None,
        )[0][1]
        assert abs(a2_only - a2_m) / a2_m > 0.005

    def test_recovers_constants_with_dropped_point(self, aiida_profile_clean):
        """A non-converged distortion point can be dropped: the fit (with a
        free intercept) still recovers c′ / C44 from the survivors."""
        bulk_gpa = 138.0
        c_prime = 40.0
        c44 = 30.0
        volume = 40.0
        energy0 = -27000.0
        a2_o = 2.0 * volume * (c_prime / EV_PER_A3_TO_GPA)
        a2_m = 2.0 * volume * (c44 / EV_PER_A3_TO_GPA)

        deltas = DEFAULT_DELTAS
        e_o = [energy0 + a2_o * d * d for d in deltas]
        e_m = [energy0 + a2_m * d * d for d in deltas]
        # drop δ = 0.02 of D_m (the failing child)
        deltas_m = [d for d in deltas if d != 0.02]
        e_m = [e for d, e in zip(deltas, e_m) if d != 0.02]

        out = _solve_cubic(
            deltas_o=orm.List(list=deltas),
            energies_o=orm.List(list=e_o),
            deltas_m=orm.List(list=deltas_m),
            energies_m=orm.List(list=e_m),
            bulk_modulus=orm.Float(bulk_gpa),
            volume_gs=orm.Float(volume),
        )
        result = out.get_dict()
        assert result["fits"]["monoclinic"]["n_points"] == 5
        assert result["constants_gpa"]["c44"] == pytest.approx(c44, rel=1e-6)
        assert result["constants_gpa"]["c11"] == pytest.approx(
            bulk_gpa + 4.0 / 3.0 * c_prime, rel=1e-6
        )


class TestVolumeScaling:
    def test_scale_to_volume_preserves_fractional(self):
        structure = _cubic_structure()
        frac0 = np.asarray(structure.get_ase().get_scaled_positions())
        target = 2.0 * structure.get_cell_volume()

        scaled, factor = scale_to_volume(structure, target)
        assert scaled.get_cell_volume() == pytest.approx(target, rel=1e-9)
        assert factor == pytest.approx((target / structure.get_cell_volume()) ** (1 / 3))
        frac1 = np.asarray(scaled.get_ase().get_scaled_positions())
        np.testing.assert_allclose(frac1, frac0, atol=1e-12)
        # lattice vectors scaled by the same factor (isotropic)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(scaled.cell), axis=1)
            / np.linalg.norm(np.asarray(structure.cell), axis=1),
            factor,
            atol=1e-9,
        )

    def test_scale_near_one_returns_same_node(self):
        structure = _cubic_structure()
        out, factor = scale_to_volume(structure, structure.get_cell_volume())
        assert out is structure
        assert factor == 1.0


class TestReport:
    def _params(self, **overrides):
        """Minimal (current) output_parameters layout."""
        params = {
            "workflow": "elastic_cubic",
            "backend": "abacus",
            "method": "energy",
            "constants_gpa": {"c11": 191.3, "c12": 111.3, "c44": 30.0},
            "fits": {
                "orthorhombic": {
                    "a2_ev": 20.0,
                    "deltas": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    "energies_ev": [-27000.0 + 20.0 * d * d for d in
                                    [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]],
                    "n_points": 6,
                    "rms_ev": 1e-9,
                },
                "monoclinic": {
                    "a2_ev": 15.0,
                    "deltas": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    "energies_ev": [-27000.0 + 15.0 * d * d for d in
                                    [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]],
                    "n_points": 6,
                    "rms_ev": 1e-9,
                },
            },
        }
        params.update(overrides)
        return params

    def _legacy_params(self):
        """Earliest stored layout: one shared ``deltas`` list."""
        return {
            "workflow": "elastic_cubic",
            "backend": "abacus",
            "method": "energy",
            "crystal_system": "cubic",
            "eos_used": False,
            "bulk_modulus_gpa": 138.0,
            "volume_gs": 40.4,
            "energy_gs_ev": -27000.0,
            "c11_gpa": 191.3,
            "c12_gpa": 111.3,
            "c44_gpa": 30.0,
            "a2_orthorhombic_ev": 20.0,
            "a2_monoclinic_ev": 15.0,
            "deltas": [0.0, 0.01, 0.02],
            "energies_orthorhombic_ev": [-27000.0, -27000.1, -27000.4],
            "energies_monoclinic_ev": [-27000.0, -27000.1, -27000.3],
        }

    def _split_deltas_params(self):
        """Intermediate layout: ``deltas_o`` / ``deltas_m``."""
        params = self._legacy_params()
        params.pop("deltas")
        params["deltas_o"] = [0.0, 0.01, 0.02]
        params["deltas_m"] = [0.0, 0.01, 0.02]
        params["n_points_o"] = 3
        params["n_points_m"] = 3
        return params

    def test_summary_table(self):
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_summary_table,
            _normalise,
        )

        table = generate_summary_table(_normalise(self._params()))
        assert "crystal system | cubic" in table
        assert "C11 | 191.3000 GPa" in table
        assert "C44 | 30.0000 GPa" in table
        assert "c′ = (C11−C12)/2 | 40.0000 GPa" in table

    def test_summary_reads_equilibrium_from_inputs(self):
        """B / V0 / E0 and the 'EOS fallback' flag come from the inputs."""
        from types import SimpleNamespace

        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            _normalise,
            generate_summary_table,
        )

        node = SimpleNamespace(
            inputs=SimpleNamespace(
                bulk_modulus=orm.Float(138.0),
                volume_gs=orm.Float(40.4),
                energy_gs_ev=orm.Float(-27000.0),
                relax_internal=orm.Bool(True),
                crystal_system=orm.Str("cubic"),
            )
        )
        table = generate_summary_table(_normalise(self._params(), node))
        assert "equilibrium data | supplied input" in table
        assert "bulk modulus B | 138.0000 GPa" in table
        assert "internal relaxation | on (relaxed-ion)" in table

        eos_node = SimpleNamespace(inputs=SimpleNamespace())
        table = generate_summary_table(_normalise(self._params(), eos_node))
        assert "equilibrium data | EOS fallback" in table
        assert "bulk modulus B | — GPa" in table

    def test_generate_report(self):
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_report,
        )

        report = generate_report(self._params(), 42, "abacus")
        assert "Cubic Elastic Constants Report" in report
        assert "(PK: 42)" in report
        assert "## Summary" in report
        assert "## Elastic Tensor (Voigt, GPa)" in report
        assert "## D_o / D_m distortion scans" in report
        # stable cubic: all Born criteria OK
        assert "## Cubic stability (Born criteria)" in report
        assert "* OK — C11 + 2·C12 > 0" in report
        assert "* OK — C11 − C12 > 0 (c′ > 0)" in report
        assert "* OK — C44 > 0" in report
        assert "*Generated by aiida-uranium-workflow*" in report

    def test_generate_report_with_figure(self, tmp_path):
        """figure_dir renders the ΔE(δ) fits figure and embeds it."""
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_report,
        )

        report = generate_report(self._params(), 43, "abacus", figure_dir=tmp_path)
        png = tmp_path / "elastic_cubic_delta_scan_pk43.png"
        assert png.is_file() and png.stat().st_size > 0
        assert "## Quadratic fits (ΔE vs δ)" in report
        assert "elastic_cubic_delta_scan_pk43.png" in report

    def test_legacy_output_layout_still_renders(self, tmp_path):
        """Earliest layout (shared ``deltas``) renders AND produces the figure."""
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_report,
        )

        report = generate_report(
            self._legacy_params(), 44, "abacus", figure_dir=tmp_path
        )
        assert "C11 | 191.3000 GPa" in report
        assert "equilibrium data | supplied input" in report
        assert "## D_o / D_m distortion scans" in report
        # shared-deltas layout must still yield the δ-scan figure
        assert "## Quadratic fits (ΔE vs δ)" in report
        png = tmp_path / "elastic_cubic_delta_scan_pk44.png"
        assert png.is_file() and png.stat().st_size > 0

    def test_split_deltas_layout_still_renders(self):
        """Intermediate layout (``deltas_o`` / ``deltas_m``) keeps working."""
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_report,
        )

        report = generate_report(self._split_deltas_params(), 45, "abacus")
        assert "C11 | 191.3000 GPa" in report
        assert "D_o fit: a₂ / points / rms" in report

    def test_generate_report_flags_instability(self):
        from aiida_uranium_workflow.utils.report.elastic_cubic import (
            generate_report,
        )

        # γ-U-like: negative c′ (C11 < C12).
        params = self._params(constants_gpa={"c11": -23.5, "c12": 218.9, "c44": 30.0})
        report = generate_report(params, 7, "abacus")
        assert "* FAILED — C11 − C12 > 0 (c′ > 0)" in report
        assert "* FAILED — C11 + 2·C12 > 0" not in report  # still positive
        assert "C11 | -23.5000 GPa" in report


class TestAdapterAndProtocol:
    def test_adapter_forwards_equilibrium(self, aiida_profile, aiida_localhost):
        from aiida_uranium_workflow.input_builders.elastic_cubic.abacus import (
            AbacusCubicElasticAdapter,
        )

        orm.InstalledCode(
            label="cubic_el_adapter",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusCubicElasticAdapter(
            code_label=f"cubic_el_adapter@{aiida_localhost.label}",
            software_params={
                "parameters": {"input": {"basis_type": "lcao"}},
                "kpoints_mesh": [7, 7, 7],
            },
            metadata={},
            workflow_data={
                "elastic_cubic": {
                    "crystal_system": "cubic",
                    "equilibrium": {
                        "bulk_modulus": 138.0,
                        "volume_gs": 40.4,
                        "energy_gs_ev": -27987.4,
                    },
                }
            },
            extra_codes={},
        )
        inputs = adapter._build_workchain_inputs(_cubic_structure())
        assert inputs["crystal_system"].value == "cubic"
        assert inputs["bulk_modulus"].value == 138.0
        assert inputs["volume_gs"].value == 40.4
        assert inputs["energy_gs_ev"].value == -27987.4
        assert inputs["base"]["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]

    def test_adapter_without_equilibrium_omits(self, aiida_profile, aiida_localhost):
        from aiida_uranium_workflow.input_builders.elastic_cubic.abacus import (
            AbacusCubicElasticAdapter,
        )

        orm.InstalledCode(
            label="cubic_el_adapter2",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusCubicElasticAdapter(
            code_label=f"cubic_el_adapter2@{aiida_localhost.label}",
            software_params={"parameters": {"input": {"calculation": "scf"}}},
            metadata={},
            workflow_data={"elastic_cubic": {"crystal_system": "cubic"}},
            extra_codes={},
        )
        inputs = adapter._build_workchain_inputs(_cubic_structure())
        # No equilibrium -> the WorkChain runs the EOS fallback.
        assert "bulk_modulus" not in inputs
        assert "volume_gs" not in inputs

    def test_protocol_parser_forwards_preset(self):
        from aiida_uranium_workflow.schedulers.elastic_cubic import (
            parse_elastic_cubic_protocol,
        )

        preset = {
            "crystal_system": "cubic",
            "equilibrium": {"bulk_modulus": 138.0},
        }
        assert parse_elastic_cubic_protocol(preset) == {"elastic_cubic": preset}
        assert parse_elastic_cubic_protocol(None) == {"elastic_cubic": {}}


class TestWorkflow:
    def _instantiate(
        self,
        aiida_profile_clean,
        aiida_localhost,
        *,
        with_equilibrium=True,
        crystal_system="cubic",
        volume_gs=40.0,
        relax_internal=None,
    ):
        orm.InstalledCode(
            label="cubic_el_abacus",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        inputs = {
            "structure": _cubic_structure(),
            "base": {
                "abacus": {
                    "code": orm.load_code("cubic_el_abacus"),
                    "parameters": orm.Dict(dict={"input": {"calculation": "scf"}}),
                }
            },
            "crystal_system": orm.Str(crystal_system),
        }
        if relax_internal is not None:
            inputs["relax_internal"] = orm.Bool(relax_internal)
        if with_equilibrium:
            inputs.update(
                {
                    "bulk_modulus": orm.Float(138.0),
                    "volume_gs": orm.Float(volume_gs),
                    "energy_gs_ev": orm.Float(-27000.0),
                }
            )
        runner = get_manager().get_runner()
        return instantiate_process(runner, AbacusCubicElasticWorkChain, **inputs)

    def test_unsupported_crystal_system(self, aiida_profile_clean, aiida_localhost):
        process = self._instantiate(
            aiida_profile_clean, aiida_localhost, crystal_system="hexagonal"
        )
        assert (
            process.prepare()
            == AbacusCubicElasticWorkChain.exit_codes.ERROR_UNSUPPORTED_CRYSTAL_SYSTEM
        )

    def test_equilibrium_supplied_skips_eos(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = self._instantiate(aiida_profile_clean, aiida_localhost)
        assert process.prepare() is None
        captured = []

        def fake_submit(cls, **inputs):
            captured.append((cls, inputs))
            return SimpleNamespace(pk=1, process_label="child")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.run_eos()
        assert captured == []  # no EOS child
        assert process.ctx.eos_used is False

    def test_volume_gs_rescales_input_cell(
        self, aiida_profile_clean, aiida_localhost
    ):
        """volume_gs far from the input volume → the cell is isotropically
        rescaled to V0 while the fractional coordinates stay put."""
        process = self._instantiate(
            aiida_profile_clean, aiida_localhost, volume_gs=80.0
        )
        input_frac = np.asarray(
            process.inputs.structure.get_ase().get_scaled_positions()
        )
        assert process.prepare() is None
        process.run_eos()
        assert process.ctx.work_structure is not process.inputs.structure
        assert process.ctx.work_structure.get_cell_volume() == pytest.approx(
            80.0, rel=1e-9
        )
        frac1 = np.asarray(process.ctx.work_structure.get_ase().get_scaled_positions())
        np.testing.assert_allclose(frac1, input_frac, atol=1e-12)
        assert process.ctx.volume_scale == pytest.approx(
            (80.0 / process.inputs.structure.get_cell_volume()) ** (1 / 3)
        )

    def test_full_flow_recovers_constants(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """prepare → run_eos (supplied) → run_distortions → gather."""
        process = self._instantiate(aiida_profile_clean, aiida_localhost)
        assert process.prepare() is None

        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label="AbacusBaseWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.run_eos()
        process.run_distortions()

        # 2 distortions x 6 δ values (default relax_internal=True → the
        # children are abacus.relax position-relax runs).
        assert len(captured) == 12
        assert all("relax_settings" in child for child in captured)
        assert process.ctx.equilibrium["bulk_modulus"] == 138.0
        assert process.ctx.equilibrium["volume_gs"] == 40.0

        # Synthetic energies for the submitted distorted cells.
        c_prime = 40.0
        c44 = 30.0
        volume = 40.0
        energy0 = -27000.0
        a2_o = 2.0 * volume * (c_prime / EV_PER_A3_TO_GPA)
        a2_m = 2.0 * volume * (c44 / EV_PER_A3_TO_GPA)
        by_kind = {
            f"scf_o_{d:.2f}".replace(".", "_"): energy0 + a2_o * d * d
            for d in DEFAULT_DELTAS
        }
        by_kind.update(
            {
                f"scf_m_{d:.2f}".replace(".", "_"): energy0 + a2_m * d * d
                for d in DEFAULT_DELTAS
            }
        )
        for key, value in by_kind.items():
            process.ctx[key] = SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    misc=SimpleNamespace(get_dict=lambda v=value: {"total_energy": v})
                ),
            )

        assert process.gather() is None
        para = process._outputs["output_parameters"].get_dict()  # noqa: SLF001
        constants = para["constants_gpa"]
        assert constants["c11"] == pytest.approx(
            138.0 + 4.0 / 3.0 * c_prime, rel=1e-6
        )
        assert constants["c12"] == pytest.approx(
            138.0 - 2.0 / 3.0 * c_prime, rel=1e-6
        )
        assert constants["c44"] == pytest.approx(c44, rel=1e-6)
        assert para["backend"] == "abacus"

    def test_relax_internal_false_uses_scf_children(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = self._instantiate(
            aiida_profile_clean, aiida_localhost, relax_internal=False
        )
        assert process.prepare() is None
        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label="AbacusBaseWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.run_eos()
        process.run_distortions()
        assert len(captured) == 12
        # clamped-ion path: fixed-lattice SCF children (abacus namespace)
        assert all("relax_settings" not in child for child in captured)
        assert all("abacus" in child and "structure" not in child for child in captured)

    def test_gather_skips_failed_distortion(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """One non-converged distortion point is skipped, not fatal."""
        process = self._instantiate(aiida_profile_clean, aiida_localhost)
        assert process.prepare() is None
        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label="child")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.run_eos()
        process.run_distortions()

        c_prime = 40.0
        c44 = 30.0
        volume = 40.0
        energy0 = -27000.0
        a2_o = 2.0 * volume * (c_prime / EV_PER_A3_TO_GPA)
        a2_m = 2.0 * volume * (c44 / EV_PER_A3_TO_GPA)
        for kind, a2 in (("o", a2_o), ("m", a2_m)):
            for delta in DEFAULT_DELTAS:
                key = f"scf_{kind}_{delta:.2f}".replace(".", "_")
                value = energy0 + a2 * delta * delta
                process.ctx[key] = SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(
                        misc=SimpleNamespace(
                            get_dict=lambda v=value: {"total_energy": v}
                        )
                    ),
                )
        # δ = 0.02 of D_m failed to converge.
        process.ctx.scf_m_0_02 = SimpleNamespace(is_finished_ok=False, pk=99)

        assert process.gather() is None
        para = process._outputs["output_parameters"].get_dict()  # noqa: SLF001
        assert para["fits"]["monoclinic"]["n_points"] == 5
        assert para["fits"]["orthorhombic"]["n_points"] == 6
        assert para["constants_gpa"]["c44"] == pytest.approx(c44, rel=1e-6)
        assert para["constants_gpa"]["c11"] == pytest.approx(
            138.0 + 4.0 / 3.0 * c_prime, rel=1e-6
        )

    def test_gather_too_few_points_fails(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = self._instantiate(aiida_profile_clean, aiida_localhost)
        assert process.prepare() is None
        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label="child")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.run_eos()
        process.run_distortions()
        # only three D_m points survive → below the minimum
        for delta in DEFAULT_DELTAS:
            process.ctx[f"scf_o_{delta:.2f}".replace(".", "_")] = SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    misc=SimpleNamespace(get_dict=lambda: {"total_energy": -1.0})
                ),
            )
        for delta in DEFAULT_DELTAS[:3]:
            process.ctx[f"scf_m_{delta:.2f}".replace(".", "_")] = SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    misc=SimpleNamespace(get_dict=lambda: {"total_energy": -1.0})
                ),
            )
        assert (
            process.gather() == AbacusCubicElasticWorkChain.exit_codes.ERROR_CHILD
        )
