"""Unit tests for the elastic-constants utilities."""

import numpy as np
import pytest

from aiida_uranium_workflow.utils.elastic import (
    COMBINED_STRAIN_MODES,
    fit_elastic_from_energy,
    fit_elastic_from_stress,
    generate_deformations,
)
from aiida_uranium_workflow.utils.report.elastic import generate_summary_table

#: eV per J — used to synthesize energies from a known tensor.
_EV_PER_J = 1.602176634e-19


def _known_cubic():
    return np.array(
        [
            [200, 120, 120, 0, 0, 0],
            [120, 200, 120, 0, 0, 0],
            [120, 120, 200, 0, 0, 0],
            [0, 0, 0, 60, 0, 0],
            [0, 0, 0, 0, 60, 0],
            [0, 0, 0, 0, 0, 60],
        ],
        dtype=float,
    )


def _known_orthorhombic():
    """9 independent constants (α-U style): C11≠C22≠C33, C12≠C13≠C23."""
    return np.array(
        [
            [230, 130, 110, 0, 0, 0],
            [130, 260, 120, 0, 0, 0],
            [110, 120, 240, 0, 0, 0],
            [0, 0, 0, 70, 0, 0],
            [0, 0, 0, 0, 55, 0],
            [0, 0, 0, 0, 0, 80],
        ],
        dtype=float,
    )


def _strain_set(with_combined=True):
    return np.array(
        [
            d[0]
            for d in generate_deformations(
                [-0.010, -0.005, 0.005, 0.010],
                [-0.010, -0.005, 0.005, 0.010],
                COMBINED_STRAIN_MODES if with_combined else None,
            )
        ]
    )


def _synthesize_energies(C_gpa, strains, V, e0=-56165.0):
    """E(ε) = E0 + (V/2)·εᵀ·C·ε with C in GPa, energies in eV."""
    return (
        e0
        + V
        * 1e-30
        / 2
        * np.einsum("ni,ij,nj->n", strains, C_gpa * 1e9, strains)
        / _EV_PER_J
    )


class TestGenerateDeformations:
    def test_count_and_states(self):
        norm = [-0.010, -0.005, 0.005, 0.010]
        shear = [-0.010, -0.005, 0.005, 0.010]
        defs = generate_deformations(norm, shear)
        assert len(defs) == 24  # 3 normal + 3 shear, ×4 magnitudes
        # First normal strain: ε1 = δ = -0.01
        assert defs[0][0][0] == pytest.approx(-0.01)
        # First shear strain: Voigt ε6 = 2δ = -0.02
        assert defs[12][0][5] == pytest.approx(-0.02)

    def test_combined_strains_count_and_values(self):
        norm = [-0.010, -0.005, 0.005, 0.010]
        shear = [-0.010, -0.005, 0.005, 0.010]
        defs = generate_deformations(norm, shear, COMBINED_STRAIN_MODES)
        assert len(defs) == 24 + len(COMBINED_STRAIN_MODES) * len(norm)
        # First combined mode ε7 = (δ, δ, 0, 0, 0, 0) at δ = -0.01.
        combined = [d for d in defs if d[2].startswith("c0")]
        assert combined[0][0][0] == pytest.approx(-0.01)
        assert combined[0][0][1] == pytest.approx(-0.01)
        assert combined[0][0][2] == pytest.approx(0.0)
        # Labels: c0..c4 for the 5 modes.
        labels = sorted({d[2] for d in defs if d[2].startswith("c")})
        assert labels == [f"c{i}" for i in range(len(COMBINED_STRAIN_MODES))]

    def test_default_combined_modes_cover_user_sets(self):
        """The default modes span the orthorhombic biaxial + cubic sets."""
        as_tuples = [tuple(m) for m in COMBINED_STRAIN_MODES]
        # orthorhombic ε7..ε9 and cubic ε1 / ε2 from the method notes.
        for mode in [
            (1, 1, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0),
            (0, 1, 1, 0, 0, 0),
            (1, 1, 1, 0, 0, 0),
            (0, 0, 0, 1, 1, 1),
        ]:
            assert mode in as_tuples


class TestApplyDeformationMatrix:
    """The deformed cell must be ``L · Fᵀ`` (``StructureData.cell`` rows are
    the lattice vectors).

    Regression: ``F · L`` applies the *transposed* strain whenever the cell
    is not isotropic — for the α-U cell (a=2.80, b=5.84, c=4.91 Å) the xy
    shear came out b/a ≈ 2.1× too large, xz c/a ≈ 1.75× too large and yz
    c/b ≈ 0.84× too small, i.e. C66/C55/C44 were wrong by exactly those
    ratios while C11..C23 (diagonal F) were fine. The manual VASP reference
    run (pymatgen ``DeformedStructureSet``) gives C66 ≈ 97 GPa where the
    buggy version gave ≈ 204 GPa.
    """

    CELL = np.array([[2.80273, 0, 0], [0, 5.83665, 0], [0, 0, 4.90696]])
    FRAC = [
        [0.0, 0.09846, 0.25],
        [0.0, 0.90154, 0.75],
        [0.5, 0.59846, 0.25],
        [0.5, 0.40154, 0.75],
    ]

    def _structure(self):
        from aiida import orm

        structure = orm.StructureData(cell=self.CELL, pbc=True)
        for frac in self.FRAC:
            structure.append_atom(position=np.asarray(frac) @ self.CELL, symbols=["U"])
        return structure

    def test_matches_pymatgen_deformed_structure_set(self, aiida_profile):
        """Byte-identical to pymatgen's DeformedStructureSet (the method the
        manual elastic example uses)."""
        from pymatgen.analysis.elasticity.strain import DeformedStructureSet
        from pymatgen.core import Lattice, Structure

        from aiida_uranium_workflow.utils.elastic import apply_deformation_matrix

        structure = self._structure()
        reference = DeformedStructureSet(
            Structure(Lattice(self.CELL), ["U"] * 4, self.FRAC),
            symmetry=False,
            norm_strains=[-0.005, 0.005],
            shear_strains=[-0.005, 0.005],
        )
        defs = generate_deformations([-0.005, 0.005], [-0.005, 0.005])
        for (_, deformation, _), expected in zip(defs, reference.deformed_structures):
            got = np.asarray(apply_deformation_matrix(structure, deformation).cell)
            np.testing.assert_allclose(
                got,
                expected.lattice.matrix,
                atol=1e-8,
                err_msg="deformed cell mismatch",
            )

    def test_shear_angle_is_not_amplified_by_the_cell_shape(self, aiida_profile):
        """ε_xy = 0.005 must give an engineering shear γ = 0.01 (not γ·b/a)."""
        from aiida_uranium_workflow.utils.elastic import apply_deformation_matrix

        structure = self._structure()
        _, deformation, _ = [
            d
            for d in generate_deformations([-0.005, 0.005], [-0.005, 0.005])
            if d[2] == "s(0, 1)"
        ][0]
        cell = np.asarray(apply_deformation_matrix(structure, deformation).cell)
        a, b = cell[0], cell[1]
        gamma = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        # cos(angle) ≈ −γ for the tilted vector; |γ| must be the Voigt ε6.
        assert abs(gamma) == pytest.approx(0.01, abs=1e-9)


class TestFitElasticFromStress:
    def test_recovers_cubic_tensor(self):
        C = _known_cubic()
        strains = np.array(
            [
                d[0]
                for d in generate_deformations(
                    [-0.010, -0.005, 0.005, 0.010], [-0.010, -0.005, 0.005, 0.010]
                )
            ]
        )

        def voigt_to_full(v):
            return np.array(
                [[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]]
            )

        stresses = np.array(
            [voigt_to_full(s) for s in np.einsum("ij,nj->ni", C, strains)]
        )
        # ABACUS reports positive stress as compression (opposite of
        # pymatgen's positive = tension) — feed the raw kbar values with
        # the ABACUS sign so the fit's internal negation is exercised.
        res = fit_elastic_from_stress(strains, -(stresses / 0.1))  # GPa -> kbar
        c = np.array(res["elastic_tensor_gpa"])
        assert c[0, 0] == pytest.approx(200, abs=1e-6)
        assert c[0, 1] == pytest.approx(120, abs=1e-6)
        assert c[3, 3] == pytest.approx(60, abs=1e-6)
        assert res["bulk_modulus_gpa"] == pytest.approx(146.667, abs=0.01)
        # Young modulus from the VRH averages: Y = 9KG/(3K+G).
        assert res["young_modulus_gpa"] == pytest.approx(
            9 * 146.667 * 51 / (3 * 146.667 + 51), abs=0.05
        )


class TestFitElasticFromEnergy:
    def test_recovers_full_cubic_with_combined_strains(self):
        """With the combined-strain modes the full cubic tensor
        (C11, C12, C44) is recovered — including the off-diagonal C12."""
        C = _known_cubic()
        strains = _strain_set(with_combined=True)
        V = 3.45**3
        energies = _synthesize_energies(C, strains, V)

        res = fit_elastic_from_energy(strains, energies, V)
        c = np.array(res["elastic_tensor_gpa"])
        assert c[0, 0] == pytest.approx(200, abs=1e-3)
        assert c[1, 1] == pytest.approx(200, abs=1e-3)
        assert c[3, 3] == pytest.approx(60, abs=1e-3)
        assert c[0, 1] == pytest.approx(120, abs=1e-3)  # off-diagonal!
        assert c[0, 2] == pytest.approx(120, abs=1e-3)
        assert c[1, 2] == pytest.approx(120, abs=1e-3)
        assert res["diagonal_only"] is False
        assert res["energy_fit_e0_ev"] == pytest.approx(-56165.0, abs=1e-6)
        assert res["bulk_modulus_gpa"] == pytest.approx(146.667, abs=0.01)

    def test_recovers_full_orthorhombic(self):
        """All 9 independent orthorhombic constants recovered."""
        C = _known_orthorhombic()
        strains = _strain_set(with_combined=True)
        V = 3.45**3
        energies = _synthesize_energies(C, strains, V)

        res = fit_elastic_from_energy(strains, energies, V)
        c = np.array(res["elastic_tensor_gpa"])
        diag = [230, 260, 240, 70, 55, 80]
        off = [(0, 1, 130), (0, 2, 110), (1, 2, 120)]
        for i, expected in enumerate(diag):
            assert c[i, i] == pytest.approx(expected, abs=1e-3)
        for i, j, expected in off:
            assert c[i, j] == pytest.approx(expected, abs=1e-3)
        assert res["diagonal_only"] is False
        # Unprobed elements (e.g. C14) stay zero.
        assert c[0, 3] == pytest.approx(0.0, abs=1e-6)

    def test_single_strain_only_marks_diagonal_only(self):
        """Without combined strains the off-diagonal elements have no
        data and the result is marked diagonal_only (regression of the
        old behaviour)."""
        C = _known_cubic()
        strains = _strain_set(with_combined=False)
        V = 3.45**3
        energies = _synthesize_energies(C, strains, V)

        res = fit_elastic_from_energy(strains, energies, V)
        c = np.array(res["elastic_tensor_gpa"])
        assert c[0, 0] == pytest.approx(200, abs=1e-3)
        assert c[3, 3] == pytest.approx(60, abs=1e-3)
        # No combined strain → C12 has no data → zero, marked diagonal-only.
        assert c[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert res["diagonal_only"] is True
        assert res["note"] is not None


class TestReport:
    def test_young_modulus_derived_for_legacy_nodes(self):
        """Reports for workchains run before ``young_modulus_gpa`` existed
        derive it from the stored K_VRH / G_VRH (Y = 9KG/(3K+G))."""
        # 用用户报告的真实数据（K=151.182, G=32.353）
        table = generate_summary_table(
            {
                "method": "stress",
                "bulk_modulus_gpa": 151.182,
                "shear_modulus_gpa": 32.353,
                "bulk_modulus_voigt": 151.182,
                "bulk_modulus_reuss": 151.182,
                "shear_modulus_voigt": -20.297,
                "shear_modulus_reuss": 85.003,
                "universal_anisotropy": -6.194,
                "poisson_ratio": 0.400,
                # 无 young_modulus_gpa 键 —— 模拟旧节点
            },
            "abacus",
        )
        expected = 9 * 151.182 * 32.353 / (3 * 151.182 + 32.353)
        assert f"Young modulus Y_VRH | {expected:.3f} GPa" in table

    def test_young_modulus_uses_stored_value(self):
        """New nodes carry young_modulus_gpa — the report uses it directly."""
        table = generate_summary_table(
            {
                "method": "stress",
                "bulk_modulus_gpa": 151.182,
                "shear_modulus_gpa": 32.353,
                "young_modulus_gpa": 90.6,
            },
            "abacus",
        )
        assert "Young modulus Y_VRH | 90.600 GPa" in table

    def test_deformed_lattice_source_is_reported(self):
        """``require_lattice_relax`` failures must not be silent in the report."""
        relaxed = generate_summary_table(
            {"method": "stress", "equilibrium_structure_source": "lattice_relax"},
            "vasp",
        )
        assert "deformed lattice | relaxed (lattice_relax)" in relaxed
        assert "the strains were applied to the INPUT cell" not in relaxed

        unrelaxed = generate_summary_table(
            {"method": "stress", "equilibrium_structure_source": "input"},
            "abacus",
        )
        assert "deformed lattice | INPUT cell (not relaxed)" in unrelaxed
        assert "the strains were applied to the INPUT cell" in unrelaxed


class TestElasticSchedulerPresetNames:
    """Preset-name resolution for the elastic scheduler (vasp subkey)."""

    @staticmethod
    def _make_orchestrator(parameters_section):
        from aiida_uranium_workflow.schedulers.elastic import (
            ElasticWorkflowOrchestrator,
        )

        class _Bundle:
            pass

        bundle = _Bundle()
        bundle.input_params = {"parameters": parameters_section}
        bundle.software_params = {"abacus": [{}], "vasp": [{}], "fleur": [{}]}
        return ElasticWorkflowOrchestrator(bundle)

    def test_vasp_dict_form_yields_preset_names(self):
        """``{"vasp": {"elastic": ["test_u"]}}`` → ``["test_u"]`` (the vasp
        elastic presets are self-contained, subkey ``elastic``)."""
        orchestrator = self._make_orchestrator(
            {"vasp": {"elastic": ["test_u", "test_u_soc"]}}
        )
        assert orchestrator._preset_names_for("vasp") == ["test_u", "test_u_soc"]

    def test_vasp_string_form(self):
        orchestrator = self._make_orchestrator({"vasp": {"elastic": "test_u"}})
        assert orchestrator._preset_names_for("vasp") == ["test_u"]

    def test_abacus_uses_scf_subkey(self):
        """abacus presets come from the shared SCF base (subkey ``scf``)."""
        orchestrator = self._make_orchestrator(
            {"abacus": {"scf": ["test", "test_soc"]}}
        )
        assert orchestrator._preset_names_for("abacus") == ["test", "test_soc"]

    def test_preset_subkeys_registered(self):
        from aiida_uranium_workflow.schedulers.elastic import (
            ElasticWorkflowOrchestrator,
        )

        assert ElasticWorkflowOrchestrator.PRESET_SUBKEYS == {
            "abacus": "scf",
            "vasp": "elastic",
            "fleur": "scf",
            "qe": "scf",
        }
        assert "vasp" in ElasticWorkflowOrchestrator.ADAPTERS
        assert "qe" in ElasticWorkflowOrchestrator.ADAPTERS
        assert "qe" in ElasticWorkflowOrchestrator.BACKENDS


class TestElasticStability:
    """Born stability check: an unstable tensor must be flagged.

    Case 3-elastic (bcc-U, gamma project): C11 < C12 for both VASP and
    ABACUS, i.e. C' = (C11-C12)/2 < 0. The Reuss shear modulus denominator
    4(S11-S12)+3S44 then nearly cancels, so G_Reuss/G_VRH/Y flip between
    hundreds of GPa (VASP 822/417/611) and (ABACUS 200/104/247) for tensors
    that differ by only ~10 GPa — the moduli are not physical and must be
    flagged, not silently reported.
    """

    #: bcc-U-like unstable tensor (C11 < C12), GPa.
    UNSTABLE = np.array(
        [
            [94.31, 151.89, 151.89, 0.0, 0.0, 0.0],
            [151.89, 94.31, 151.89, 0.0, 0.0, 0.0],
            [151.89, 151.89, 94.31, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 39.71, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 39.71, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 39.71],
        ]
    )

    def test_result_flags_unstable_tensor(self):
        from pymatgen.analysis.elasticity import ElasticTensor

        from aiida_uranium_workflow.utils.elastic import _elastic_result

        res = _elastic_result(ElasticTensor.from_voigt(self.UNSTABLE), "stress")
        assert res["stable"] is False
        assert res["min_eigenvalue_gpa"] == pytest.approx(-57.58, abs=0.01)
        assert res["born_cubic"]["C11_minus_C12"] == pytest.approx(-57.58, abs=0.01)
        # The moduli are still computed (unchanged behaviour), but they are
        # the unstable-tensor VRH numbers — hence the flag.
        assert res["shear_modulus_gpa"] == pytest.approx(417.4, abs=0.5)

    def test_result_flags_stable_tensor(self):
        from pymatgen.analysis.elasticity import ElasticTensor

        from aiida_uranium_workflow.utils.elastic import _elastic_result

        res = _elastic_result(ElasticTensor.from_voigt(_known_cubic()), "stress")
        assert res["stable"] is True
        assert res["min_eigenvalue_gpa"] > 0

    def test_report_warns_about_unstable_tensor(self):
        from aiida_uranium_workflow.utils.report.elastic import (
            generate_summary_table,
        )

        table = generate_summary_table(
            {
                "method": "stress",
                "elastic_tensor_gpa": self.UNSTABLE.tolist(),
                "bulk_modulus_gpa": 132.697,
                "shear_modulus_gpa": 416.933,
                "young_modulus_gpa": 610.942,
                "bulk_modulus_voigt": 132.697,
                "bulk_modulus_reuss": 132.697,
                "shear_modulus_voigt": 12.308,
                "shear_modulus_reuss": 821.559,
                "universal_anisotropy": -4.925,
                "poisson_ratio": -0.267,
                "stable": False,
                "min_eigenvalue_gpa": -57.581,
                "born_cubic": {
                    "C11_minus_C12": -57.581,
                    "C11_plus_2C12": 398.092,
                    "C44": 39.707,
                },
            },
            "vasp",
        )
        assert "**mechanically UNSTABLE**" in table
        assert "C11−C12 = -57.581 GPa" in table
        assert "not physically meaningful" in table
        # the meaningless rows are marked
        assert "| Shear modulus G_VRH ⚠️ |" in table
        assert "| Poisson ratio ⚠️ |" in table

    def test_old_node_stability_is_recomputed(self):
        from aiida_uranium_workflow.utils.report.elastic import (
            generate_summary_table,
            stability_report,
        )

        # nodes created before the "stable" key existed
        old = {"elastic_tensor_gpa": self.UNSTABLE.tolist()}
        stable, min_eig, born = stability_report(old)
        assert stable is False
        assert min_eig == pytest.approx(-57.58, abs=0.01)
        assert born["C11_minus_C12"] == pytest.approx(-57.58, abs=0.01)
        table = generate_summary_table(old, "abacus")
        assert "**mechanically UNSTABLE**" in table

    def test_stable_tensor_has_no_warning(self):
        from aiida_uranium_workflow.utils.report.elastic import (
            generate_summary_table,
        )

        table = generate_summary_table(
            {
                "method": "stress",
                "elastic_tensor_gpa": _known_cubic().tolist(),
                "bulk_modulus_gpa": 146.667,
                "shear_modulus_gpa": 51.0,
                "young_modulus_gpa": 131.0,
                "stable": True,
                "min_eigenvalue_gpa": 60.0,
            },
            "abacus",
        )
        assert "UNSTABLE" not in table
        assert "⚠️" not in table
