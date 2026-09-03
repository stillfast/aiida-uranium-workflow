"""Tests for the phonopy workflow (protocol parsing, adapter, report, plot)."""

from __future__ import annotations

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.phonopy.abacus import (
    AbacusPhonopyAdapter,
)
from aiida_uranium_workflow.schedulers.phonopy import (
    parse_phonopy_protocol,
)
from aiida_uranium_workflow.utils.plot.phonon import (
    _resolve_labels,
    _segment_boundaries,
)
from aiida_uranium_workflow.utils.report.phonopy import (
    generate_report,
    generate_summary_table,
)

# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------


class TestProtocolParser:
    def test_forwards_phonopy_block(self):
        protocol = {
            "supercell_matrix": [3, 3, 3],
            "primitive_matrix": "auto",
            "band_mode": "auto",
            "dos": True,
        }
        out = parse_phonopy_protocol(protocol)
        assert out == {"phonopy": protocol}

    def test_empty(self):
        assert parse_phonopy_protocol(None) == {"phonopy": {}}
        assert parse_phonopy_protocol({}) == {"phonopy": {}}


# ---------------------------------------------------------------------------
# Adapter pure helpers
# ---------------------------------------------------------------------------


class TestBuildPhonopyParameters:
    def test_auto_mode(self):
        params = AbacusPhonopyAdapter.build_phonopy_parameters(
            {"band_mode": "auto", "band_paths": "bcc", "band_points": 61}
        )
        assert params["band"] == "bcc"  # BAND run-mode tag (phonopy 4.x)
        assert params["band_points"] == 61
        assert "band_paths" not in params

    def test_auto_defaults_to_seekpath(self):
        params = AbacusPhonopyAdapter.build_phonopy_parameters({})
        assert params["band"] == "auto"

    def test_manual_mode(self):
        params = AbacusPhonopyAdapter.build_phonopy_parameters(
            {
                "band_mode": "manual",
                "band": [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],
                "band_labels": ["Γ", "H"],
                "band_points": 51,
            }
        )
        assert params["band"] == [0.0, 0.0, 0.0, 0.5, -0.5, 0.5]
        assert params["band_labels"] == ["Γ", "H"]
        assert "band_paths" not in params

    def test_manual_mode_requires_band(self):
        with pytest.raises(ValueError, match="requires an explicit 'band'"):
            AbacusPhonopyAdapter.build_phonopy_parameters({"band_mode": "manual"})

    def test_dos(self):
        params = AbacusPhonopyAdapter.build_phonopy_parameters(
            {"dos": True, "mesh": [11, 11, 11], "fmax": 12.0}
        )
        assert params["dos"] is True
        assert params["mesh"] == [11, 11, 11]
        assert params["fmax"] == 12.0

    def test_no_dos(self):
        params = AbacusPhonopyAdapter.build_phonopy_parameters({"dos": False})
        assert "dos" not in params
        assert "mesh" not in params


class TestBuildDisplacementGenerator:
    def test_default_distance(self):
        gen = AbacusPhonopyAdapter.build_displacement_generator({})
        assert gen["distance"] == 0.01
        assert gen["is_plusminus"] == "auto"

    def test_custom_distance(self):
        gen = AbacusPhonopyAdapter.build_displacement_generator(
            {"displacement_distance": 0.02}
        )
        assert gen["distance"] == 0.02


# ---------------------------------------------------------------------------
# Adapter + WorkChain input assembly (needs the AiiDA profile)
# ---------------------------------------------------------------------------


class TestAdapterInputs:
    def test_build_inputs_and_validate_spec(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from aiida_uranium_workflow.workflows.phonopy.abacus import (
            AbacusPhonopyWorkChain,
        )
        from ase.build import bulk

        computer = aiida_localhost
        # NOTE: code labels must NOT contain '@' — ``load_code("abacus@localhost")``
        # would parse that as label "abacus" + computer "localhost".
        orm.InstalledCode(
            label="phonopy_test1_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="phonopy_test1_phonopy",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        abacus_identifier = f"phonopy_test1_abacus@{computer.label}"
        phonopy_identifier = f"phonopy_test1_phonopy@{computer.label}"

        adapter = AbacusPhonopyAdapter(
            code_label=abacus_identifier,
            software_params={
                "parameters": {"input": {"nspin": 2, "basis_type": "lcao"}},
                "kpoints_mesh": [11, 11, 11],
                "pseudo_family": "sg15_sz",
            },
            metadata={"options": {"resources": {"num_machines": 1}}},
            workflow_data={
                "phonopy": {
                    "supercell_matrix": [3, 3, 3],
                    "primitive_matrix": "auto",
                    "symprec": 1e-5,
                    "displacement_distance": 0.01,
                    "band_mode": "manual",
                    "band": [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],
                    "band_labels": ["Γ", "H"],
                    "band_points": 51,
                    "dos": True,
                    "mesh": [11, 11, 11],
                }
            },
            extra_codes={"phonopy": phonopy_identifier},
        )

        structure = orm.StructureData(ase=bulk("Si", crystalstructure="bcc", a=3.09))
        inputs = adapter._build_workchain_inputs(structure)

        # ABACUS SCF must be a fixed-lattice force run.
        abacus_input = inputs["base"]["abacus"]["parameters"].get_dict()["input"]
        assert abacus_input["calculation"] == "scf"
        assert abacus_input["cal_force"] == 1

        # phonopy inputs
        assert inputs["phonopy_code"].label == "phonopy_test1_phonopy"
        assert inputs["supercell_matrix"].get_list() == [3, 3, 3]
        assert inputs["primitive_matrix"].value == "auto"
        params = inputs["phonopy_parameters"].get_dict()
        assert params["band"] == [0.0, 0.0, 0.0, 0.5, -0.5, 0.5]
        assert params["band_labels"] == ["Γ", "H"]
        assert params["dos"] is True
        assert inputs["band_labels"].get_list() == ["Γ", "H"]
        gen = inputs["displacement_generator"].get_dict()
        assert gen["distance"] == 0.01

        # The assembled inputs must satisfy the WorkChain spec.
        err = AbacusPhonopyWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"

    def test_auto_mode_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from aiida_uranium_workflow.workflows.phonopy.abacus import (
            AbacusPhonopyWorkChain,
        )
        from ase.build import bulk

        computer = aiida_localhost
        # NOTE: code labels must NOT contain '@' (see first adapter test).
        orm.InstalledCode(
            label="phonopy_test2_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="phonopy_test2_phonopy",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        abacus_identifier = f"phonopy_test2_abacus@{computer.label}"
        phonopy_identifier = f"phonopy_test2_phonopy@{computer.label}"

        adapter = AbacusPhonopyAdapter(
            code_label=abacus_identifier,
            software_params={
                "parameters": {"input": {"nspin": 2}},
                "kpoints_distance": 0.1,
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "phonopy": {
                    "supercell_matrix": [2, 2, 2],
                    "primitive_matrix": None,
                    "band_mode": "auto",
                    "band_paths": "auto",
                    "dos": False,
                }
            },
            extra_codes={"phonopy": phonopy_identifier},
        )
        structure = orm.StructureData(ase=bulk("Si", crystalstructure="bcc", a=3.09))
        inputs = adapter._build_workchain_inputs(structure)

        assert "primitive_matrix" not in inputs  # None -> omit
        assert inputs["base"]["kpoints_distance"].value == 0.1
        params = inputs["phonopy_parameters"].get_dict()
        assert params["band"] == "auto"
        assert "band_paths" not in params
        assert "dos" not in params

        err = AbacusPhonopyWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_summary_table(self):
        output_params = {
            "backend": "abacus",
            "structure_formula": "Si2",
            "n_supercells": 1,
            "supercell_matrix": [3, 3, 3],
            "primitive_matrix": "auto",
            "symprec": 1e-5,
            "band_labels": ["Γ", "H"],
            "phonopy_parameters": {"band_paths": "auto", "band_points": 51},
            "frequency_min_thz": -2.5,
            "frequency_max_thz": 8.5,
            "n_imaginary_modes": 157,
            "phonopy_pk": 123,
            "phonopy_uuid": "abcd-1234",
        }
        table = generate_summary_table(output_params)
        assert "backend | abacus" in table
        assert "displaced supercells | 1" in table
        assert "band mode | auto (BAND_PATHS=auto)" in table
        assert "-2.5000" in table and "8.5000" in table
        assert "157" in table

    def test_generate_report_without_figure(self):
        output_params = {"backend": "abacus", "n_supercells": 1}
        report = generate_report(output_params, 42, "abacus")
        assert "Phonon Report" in report
        assert "(PK: 42)" in report
        assert "## Summary" in report
        assert "*Generated by aiida-uranium-workflow*" in report


# ---------------------------------------------------------------------------
# Plot helpers (pure)
# ---------------------------------------------------------------------------


class TestPlotHelpers:
    def test_segment_boundaries(self):
        assert _segment_boundaries(256, 6) == [0, 51, 102, 153, 204, 255]
        assert _segment_boundaries(256, 1) == [0]

    def test_resolve_labels_prefers_explicit(self):
        class FakeBands:
            def get_kpoints(self):
                import numpy as np

                return np.zeros((256, 3))

        pos, labels = _resolve_labels(FakeBands(), ["Γ", "H", "N", "Γ"])
        assert labels == ["Γ", "H", "N", "Γ"]
        assert pos == _segment_boundaries(256, 4)

    def test_resolve_labels_filters_broken(self):
        class FakeBands:
            def get_kpoints(self):
                import numpy as np

                return np.zeros((100, 3))

            class base:
                attributes = {"labels": [[0, "?"], [49, "?"]]}

        pos, labels = _resolve_labels(FakeBands(), None)
        assert pos is None and labels is None


class TestFleurForceExtraction:
    def test_extract_forces_converts_units(self, aiida_profile_clean):
        """FLEUR out.xml force_atoms (Htr/bohr) → forces ArrayData (eV/Å)."""
        import numpy as np

        from aiida import orm

        from aiida_uranium_workflow.workflows.phonopy.fleur import (
            _extract_fleur_forces,
            HTR_PER_BOHR_TO_EV_PER_ANG,
        )

        # 2 atoms: [(atom_type, [fx, fy, fz]), ...] in Htr/bohr.
        force_atoms = [[
            (1, [0.001, 0.0, 0.0]),
            (1, [0.0, -0.002, 0.0]),
        ]]
        para = orm.Dict(dict={"force_atoms": force_atoms})
        array = _extract_fleur_forces(output_parameters=para)
        forces = np.asarray(array.get_array("forces"))
        assert forces.shape == (2, 3)
        assert forces[0, 0] == pytest.approx(0.001 * HTR_PER_BOHR_TO_EV_PER_ANG)
        assert forces[1, 1] == pytest.approx(-0.002 * HTR_PER_BOHR_TO_EV_PER_ANG)

    def test_extract_forces_missing_force_atoms(self, aiida_profile_clean):
        from aiida import orm

        from aiida_uranium_workflow.workflows.phonopy.fleur import _extract_fleur_forces

        with pytest.raises(ValueError, match="force_atoms"):
            _extract_fleur_forces(output_parameters=orm.Dict(dict={}))

    @pytest.mark.parametrize(
        "last_iter,expected",
        [
            # [(atomType, [fx, fy, fz]), ...]
            ([[1, [0.001, 0.0, 0.0]], [1, [0.0, -0.002, 0.0]]], [[0.001, 0, 0], [0, -0.002, 0]]),
            # [[fx, fy, fz], ...]
            ([[0.001, 0.0, 0.0], [0.0, -0.002, 0.0]], [[0.001, 0, 0], [0, -0.002, 0]]),
            # [atomType, [fx, fy, fz], atomType, ...] flat alternating
            ([1, [0.001, 0.0, 0.0], 2, [0.0, -0.002, 0.0]], [[0.001, 0, 0], [0, -0.002, 0]]),
            # [atomType, fx, fy, fz] rows
            ([[1, 0.001, 0.0, 0.0], [1, 0.0, -0.002, 0.0]], [[0.001, 0, 0], [0, -0.002, 0]]),
        ],
    )
    def test_parse_force_atoms_layouts(self, last_iter, expected):
        import numpy as np

        from aiida_uranium_workflow.workflows.phonopy.fleur import _parse_force_atoms

        out = _parse_force_atoms(last_iter)
        assert np.allclose(out, np.asarray(expected, dtype=float))


class TestBreakFleurSymmetry:
    def test_distinct_kind_per_atom_order_preserved(self, aiida_profile_clean):
        """Each supercell atom gets its own kind; positions/order unchanged."""
        import numpy as np

        from aiida import orm

        from aiida_uranium_workflow.workflows.phonopy.fleur import (
            _break_fleur_symmetry,
        )
        from ase.build import bulk

        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45).repeat((2, 1, 1)))
        out = _break_fleur_symmetry(structure=structure)

        # One kind per atom, in atom order.
        assert out.get_kind_names() == ["U1", "U2"]
        assert [out.get_kind(site.kind_name).symbol for site in out.sites] == [
            "U",
            "U",
        ]
        for got, want in zip(out.sites, structure.sites):
            np.testing.assert_allclose(got.position, want.position)
        np.testing.assert_allclose(out.cell, structure.cell)
        assert out.pbc == structure.pbc

    def test_inpgen_input_gets_fractional_species(self, aiida_profile_clean):
        """The inpgen input must contain '92.1'/'92.2' species lines (P1)."""
        import io

        from aiida import orm

        from aiida_uranium_workflow.workflows.phonopy.fleur import (
            _break_fleur_symmetry,
        )
        from aiida_fleur.calculation.fleurinputgen import (
            write_inpgen_file_aiida_struct,
        )
        from ase.build import bulk

        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45).repeat((2, 1, 1)))
        out = _break_fleur_symmetry(structure=structure)

        buf = io.StringIO()
        write_inpgen_file_aiida_struct(
            out, buf, input_params={"input": {"cartesian": False}}
        )
        content = buf.getvalue()
        # Fractional atomic numbers → each atom its own FLEUR species, so
        # only the identity symmetry operation survives.
        assert "92.1" in content
        assert "92.2" in content
        assert content.count("92.") == 2


class TestFleurPhonopyAdapter:
    def test_build_inputs_forces_force_mode(self, aiida_profile, aiida_localhost):
        from aiida import orm

        from aiida_uranium_workflow.input_builders.phonopy.fleur import FleurPhonopyAdapter
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="phonopy_fleur_test",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="phonopy_inpgen_test",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="phonopy_phonopy_test",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = FleurPhonopyAdapter(
            code_label=f"phonopy_fleur_test@{computer.label}",
            software_params={
                "wf_parameters": {"mode": "density", "itmax_per_run": 100},
                "calc_parameters": {"comp": {"kmax": 7.0}},
            },
            metadata={},
            workflow_data={
                "phonopy": {
                    "supercell_matrix": [2, 2, 2],
                    "primitive_matrix": "auto",
                    "band_paths": "auto",
                    "dos": True,
                }
            },
            extra_codes={
                "inpgen": f"phonopy_inpgen_test@{computer.label}",
                "phonopy": f"phonopy_phonopy_test@{computer.label}",
            },
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)

        # The SCF stays in density mode (a fixed-lattice phonopy run has
        # no relax.xml, so aiida-fleur's force mode would never converge)
        # but forces are enabled via an explicit l_f inpxml change.
        wf = inputs["base"]["wf_parameters"].get_dict()
        assert wf["mode"] == "density"
        assert "force_dict" in wf
        assert any(
            isinstance(change, list)
            and change[:1] == ["set_inpchanges"]
            and change[1].get("changes", {}).get("l_f") is True
            for change in wf.get("inpxml_changes", [])
        )
        assert "fleur" in inputs["base"]
        assert "inpgen" in inputs["base"]
        assert "phonopy_code" in inputs
        assert inputs["supercell_matrix"].get_list() == [2, 2, 2]
