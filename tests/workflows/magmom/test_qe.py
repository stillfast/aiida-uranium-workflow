"""Tests for the QE (pw.x) magmom workflow and adapter."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import pytest

from aiida_uranium_workflow.schedulers.magmom import parse_magmom_protocol
from aiida_uranium_workflow.workflows.magmom.qe import (
    QeMagmomWorkChain,
    _magmom_to_label,
)


class TestDefinition:
    def test_workchain_definition(self):
        spec = QeMagmomWorkChain.spec()
        assert "magmom_list" in spec.inputs
        assert "output_parameters" in spec.outputs

    def test_exit_codes(self):
        assert QeMagmomWorkChain.exit_codes.SUCCESS.status == 0
        assert QeMagmomWorkChain.exit_codes.ERROR_CHILD.status == 300
        assert QeMagmomWorkChain.exit_codes.ERROR_PARSER.status == 305

    def test_magmom_to_label(self):
        assert _magmom_to_label({"U": 2.0}) == "U_2"
        assert _magmom_to_label({"U": 0.5}) == "U_0_5"
        assert _magmom_to_label({"U": -4.0}) == "U_m4"


class TestProtocolParser:
    def test_qe_magmom_list(self):
        proto = {"qe": {"magmom_list": [{"U": 0.0}, {"U": 4.0}]}}
        out = parse_magmom_protocol(proto)
        assert out["magmom_lists"]["qe"]["magmom_list"] == [{"U": 0.0}, {"U": 4.0}]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_magmom_protocol({"qe": {"magmom_list": []}})


class TestAdapter:
    def test_qe_inputs(self, aiida_profile, aiida_localhost, monkeypatch):
        from aiida import orm
        from ase.build import bulk
        from aiida_uranium_workflow.input_builders.magmom import qe as qe_mod
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        computer = aiida_localhost
        orm.InstalledCode(
            label="qe_test_pw",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        # Fake SSSP-like family: get_pseudos(structure) + recommended
        # cutoffs in Ry. The pseudo must be a real UpfData for the spec
        # validation below.
        from aiida.orm import UpfData

        upf_text = (
            "<UPF version=\"2.0.1\"><PP_HEADER element=\"U\" z_valence=\"14.0\"/>"
            "<PP_MESH><PP_R><PP_R type=\"real\">1.0 2.0</PP_R></PP_R>"
            "<PP_RAB><PP_RAB type=\"real\">1.0 2.0</PP_RAB></PP_RAB></PP_MESH>"
            "<PP_NLCC><PP_NLCC type=\"real\">0.0 0.0</PP_NLCC></PP_NLCC>"
            "<PP_LOCAL><PP_LOCAL type=\"real\">0.0 0.0</PP_LOCAL></PP_LOCAL>"
            "<PP_NONLOCAL/>"
            "<PP_PSWFC><PP_CHI type=\"real\" columns=\"2\">0.0 0.0</PP_CHI></PP_PSWFC>"
            "</UPF>"
        )
        import pathlib
        import tempfile

        upf_path = pathlib.Path(tempfile.mkdtemp()) / "U.upf"
        upf_path.write_text(upf_text, encoding="utf-8")
        fake_upf = UpfData(upf_path)
        fake_upf.store()

        class _FakeFamily:
            def __init__(self, upf):
                self._upf = upf

            def get_pseudos(self, structure=None, elements=None):
                return {"U": self._upf}

            def get_recommended_cutoffs(self, structure=None, unit=None):
                return (60.0, 480.0)

        monkeypatch.setattr(
            qe_mod, "load_group", lambda name: _FakeFamily(fake_upf)
        )

        adapter = QeMagmomAdapter(
            code_label=f"qe_test_pw@{computer.label}",
            software_params={
                "pw": {"parameters": {"SYSTEM": {"nspin": 2, "ecutwfc": 99}}},
                "kpoints_mesh": [11, 11, 11],
                "pseudo_family": "test_qe_family",
            },
            metadata={
                "options": {
                    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                    "max_wallclock_seconds": 3600,
                }
            },
            workflow_data={
                "magmom_lists": {"qe": {"magmom_list": [{"U": 0.0}, {"U": 4.0}]}}
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure, [{"U": 0.0}, {"U": 4.0}])
        # SSSP recommended cutoffs override the preset reference value.
        system = inputs["pw"]["parameters"].get_dict()["SYSTEM"]
        assert system["ecutwfc"] == 60.0
        assert system["ecutrho"] == 480.0
        # Pseudos come from the family (keyed by element).
        assert inputs["pw"]["pseudos"]["U"] is fake_upf
        # 11×11×11 mesh from the preset.
        kpoints = inputs["kpoints"]
        assert kpoints.get_kpoints_mesh()[0] == [11, 11, 11]
        # adapt() resolves the entry point (requires a reinstall); verify
        # the assembled inputs validate against the WorkChain spec with
        # the magmom_list injected the same way adapt() would.
        inputs["magmom_list"] = orm.List(list=[{"U": 0.0}, {"U": 4.0}])
        err = QeMagmomWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"


class TestSpeciesSplit:
    """AFM seeds: U1/U2 split-species relabelling."""

    def test_split_structure_and_entries(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        # 2-atom bcc conventional cell (1-atom primitive would split to U1 only).
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45, cubic=True))
        adapter = QeMagmomAdapter(
            code_label="x@localhost", software_params={}, metadata={},
            workflow_data={
                "magmom_lists": {"qe": {"magmom_list": [
                    {"U": 0.0},                 # NM
                    {"U": 2.0},                 # FM
                    {"U1": 2.0, "U2": -2.0},    # AFM
                ]}}
            },
            extra_codes={},
        )
        new_structure, entries = adapter._prepare_structure_and_entries(structure)
        # 2-atom bcc cell → split into U1/U2 alternating.
        kinds = [s.kind_name for s in new_structure.sites]
        assert kinds == ["U1", "U2"]
        # Element-level entries expanded to all species.
        assert entries == [
            {"U1": 0.0, "U2": 0.0},
            {"U1": 2.0, "U2": 2.0},
            {"U1": 2.0, "U2": -2.0},
        ]

    def test_no_split_for_element_keys(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        adapter = QeMagmomAdapter(
            code_label="x@localhost", software_params={}, metadata={},
            workflow_data={
                "magmom_lists": {"qe": {"magmom_list": [{"U": 0.0}, {"U": 2.0}]}}
            },
            extra_codes={},
        )
        new_structure, entries = adapter._prepare_structure_and_entries(structure)
        assert new_structure is structure
        assert entries == [{"U": 0.0}, {"U": 2.0}]
