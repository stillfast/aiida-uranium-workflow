"""Tests for the QE (pw.x) magmom workflow and adapter."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import pytest

from aiida_uranium_workflow.schedulers.magmom import parse_magmom_protocol
from aiida_uranium_workflow.workflows.magmom.qe import (
    QeMagmomWorkChain,
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


class TestLoadFamilyAmbiguous:
    """A collection and a family group sharing one label must resolve to
    the family-type group (regression: aiida-abacus' install-collection +
    create-family can produce both under the same label, and plain
    ``load_group`` then raises MultipleObjectsError)."""

    def test_prefers_family_type(self, aiida_profile):
        from aiida import orm
        from aiida.common.exceptions import MultipleObjectsError
        from aiida.plugins import GroupFactory
        from aiida_uranium_workflow.input_builders.magmom import qe as qe_mod
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        label = "dup_soc_qe"
        # A plain (collection-like) group and a family-type group, same label.
        orm.Group(label=label).store()
        family_cls = GroupFactory("abacus.orbital_family")
        family = family_cls(label=label).store()

        def _ambiguous(*_args, **_kwargs):
            raise MultipleObjectsError("multiple Group entries found")

        # The adapter's module-level load_group is what _load_family calls;
        # make it raise so the QueryBuilder fallback is exercised.
        monkeypatch = __import__("pytest").MonkeyPatch()
        monkeypatch.setattr(qe_mod, "load_group", _ambiguous)
        try:
            adapter = QeMagmomAdapter(
                code_label="x@localhost",
                software_params={"pseudo_family": label},
                metadata={},
                workflow_data={"magmom_lists": {"qe": {"magmom_list": [{"U": 0.0}]}}},
                extra_codes={},
            )
            loaded = adapter._load_family()
        finally:
            monkeypatch.undo()

        assert loaded.label == label
        # The family-type group (not the plain one) was selected.
        assert loaded.uuid == family.uuid

    def test_unique_label_unchanged(self, aiida_profile):
        from aiida import orm
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        grp = orm.Group(label="unique_qe_family").store()
        adapter = QeMagmomAdapter(
            code_label="x@localhost",
            software_params={"pseudo_family": "unique_qe_family"},
            metadata={},
            workflow_data={"magmom_lists": {"qe": {"magmom_list": [{"U": 0.0}]}}},
            extra_codes={},
        )
        assert adapter._load_family().pk == grp.pk


class TestSocSeeds:
    """Non-collinear / SOC seeds: reserved angle keys must not be treated
    as split species, and the WorkChain must not force nspin=2 for SOC."""

    def test_expand_entries_keeps_angles(self):
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        entries = [
            {"U1": 2.0, "U2": 2.0, "angle1": {"U1": 0.0, "U2": 180.0}},
            {"U": 0.0},
        ]
        out = QeMagmomAdapter._expand_entries(entries, ["U1", "U2"])
        assert out[0] == {
            "U1": 2.0,
            "U2": 2.0,
            "angle1": {"U1": 0.0, "U2": 180.0},
        }
        assert out[1] == {"U1": 0.0, "U2": 0.0}  # element-level FM expands

    def test_angles_do_not_trigger_species_split(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm
        from aiida_uranium_workflow.input_builders.magmom.qe import QeMagmomAdapter

        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        adapter = QeMagmomAdapter(
            code_label="x@localhost", software_params={}, metadata={},
            workflow_data={
                "magmom_lists": {"qe": {"magmom_list": [
                    {"U": 2.0, "angle1": {"U": 0.0}},
                ]}}
            },
            extra_codes={},
        )
        new_structure, entries = adapter._prepare_structure_and_entries(structure)
        # angle1 alone (U element key) must NOT trigger a U1/U2 split.
        assert [s.kind_name for s in new_structure.sites] == ["U"]
        assert entries == [{"U": 2.0, "angle1": {"U": 0.0}}]

    def test_workchain_soc_injection(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        """SOC preset: nspin stays as preset (1), angle keys land in SYSTEM."""
        from ase.build import bulk

        from aiida import orm
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.orm import UpfData
        from aiida_uranium_workflow.workflows.magmom.qe import QeMagmomWorkChain

        import pathlib
        import tempfile

        computer = aiida_localhost
        orm.InstalledCode(
            label="qe_soc_test",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        # A real (minimal) UpfData so the pw spec validation passes.
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
        upf_path = pathlib.Path(tempfile.mkdtemp()) / "U.upf"
        upf_path.write_text(upf_text, encoding="utf-8")
        fake_upf = UpfData(upf_path)
        fake_upf.store()

        runner = get_manager().get_runner()
        process = instantiate_process(
            runner,
            QeMagmomWorkChain,
            pw={
                "code": orm.load_code("qe_soc_test"),
                "structure": orm.StructureData(ase=bulk("U", "bcc", a=3.45)),
                "parameters": orm.Dict(dict={
                    "CONTROL": {"calculation": "scf"},
                    "SYSTEM": {"ecutwfc": 60, "noncolin": True, "lspinorb": True, "nspin": 1},
                }),
                "pseudos": {"U": fake_upf},
            },
            kpoints=orm.KpointsData(),
            magmom_list=orm.List(list=[{
                "U": 2.0, "angle1": {"U": 0.0},
            }]),
        )
        # Stub submit to capture per-child parameters.
        captured = {}

        def fake_submit(cls, **inputs):
            captured["parameters"] = inputs["pw"]["parameters"].get_dict()
            return __import__("types").SimpleNamespace(pk=1)

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.submit_children()

        system = captured["parameters"]["SYSTEM"]
        assert system["nspin"] == 1  # SOC: preset value kept (not forced 2)
        assert system["starting_magnetization"] == {"U": 2.0}
        assert system["angle1"] == {"U": 0.0}