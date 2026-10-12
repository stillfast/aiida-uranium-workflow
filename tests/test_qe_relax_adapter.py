"""Tests for the QE (pw.x) relax adapter (plugin PwRelaxWorkChain wrapper)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import pytest

from aiida import orm

from aiida_uranium_workflow.input_builders.relax.qe import QeRelaxAdapter


def _upf():
    from aiida.orm import UpfData

    import pathlib
    import tempfile

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
    path = pathlib.Path(tempfile.mkdtemp()) / "U.upf"
    path.write_text(upf_text, encoding="utf-8")
    upf = UpfData(path)
    upf.store()
    return upf


class TestQeRelaxAdapter:
    def test_inputs(self, aiida_profile, aiida_localhost, monkeypatch):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.relax import qe as qe_mod

        computer = aiida_localhost
        orm.InstalledCode(
            label="qe_relax_adapter",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()
        upf = _upf()

        class _FakeFamily:
            def get_pseudos(self, structure=None, elements=None):
                return {"U": upf}

            def get_recommended_cutoffs(self, structure=None, unit=None):
                return (60.0, 480.0)

        monkeypatch.setattr(qe_mod, "load_group", lambda name: _FakeFamily())

        adapter = QeRelaxAdapter(
            code_label=f"qe_relax_adapter@{computer.label}",
            software_params={
                "pw": {"parameters": {"SYSTEM": {"ecutwfc": 99}}},
                "kpoints_mesh": [7, 7, 7],
                "pseudo_family": "test_qe_relax_family",
            },
            metadata={
                "options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}
            },
            workflow_data={
                "qe": {"meta_convergence": True, "max_meta_convergence_iterations": 5}
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)

        # PwRelaxWorkChain layout: structure at the top level, the
        # PwBaseWorkChain inputs under ``base_relax``.
        assert inputs["structure"] is structure
        base_relax = inputs["base_relax"]
        assert base_relax["pw"]["code"].label == "qe_relax_adapter"
        assert base_relax["pw"]["structure"] is structure
        assert base_relax["pw"]["pseudos"]["U"] is upf
        system = base_relax["pw"]["parameters"].get_dict()["SYSTEM"]
        assert system["ecutwfc"] == 60.0  # family cutoffs applied
        assert base_relax["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]
        # Scheduler options landed under the pw namespace.
        opts = base_relax["pw"]["metadata"]["options"]
        assert opts["resources"]["num_machines"] == 1
        # WorkChain-level controls forwarded.
        assert inputs["meta_convergence"].value is True
        assert inputs["max_meta_convergence_iterations"].value == 5

        # The assembled inputs validate against the plugin spec.
        from aiida.plugins import WorkflowFactory

        err = WorkflowFactory("quantumespresso.pw.relax").spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"


class TestQeRelaxMode:
    """``relax_type`` / ``force_criterion`` protocol keys map onto pw params."""

    @pytest.fixture(autouse=True)
    def _code(self, aiida_profile_clean, aiida_localhost):
        orm.InstalledCode(
            label="qe_relax_mode",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        self.code_label = f"qe_relax_mode@{aiida_localhost.label}"

    def _adapter(self, workflow_data):
        return QeRelaxAdapter(
            code_label=self.code_label,
            software_params={
                "pw": {"parameters": {"CONTROL": {"calculation": "relax"}}},
            },
            metadata={},
            workflow_data=workflow_data,
            extra_codes={},
        )

    def test_positions_default(self):
        adapter = self._adapter({"qe": {}})
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "relax"

    def test_volume_mode(self):
        # QE only accepts cell_dofree='volume' for ibrav=1; for other
        # lattices a plain vc-relax keeps the shape by symmetry. A
        # permissive press_conv_thr is injected so aiida-qe does not
        # flag the converged final SCF (default 0.5 kbar is stricter
        # than QE's own stopping residual).
        adapter = self._adapter({"qe": {"relax_type": "volume"}})
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "vc-relax"
        assert "cell_dofree" not in params.get("CELL", {})
        assert params["CELL"]["press_conv_thr"] == 1.0

    def test_press_conv_thr_from_protocol(self):
        adapter = self._adapter(
            {"qe": {"relax_type": "volume", "press_conv_thr": 2.0}}
        )
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CELL"]["press_conv_thr"] == 2.0

    def test_full_mode(self):
        adapter = self._adapter({"qe": {"relax_type": "full"}})
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "vc-relax"
        assert params["CELL"]["cell_dofree"] == "all"

    def test_cell_dofree_passthrough(self):
        # An explicit protocol cell_dofree overrides the relax_type map.
        adapter = self._adapter(
            {"qe": {"relax_type": "volume", "cell_dofree": "volume"}}
        )
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "vc-relax"
        assert params["CELL"]["cell_dofree"] == "volume"

    def test_invalid_relax_type(self):
        adapter = self._adapter({"qe": {"relax_type": "bogus"}})
        with pytest.raises(ValueError, match="relax_type"):
            adapter._build_workchain_inputs(_structure())

    def test_force_criterion_converted(self):
        # 0.03 eV/Å → CONTROL.forc_conv_thr Ry/bohr (÷ 25.711) — the
        # force criterion lives in &CONTROL for QE 7.5, not &IONS.
        adapter = self._adapter({"qe": {"force_criterion": 0.03}})
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["forc_conv_thr"] == pytest.approx(0.03 / 25.711)
        assert "force_conv_thr" not in params.get("IONS", {})

    def test_preset_survives_without_protocol(self):
        adapter = self._adapter({"qe": {}})
        params = adapter._build_workchain_inputs(
            _structure()
        )["base_relax"]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "relax"


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))
