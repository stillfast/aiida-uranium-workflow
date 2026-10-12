"""Tests for the QE (pw.x + ph.x) phonon workflow and adapter.

The QE phonon workflow is the aiida-qe **DFPT** chain — pw.x SCF →
``quantumespresso.ph.base`` (ph.x on a q-mesh) →
``quantumespresso.q2r.base`` → ``quantumespresso.matdyn.base`` (band
path + DOS) — running on the seekpath primitive cell of the input
structure. The tests drive each outline step with stubbed children
(``submit`` monkeypatched) and verify the wiring: primitive-cell SCF,
ph.x parent folder + q-mesh, q2r parent folder, matdyn force constants
+ band path / DOS mesh, and the gather conversions (labels, cm⁻¹→THz).
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import numpy as np
import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.phonopy.qe import QePhonopyWorkChain


def _structure(a=3.45):
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=a, cubic=True))


def _primitive(a=3.45):
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=a, cubic=False))


def _fake_upf():
    """A real (minimal) UpfData so the pw spec validation passes."""
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
    upf_path = pathlib.Path(tempfile.mkdtemp()) / "U.upf"
    upf_path.write_text(upf_text, encoding="utf-8")
    upf = orm.UpfData(upf_path)
    upf.store()
    return upf


def _codes(aiida_localhost, prefix="qe_phonopy_test"):
    """Store the four chain codes (pw / ph / q2r / matdyn)."""
    labels = {}
    for key in ("pw", "ph", "q2r", "matdyn"):
        label = f"{prefix}_{key}"
        orm.InstalledCode(
            label=label,
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        labels[key] = f"{label}@{aiida_localhost.label}"
    return labels


def _instantiate(
    aiida_profile_clean,
    aiida_localhost,
    *,
    dos=True,
    dos_mesh=(9, 9, 9),
):
    labels = _codes(aiida_localhost)
    runner = get_manager().get_runner()
    return instantiate_process(
        runner,
        QePhonopyWorkChain,
        structure=_structure(),
        base={
            "pw": {
                "code": orm.load_code(labels["pw"]),
                "structure": _structure(),
                "parameters": orm.Dict(dict={
                    "CONTROL": {"calculation": "scf"},
                    "SYSTEM": {"ecutwfc": 60, "ecutrho": 480},
                }),
                "pseudos": {"U": _fake_upf()},
            },
            "kpoints_distance": orm.Float(0.1),
        },
        ph_code=orm.load_code(labels["ph"]),
        ph_parameters=orm.Dict(dict={"INPUTPH": {"tr2_ph": 1.0e-14}}),
        qpoints_mesh=orm.List(list=[4, 4, 4]),
        q2r_code=orm.load_code(labels["q2r"]),
        matdyn_code=orm.load_code(labels["matdyn"]),
        dos=orm.Bool(dos),
        dos_mesh=orm.List(list=list(dos_mesh)),
    )


def _path_kpoints():
    """A seekpath-style explicit q-point path with labels."""
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(_primitive())
    kpoints.set_kpoints(
        np.array(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.0, 0.0],
             [0.5, 0.25, 0.25], [0.5, 0.5, 0.5]]
        ),
        labels=[[0, "Γ"], [4, "H"]],
    )
    return kpoints


def _fake_seekpath(monkeypatch):
    """Patch the seekpath calcfunction: primitive cell + explicit path."""
    from aiida_uranium_workflow.workflows.phonopy import qe as qe_mod

    def fake(structure, **kwargs):
        return {
            "primitive_structure": _primitive(),
            "explicit_kpoints": _path_kpoints(),
            "parameters": orm.Dict(dict={}),
        }

    monkeypatch.setattr(qe_mod, "seekpath_structure_analysis", fake)


def _capture_submit(monkeypatch, process, node=None):
    captured = []

    def fake_submit(cls, **inputs):
        captured.append(inputs)
        return node or SimpleNamespace(pk=1, process_label="child")

    monkeypatch.setattr(process, "submit", fake_submit)
    monkeypatch.setattr(process, "to_context", lambda **kw: None)
    return captured


class TestDefinition:
    def test_spec(self):
        spec = QePhonopyWorkChain.spec()
        for key in (
            "structure", "base", "ph_code", "ph_parameters",
            "qpoints_mesh", "q2r_code", "matdyn_code", "dos",
        ):
            assert key in spec.inputs
        for key in ("phonon_bands", "force_constants", "output_parameters"):
            assert key in spec.outputs

    def test_exit_codes(self):
        assert QePhonopyWorkChain.exit_codes.SUCCESS.status == 0
        assert QePhonopyWorkChain.exit_codes.ERROR_CHILD.status == 300
        assert QePhonopyWorkChain.exit_codes.ERROR_PARSER.status == 305


class TestPrepare:
    def test_primitivises_and_keeps_path(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        _fake_seekpath(monkeypatch)
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        assert process.prepare() is None
        # The chain runs on the primitive (1-atom bcc) cell.
        assert len(process.ctx.primitive_structure.sites) == 1
        assert len(process.ctx.band_kpoints.get_kpoints()) == 5
        assert process.ctx.band_kpoints.base.attributes.get("labels") == ["Γ", "H"]
        assert process.ctx.band_kpoints.base.attributes.get("label_numbers") == [0, 4]


class TestRunScf:
    def test_scf_on_primitive_cell(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        _fake_seekpath(monkeypatch)
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.prepare()
        captured = _capture_submit(monkeypatch, process)
        process.run_scf()

        assert len(captured) == 1
        child = captured[0]
        assert child["pw"]["structure"] is process.ctx.primitive_structure
        params = child["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "scf"
        assert params["SYSTEM"]["ecutwfc"] == 60  # preset survives


class TestRunPh:
    def test_ph_parent_and_qmesh(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.scf = SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(remote_folder=SimpleNamespace(pk=7)),
        )
        captured = _capture_submit(monkeypatch, process)
        process.run_ph()

        assert len(captured) == 1
        ph = captured[0]["ph"]
        assert ph["parent_folder"].pk == 7  # the SCF remote folder
        assert ph["code"].label.startswith("qe_phonopy_test_ph")
        params = ph["parameters"].get_dict()
        assert params["INPUTPH"]["tr2_ph"] == 1.0e-14
        mesh, _offset = captured[0]["qpoints"].get_kpoints_mesh()
        assert mesh == [4, 4, 4]

    def test_ph_requires_finished_scf(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.scf = SimpleNamespace(is_finished_ok=False)
        _capture_submit(monkeypatch, process)
        assert process.run_ph() == QePhonopyWorkChain.exit_codes.ERROR_CHILD


class TestRunQ2r:
    def test_q2r_parent(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.ph_run = SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(remote_folder=SimpleNamespace(pk=11)),
        )
        captured = _capture_submit(monkeypatch, process)
        process.run_q2r()

        assert len(captured) == 1
        q2r = captured[0]["q2r"]
        assert q2r["parent_folder"].pk == 11  # the ph.x remote folder
        assert q2r["code"].label.startswith("qe_phonopy_test_q2r")


class TestRunMatdyn:
    def test_band_and_dos_runs(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.q2r_run = SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(force_constants=SimpleNamespace(pk=13)),
        )
        process.ctx.band_kpoints = _path_kpoints()
        captured = _capture_submit(monkeypatch, process)
        process.run_matdyn()

        assert len(captured) == 2
        band, dos = captured
        # Band run: seekpath path kpoints, no DOS.
        assert band["matdyn"]["force_constants"].pk == 13
        assert band["matdyn"]["kpoints"] is process.ctx.band_kpoints
        band_input = band["matdyn"]["parameters"].get_dict()["INPUT"]
        assert band_input["dos"] is False
        # Acoustic-sum rule enforced at interpolation time (QE 7.5 matdyn
        # reads ``asr``; q2r.x reads ``zasr``).
        assert band_input["asr"] == "crystal"
        assert band["matdyn"]["code"].label.startswith("qe_phonopy_test_matdyn")
        # DOS run: dense mesh kpoints, dos=True.
        assert dos["matdyn"]["force_constants"].pk == 13
        mesh, _offset = dos["matdyn"]["kpoints"].get_kpoints_mesh()
        assert mesh == [9, 9, 9]
        dos_input = dos["matdyn"]["parameters"].get_dict()["INPUT"]
        assert dos_input["dos"] is True
        assert dos_input["asr"] == "crystal"

    def test_dos_disabled(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, dos=False)
        process.ctx.q2r_run = SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(force_constants=SimpleNamespace(pk=13)),
        )
        process.ctx.band_kpoints = _path_kpoints()
        captured = _capture_submit(monkeypatch, process)
        process.run_matdyn()
        assert len(captured) == 1  # band only

    def test_requires_finished_q2r(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.q2r_run = SimpleNamespace(is_finished_ok=False)
        _capture_submit(monkeypatch, process)
        assert process.run_matdyn() == QePhonopyWorkChain.exit_codes.ERROR_CHILD


class TestGather:
    @staticmethod
    def _frequencies_folder(freqs_cm, nks):
        """A FolderData with a QE-7.5-style ``phonon_frequencies.dat``.

        Layout per q-point: 4 q-coordinate tokens + ``nbnd`` frequencies
        in cm⁻¹ (this is the format the aiida-qe 5.0 parser misaligns —
        the regression this test guards against).
        """
        import io

        nbnd = len(freqs_cm[0])
        lines = [f"&plot nbnd={nbnd:4d}, nks={nks:4d} /"]
        for index in range(nks):
            lines.append(f"{index * 0.1:.6f} 0.0 0.0 0.0")
            lines.append("  ".join(f"{f:.6f}" for f in freqs_cm[index]))
        folder = orm.FolderData()
        folder.base.repository.put_object_from_filelike(
            io.BytesIO(("\n".join(lines) + "\n").encode()), "phonon_frequencies.dat"
        )
        folder.store()
        return folder

    def _dos_node(self):
        dos = orm.XyData()
        x_cm = np.array([0.0, 33.356, 66.713])  # 0 / 1 / 2 THz in cm⁻¹
        dos.set_x(x_cm, "frequency", "cm^(-1)")
        dos.set_y(np.array([1.0, 2.0, 3.0]), "dos", "states * cm")
        dos.store()
        return dos

    def test_gather_outputs(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.band_kpoints = _path_kpoints()
        process.ctx.scf = SimpleNamespace(pk=1)
        process.ctx.ph_run = SimpleNamespace(pk=2)
        fc = orm.ArrayData()
        fc.store()
        process.ctx.q2r_run = SimpleNamespace(
            pk=3, is_finished_ok=True, outputs=SimpleNamespace(force_constants=fc)
        )
        # 5 q-points x 1 band; cm⁻¹ values that map to [-1..3] THz.
        invcm = 0.0299792458
        freqs_cm = [[value / invcm] for value in [-1.0, 0.0, 1.0, 2.0, 3.0]]
        retrieved = self._frequencies_folder(freqs_cm, nks=5)
        process.ctx.matdyn_band = SimpleNamespace(
            is_finished_ok=True, pk=41, uuid="bbbb-1111",
            outputs=SimpleNamespace(retrieved=retrieved),
        )
        process.ctx.matdyn_dos = SimpleNamespace(
            is_finished_ok=True, pk=42, uuid="bbbb-2222",
            outputs=SimpleNamespace(output_phonon_dos=self._dos_node()),
        )

        assert process.gather_results() is None
        outputs = getattr(process, "_outputs", {})

        bands = outputs["phonon_bands"]
        assert bands.base.attributes.get("labels") == [[0, "Γ"], [4, "H"]]
        np.testing.assert_allclose(
            np.asarray(bands.get_bands()).ravel(), [-1.0, 0.0, 1.0, 2.0, 3.0],
            atol=1e-6,
        )

        # DOS rescaled cm⁻¹ → THz (x down, y up to preserve the integral).
        dos = outputs["total_phonon_dos"]
        x_thz = np.asarray(dos.get_x()[1])
        (_name, y, _units), = dos.get_y()
        np.testing.assert_allclose(x_thz, [0.0, 1.0, 2.0], atol=1e-3)
        np.testing.assert_allclose(
            np.asarray(y), [33.356, 66.713, 100.069], atol=1e-2
        )

        para = outputs["output_parameters"].get_dict()
        assert para["workflow"] == "phonopy"
        assert para["backend"] == "qe"
        assert para["method"] == "dfpt"
        assert para["qpoints_mesh"] == [4, 4, 4]
        assert para["band_labels"] == ["Γ", "H"]
        assert para["n_imaginary_modes"] == 1
        assert para["frequency_min_thz"] == pytest.approx(-1.0)
        assert para["phonopy_pk"] == 41
        assert para["matdyn_band_pk"] == 41
        assert para["matdyn_dos_pk"] == 42

    def test_gather_missing_band_file(self, aiida_profile_clean, aiida_localhost):
        """No frequency file in the retrieved folder → ERROR_PARSER
        (previously the plugin BandsData was trusted, which silently held
        the misaligned rolling-zero data)."""
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.band_kpoints = _path_kpoints()
        empty = orm.FolderData()
        empty.store()
        process.ctx.matdyn_band = SimpleNamespace(
            is_finished_ok=True, outputs=SimpleNamespace(retrieved=empty)
        )
        assert process.gather_results() == QePhonopyWorkChain.exit_codes.ERROR_PARSER

    def test_gather_missing_band(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.ctx.band_kpoints = _path_kpoints()
        process.ctx.matdyn_band = SimpleNamespace(
            is_finished_ok=True, outputs=SimpleNamespace()
        )
        assert process.gather_results() == QePhonopyWorkChain.exit_codes.ERROR_PARSER


class TestAdapter:
    def test_inputs_and_spec(self, aiida_profile, aiida_localhost, monkeypatch):
        from aiida_uranium_workflow.input_builders import phonopy as phonopy_mod
        from aiida_uranium_workflow.input_builders.phonopy.qe import (
            QePhonopyAdapter,
        )

        labels = _codes(aiida_localhost, prefix="qe_ph_adapter")
        upf = _fake_upf()

        class _FakeFamily:
            def get_pseudos(self, structure=None, elements=None):
                return {"U": upf}

            def get_recommended_cutoffs(self, structure=None, unit=None):
                return (60.0, 480.0)

        monkeypatch.setattr(phonopy_mod.qe, "load_group", lambda name: _FakeFamily())

        adapter = QePhonopyAdapter(
            code_label=labels["pw"],
            software_params={
                "pw": {"parameters": {"SYSTEM": {"ecutwfc": 99}}},
                "kpoints_mesh": [7, 7, 7],
                "pseudo_family": "test_qe_ph_family",
            },
            metadata={
                "options": {
                    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}
                }
            },
            workflow_data={
                "phonopy": {
                    "qe": {
                        "qpoints_mesh": [3, 3, 3],
                        "ph": {"INPUTPH": {"tr2_ph": 1.0e-16}},
                        "q2r": {"INPUT": {"zasr": "crystal"}},
                        "dos": True,
                        "dos_mesh": [11, 11, 11],
                    }
                }
            },
            extra_codes={
                "ph": labels["ph"],
                "q2r": labels["q2r"],
                "matdyn": labels["matdyn"],
            },
        )
        structure = _structure()
        inputs = adapter._build_workchain_inputs(structure)

        assert inputs["structure"] is structure
        pw = inputs["base"]["pw"]
        assert pw["code"].label.startswith("qe_ph_adapter_pw")
        assert pw["pseudos"]["U"] is upf
        # Recommended cutoffs override the preset reference.
        system = pw["parameters"].get_dict()["SYSTEM"]
        assert system["ecutwfc"] == 60.0
        assert system["ecutrho"] == 480.0
        assert pw["parameters"].get_dict()["CONTROL"]["calculation"] == "scf"
        assert inputs["base"]["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]

        # DFPT codes + knobs.
        assert inputs["ph_code"].label.startswith("qe_ph_adapter_ph")
        assert inputs["q2r_code"].label.startswith("qe_ph_adapter_q2r")
        assert inputs["matdyn_code"].label.startswith("qe_ph_adapter_matdyn")
        inputph = inputs["ph_parameters"].get_dict()["INPUTPH"]
        assert inputph["tr2_ph"] == 1.0e-16  # preset override
        assert inputph["alpha_mix"] == 0.4  # default survives
        assert inputph["epsil"] is False  # yeesuan-validated default
        assert inputph["asr"] is True  # yeesuan-validated default
        assert inputs["qpoints_mesh"].get_list() == [3, 3, 3]
        assert inputs["q2r_parameters"].get_dict() == {"INPUT": {"zasr": "crystal"}}
        assert inputs["dos"].value is True
        assert inputs["dos_mesh"].get_list() == [11, 11, 11]
        assert "options" in inputs  # scheduler options forwarded

        # adapt() resolves the local class; spec validation passes.
        adapted = adapter.adapt(structure)
        assert adapted.workchain_cls is QePhonopyWorkChain
        err = QePhonopyWorkChain.spec().inputs.validate(adapted.inputs)
        assert err is None, f"spec validation failed: {err}"

    def test_missing_chain_codes_raise(self, aiida_profile, aiida_localhost):
        from aiida_uranium_workflow.input_builders.phonopy.qe import (
            QePhonopyAdapter,
        )

        labels = _codes(aiida_localhost, prefix="qe_ph_nocode")
        adapter = QePhonopyAdapter(
            code_label=labels["pw"],
            software_params={"pw": {"parameters": {}}},
            metadata={},
            workflow_data={"phonopy": {}},
            extra_codes={"ph": labels["ph"]},  # q2r / matdyn missing
        )
        with pytest.raises(ValueError, match="q2r"):
            adapter._build_workchain_inputs(_structure())
