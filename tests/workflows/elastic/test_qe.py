"""Tests for the QE (pw.x) elastic workflow and adapter.

QE elastic uses the stress method: 24 deformed structures, each run
through a ``PwBaseWorkChain`` (``CONTROL.calculation`` = 'relax' or
'scf'); the last ionic step's 3×3 stress (GPa) is read from the
``output_trajectory``'s ``stress`` array and fitted after the usual
compression→tension flip (QE, like ABACUS / VASP, reports positive =
compression). The end-to-end gather test below returns a *negative*
bulk modulus if the sign handling is ever regressed.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import numpy as np
import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.elastic.qe import QeElasticWorkChain


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _known_cubic_tensor():
    c11, c12, c44 = 500.0, 200.0, 150.0
    tensor = np.zeros((6, 6))
    for i in range(3):
        tensor[i, i] = c11
        for j in range(3):
            if i != j:
                tensor[i, j] = c12
    tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = c44
    return tensor


def _fake_upf():
    """A real (minimal) UpfData so the pw spec validation passes."""
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
    upf_path = pathlib.Path(tempfile.mkdtemp()) / "U.upf"
    upf_path.write_text(upf_text, encoding="utf-8")
    upf = UpfData(upf_path)
    upf.store()
    return upf


def _instantiate(aiida_profile_clean, aiida_localhost, relax_internal=True):
    computer = aiida_localhost
    orm.InstalledCode(
        label="qe_elastic_test",
        computer=computer,
        filepath_executable="/bin/true",
    ).store()
    runner = get_manager().get_runner()
    return instantiate_process(
        runner,
        QeElasticWorkChain,
        structure=_structure(),
        pw={
            "code": orm.load_code("qe_elastic_test"),
            "structure": _structure(),
            "parameters": orm.Dict(dict={
                "CONTROL": {"calculation": "scf"},
                "SYSTEM": {"ecutwfc": 60, "ecutrho": 480},
            }),
            "pseudos": {"U": _fake_upf()},
        },
        kpoints=orm.KpointsData(),
        norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        relax_internal=orm.Bool(relax_internal),
    )


class TestDefinition:
    def test_spec(self):
        assert "relax_internal" in QeElasticWorkChain.spec().inputs
        assert "norm_strains" in QeElasticWorkChain.spec().inputs
        assert "output_parameters" in QeElasticWorkChain.spec().outputs

    def test_exit_codes(self):
        assert QeElasticWorkChain.exit_codes.SUCCESS.status == 0
        assert QeElasticWorkChain.exit_codes.ERROR_CHILD.status == 300
        assert QeElasticWorkChain.exit_codes.ERROR_PARSER.status == 305

    def test_generate_deformations_24(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.generate_deformations()
        assert len(process.ctx.deformed_structures) == 24


class TestSubmit:
    def test_relax_mode_sets_calculation(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, relax_internal=True)
        process.generate_deformations()
        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label='PwBaseWorkChain')

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.submit_scfs()

        assert len(captured) == 24
        for child in captured[:4]:
            assert child["pw"]["structure"] is not process.inputs.structure
            params = child["pw"]["parameters"].get_dict()
            assert params["CONTROL"]["calculation"] == "relax"
            assert params["CONTROL"]["tstress"] is True  # forced by adapter
            assert params["SYSTEM"]["ecutwfc"] == 60  # preset survives

    def test_clamped_mode_keeps_scf(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, relax_internal=False)
        process.generate_deformations()
        captured = []

        def fake_submit(cls, **inputs):
            captured.append(inputs)
            return SimpleNamespace(pk=1, process_label='PwBaseWorkChain')

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.submit_scfs()

        params = captured[0]["pw"]["parameters"].get_dict()
        assert params["CONTROL"]["calculation"] == "scf"


class TestGather:
    def test_gather_fits_compression_positive_stress(self, aiida_profile_clean, aiida_localhost):
        """QE reports GPa stresses (positive = compression); the gather
        converts them to kbar (×10) and the default flip must recover the
        known *positive* tensor."""
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.generate_deformations()

        tensor = _known_cubic_tensor()

        def voigt_to_full(v):
            # Stress Voigt → 3×3 (shear components are NOT doubled for stress).
            return np.array(
                [[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]]
            )

        for idx, sv in enumerate(process.ctx.strain_voigts):
            sigma6_gpa = tensor @ np.asarray(sv, dtype=float)
            stress3 = voigt_to_full(sigma6_gpa)  # tension positive, GPa
            # QE reports compression positive → negate the tension response.
            stress_gpa = (-stress3)[np.newaxis, :, :]  # (1, 3, 3)
            setattr(
                process.ctx,
                f"scf_{idx}",
                SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(
                        output_trajectory=SimpleNamespace(
                            get_array=lambda *a, s=stress_gpa: s
                        )
                    ),
                ),
            )

        assert process.gather_results() is None
        params = process._outputs["output_parameters"].get_dict()
        assert params["method"] == "stress"
        fitted = np.asarray(params["elastic_tensor_gpa"])
        np.testing.assert_allclose(fitted, tensor, atol=1.0)

    def test_gather_missing_trajectory(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate(aiida_profile_clean, aiida_localhost)
        process.generate_deformations()

        for idx in range(24):
            setattr(
                process.ctx,
                f"scf_{idx}",
                SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(output_trajectory=SimpleNamespace()),
                ),
            )

        # No trajectory output → the child data is unusable → ERROR_CHILD.
        assert (
            process.gather_results() == QeElasticWorkChain.exit_codes.ERROR_CHILD
        )


class TestAdapter:
    def test_inputs(self, aiida_profile, aiida_localhost, monkeypatch):
        from aiida_uranium_workflow.input_builders import elastic as elastic_mod
        from aiida_uranium_workflow.input_builders.elastic.qe import QeElasticAdapter

        computer = aiida_localhost
        orm.InstalledCode(
            label="qe_elastic_adapter",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        upf = _fake_upf()

        class _FakeFamily:
            def get_pseudos(self, structure=None, elements=None):
                return {"U": upf}

            def get_recommended_cutoffs(self, structure=None, unit=None):
                return (60.0, 480.0)

        monkeypatch.setattr(
            elastic_mod.qe, "load_group", lambda name: _FakeFamily()
        )

        adapter = QeElasticAdapter(
            code_label=f"qe_elastic_adapter@{computer.label}",
            software_params={
                "pw": {"parameters": {"SYSTEM": {"ecutwfc": 99}}},
                "kpoints_mesh": [7, 7, 7],
                "pseudo_family": "test_qe_el_family",
            },
            metadata={
                "options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}
            },
            workflow_data={
                "qe": {
                    "norm_strains": [-0.01, 0.01],
                    "shear_strains": [-0.01, 0.01],
                    "relax_internal": False,
                }
            },
            extra_codes={},
        )
        structure = _structure()
        inputs = adapter._build_workchain_inputs(structure)

        assert inputs["structure"] is structure
        assert inputs["pw"]["code"].label == "qe_elastic_adapter"
        assert inputs["pw"]["pseudos"]["U"] is upf
        # Recommended cutoffs override the preset reference.
        system = inputs["pw"]["parameters"].get_dict()["SYSTEM"]
        assert system["ecutwfc"] == 60.0
        assert system["ecutrho"] == 480.0
        # Stress output forced for the elastic stress method.
        control = inputs["pw"]["parameters"].get_dict()["CONTROL"]
        assert control["tstress"] is True
        assert inputs["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]
        assert inputs["norm_strains"].get_list() == [-0.01, 0.01]
        assert inputs["relax_internal"].value is False

        # adapt() resolves the local class; spec validation passes with
        # everything the workflow needs.
        adapted = adapter.adapt(structure)
        assert adapted.workchain_cls is QeElasticWorkChain
        err = QeElasticWorkChain.spec().inputs.validate(adapted.inputs)
        assert err is None, f"spec validation failed: {err}"
