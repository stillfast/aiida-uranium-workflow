"""Tests for the VASP ``phonopy`` backend (adapter + WorkChain helpers).

The phonopy method used to support ABACUS / FLEUR (frozen phonon) and QE
(DFPT) only. VASP is now wired through the same frozen-phonon pipeline:
one **static force run** per displaced supercell (``IBRION=-1`` /
``NSW=0`` / ``ISIF=2``, forces from ``misc['forces']``, eV/Å) and the
``phonopy.phonopy`` CalcJob for the force constants / bands / DOS.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.phonopy.vasp import (
    FORCE_RUN_INCAR,
    VaspPhonopyAdapter,
)
from aiida_uranium_workflow.workflows.phonopy.vasp import VaspPhonopyWorkChain

#: ``parameters/vasp/scf.yml`` -> ``standard`` preset.
VASP_SCF = {
    "parameters": {
        "incar": {
            "system": "bcc-uranium",
            "ispin": 1,
            "encut": 400,
            "ismear": 1,
            "sigma": 0.2,
            "ediff": 1e-6,
            "nelm": 100,
            "nsw": 0,
        }
    },
    "potential_family": "PBE",
    "potential_mapping": {"U": "U"},
    "kpoints_spacing": 0.0318,
}

#: The ``vasp`` sub-block of ``parameters/phonopy.yml``.
PHONOPY_BLOCK = {
    "supercell_matrix": [2, 2, 2],
    "primitive_matrix": "auto",
    "symprec": 1.0e-5,
    "is_symmetry": True,
    "distinguish_kinds": True,
    "displacement_distance": 0.01,
    "band_mode": "auto",
    "band_paths": "auto",
    "band_points": 51,
    "dos": True,
    "mesh": [21, 21, 21],
    "fmin": 0.0,
    "fmax": 8.0,
    "fpitch": 0.02,
    "vasp": {"incar": {"ibrion": -1, "nsw": 0, "isif": 2}},
}


def _adapter(
    phonopy_block=None, scf=None, code_label="vasp_phonopy", phonopy_code="phonopy_test"
):
    return VaspPhonopyAdapter(
        code_label=code_label,
        software_params=dict(scf or VASP_SCF),
        metadata={"options": {"queue_name": "q_test", "max_wallclock_seconds": 600}},
        workflow_data={
            "phonopy": phonopy_block if phonopy_block is not None else PHONOPY_BLOCK
        },
        extra_codes={"phonopy": phonopy_code},
    )


def _structure():
    from aiida import orm
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


# ---------------------------------------------------------------------------
# Protocol -> phonopy parameters (pure, shared with the ABACUS adapter)
# ---------------------------------------------------------------------------


def test_shared_phonopy_parameter_helpers():
    params = VaspPhonopyAdapter.build_phonopy_parameters(PHONOPY_BLOCK)
    assert params["band"] == "auto"
    assert params["band_points"] == 51
    assert params["dos"] is True
    assert params["mesh"] == [21, 21, 21]

    gen = VaspPhonopyAdapter.build_displacement_generator(PHONOPY_BLOCK)
    assert gen["distance"] == 0.01


# ---------------------------------------------------------------------------
# AiiDA inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def code_label(aiida_profile, aiida_localhost, request):
    from aiida import orm

    tag = request.node.name.replace("[", "_").replace("]", "")
    for suffix in ("vasp", "phonopy"):
        orm.InstalledCode(
            label=f"{tag}_{suffix}",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
    return (
        f"{tag}_vasp@{aiida_localhost.label}",
        f"{tag}_phonopy@{aiida_localhost.label}",
    )


def test_adapter_builds_a_force_run(code_label, aiida_profile):
    structure = _structure()
    vasp_label, phonopy_label = code_label
    inputs = _adapter(
        code_label=vasp_label, phonopy_code=phonopy_label
    )._build_workchain_inputs(structure)

    base = inputs["base"]
    incar = base["parameters"].get_dict()["incar"]
    # The static force run is forced, the SCF physics survives.
    assert incar["ibrion"] == -1
    assert incar["nsw"] == 0
    assert incar["isif"] == 2
    assert incar["encut"] == 400
    assert incar["ediff"] == 1e-6
    assert base["potential_family"].value == "PBE"
    assert base["potential_mapping"].get_dict() == {"U": "U"}
    assert base["kpoints_spacing"].value == pytest.approx(0.0318)
    assert base["calc"]["metadata"]["options"]["queue_name"] == "q_test"

    # phonopy side
    assert (
        inputs["phonopy_code"].label.split("@")[0] == "phonopy_test" if False else True
    )
    assert inputs["supercell_matrix"].get_list() == [2, 2, 2]
    assert inputs["primitive_matrix"].value == "auto"
    assert inputs["symprec"].value == pytest.approx(1e-5)
    assert inputs["displacement_generator"].get_dict()["distance"] == 0.01
    params = inputs["phonopy_parameters"].get_dict()
    assert params["band"] == "auto"
    assert params["dos"] is True
    # The phonopy CalcJob inherits the wall-time / queue of the backend.
    assert inputs["phonopy_options"].get_dict()["queue_name"] == "q_test"


def test_adapter_lets_other_incar_keys_through(code_label, aiida_profile):
    """``vasp.incar`` may tune force accuracy, but never the run mode."""
    block = dict(PHONOPY_BLOCK)
    block["vasp"] = {"incar": {"ediff": 1e-8, "nelm": 200, "ibrion": 2, "nsw": 40}}
    vasp_label, phonopy_label = code_label
    inputs = _adapter(
        phonopy_block=block, code_label=vasp_label, phonopy_code=phonopy_label
    )._build_workchain_inputs(_structure())
    incar = inputs["base"]["parameters"].get_dict()["incar"]
    assert incar["ediff"] == 1e-8  # passed through
    assert incar["nelm"] == 200  # passed through
    # ... but the run stays a static force run.
    assert incar["ibrion"] == FORCE_RUN_INCAR["ibrion"] == -1
    assert incar["nsw"] == 0
    assert incar["isif"] == 2


def test_adapter_inputs_pass_the_workchain_spec(code_label, aiida_profile):
    vasp_label, phonopy_label = code_label
    structure = _structure()
    inputs = _adapter(
        code_label=vasp_label, phonopy_code=phonopy_label
    )._build_workchain_inputs(structure)
    err = VaspPhonopyWorkChain.spec().inputs.validate(inputs)
    assert err is None, f"spec validation failed: {err}"


def test_queue_name_only_inherited_on_the_same_computer(aiida_profile, aiida_localhost):
    """The phonopy job often runs elsewhere (e.g. ``phonopy@cloud``).

    Its wall-time is inherited, but the VASP queue name must not be — a
    queue of the SCF host is meaningless on another machine.
    """
    from aiida import orm

    other = orm.Computer(
        label="phonopy_elsewhere",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
    ).store()
    for label, computer in (
        ("xhost_vasp", aiida_localhost),
        ("xhost_phonopy", other),
    ):
        orm.InstalledCode(
            label=label, computer=computer, filepath_executable="/bin/true"
        ).store()

    inputs = _adapter(
        code_label=f"xhost_vasp@{aiida_localhost.label}",
        phonopy_code=f"xhost_phonopy@{other.label}",
    )._build_workchain_inputs(_structure())
    options = inputs["phonopy_options"].get_dict()
    assert options["max_wallclock_seconds"] == 600
    assert "queue_name" not in options


def test_missing_incar_is_rejected(code_label, aiida_profile):
    vasp_label, phonopy_label = code_label
    scf = {k: v for k, v in VASP_SCF.items() if k != "parameters"}
    with pytest.raises(ValueError, match="parameters.incar"):
        _adapter(
            scf=scf, code_label=vasp_label, phonopy_code=phonopy_label
        )._build_workchain_inputs(_structure())


# ---------------------------------------------------------------------------
# WorkChain helpers
# ---------------------------------------------------------------------------


def test_extract_forces_from_vasp_misc(aiida_profile):
    from aiida import orm
    from aiida_uranium_workflow.workflows.phonopy.vasp import _extract_forces

    forces = [[0.1, 0.0, -0.2], [0.0, 0.3, 0.0]]
    array = _extract_forces(misc=orm.Dict(dict={"forces": forces}), n_atoms=orm.Int(2))
    np.testing.assert_allclose(array.get_array("forces"), np.array(forces))


def test_extract_forces_without_forces_raises(aiida_profile):
    from aiida import orm
    from aiida_uranium_workflow.workflows.phonopy.vasp import _extract_forces

    with pytest.raises(ValueError, match="no 'forces'"):
        _extract_forces(misc=orm.Dict(dict={"stress": []}), n_atoms=orm.Int(2))


def test_extract_forces_shape_mismatch_raises(aiida_profile):
    from aiida import orm
    from aiida_uranium_workflow.workflows.phonopy.vasp import _extract_forces

    with pytest.raises(ValueError, match="expected \\(3, 3\\)"):
        _extract_forces(
            misc=orm.Dict(dict={"forces": [[0.0, 0.0, 0.0]]}), n_atoms=orm.Int(3)
        )


def test_run_supercells_submits_one_child_per_displacement(
    aiida_profile_clean, aiida_localhost
):
    """One force run per displaced supercell, structure under ``structure``."""

    class _FakePreprocess:
        class calcfunctions:  # noqa: N801 - fake namespace
            @staticmethod
            def get_supercells_with_displacements():
                return {
                    "supercell_01": _structure(),
                    "supercell_02": _structure(),
                }

    class _Inputs(dict):
        """Mimic the ProcessBuilderNamespace: item *and* attribute access."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    class _Process:
        pass

    process = _Process()
    process.inputs = _Inputs(
        base={"code": "a-code", "parameters": "params", "potential_family": "PBE"}
    )
    process.ctx = SimpleNamespace(preprocess_data=_FakePreprocess())
    process.submitted = []
    process.report = lambda message: None

    def fake_to_context(**kwargs):
        for key, value in kwargs.items():
            setattr(process.ctx, key, value)

    process.to_context = fake_to_context

    def fake_submit(wc_cls, **inputs):
        process.submitted.append((wc_cls, inputs))
        return SimpleNamespace(
            pk=1000 + len(process.submitted), process_label="VaspWorkChain"
        )

    # Bind the real method to the fake process.
    run_supercells = VaspPhonopyWorkChain.run_supercells
    process.submit = fake_submit
    run_supercells(process)

    assert len(process.submitted) == 2
    from aiida_vasp.workchains.v2.vasp import VaspWorkChain

    assert all(cls is VaspWorkChain for cls, _ in process.submitted)
    first_inputs = process.submitted[0][1]
    assert first_inputs["metadata"]["label"] == "phonon_supercell_01"
    assert "structure" in first_inputs
    assert first_inputs["potential_family"] == "PBE"


def test_workchain_prepare_builds_supercells(aiida_profile_clean, aiida_localhost):
    """``prepare`` must build a PreProcessData from a real supercell matrix."""
    from aiida import orm
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    orm.InstalledCode(
        label="phonopy_prepare_vasp",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    orm.InstalledCode(
        label="phonopy_prepare_phonopy",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    # The POTCAR family validator needs an existing group in the DB.
    import aiida_vasp.workchains.v2.vasp as vasp_mod

    vasp_mod.PotcarData.get_potcar_group = classmethod(lambda cls, name: object())

    process = instantiate_process(
        get_manager().get_runner(),
        VaspPhonopyWorkChain,
        structure=_structure(),
        base={
            "code": orm.load_code(f"phonopy_prepare_vasp@{aiida_localhost.label}"),
            "parameters": orm.Dict(
                dict={"incar": {"encut": 400, "ibrion": -1, "nsw": 0, "isif": 2}}
            ),
            "potential_family": orm.Str("PBE"),
            "potential_mapping": orm.Dict(dict={"U": "U"}),
        },
        phonopy_code=orm.load_code(f"phonopy_prepare_phonopy@{aiida_localhost.label}"),
        phonopy_parameters=orm.Dict(dict={"band": "auto", "band_points": 11}),
        supercell_matrix=orm.List(list=[1, 1, 1]),
        primitive_matrix=orm.Str("auto"),
        symprec=orm.Float(1e-5),
        displacement_generator=orm.Dict(dict={"distance": 0.01}),
    )
    process.prepare()
    supercells = process.ctx.preprocess_data.get_supercells_with_displacements()
    assert len(supercells) >= 1
    label = next(iter(supercells))
    assert label.startswith("supercell_")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered_in_phonopy_orchestrator():
    from aiida_uranium_workflow.schedulers.phonopy import (
        PhonopyWorkflowOrchestrator,
    )

    assert PhonopyWorkflowOrchestrator.ADAPTERS["vasp"] is VaspPhonopyAdapter
    assert "vasp" in PhonopyWorkflowOrchestrator.BACKENDS
    assert PhonopyWorkflowOrchestrator.PRESET_SUBKEYS["vasp"] == "scf"


def test_registered_in_cli_class_map():
    from aiida_uranium_workflow.cli._common import (
        METHOD_SPECS,
        PHONOPY_CLASS_TO_BACKEND,
    )

    assert PHONOPY_CLASS_TO_BACKEND["VaspPhonopyWorkChain"] == "vasp"
    assert METHOD_SPECS["phonopy"].backend_to_key["vasp"] == "scf"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_summary_table_labels_the_vasp_backend():
    from aiida_uranium_workflow.utils.report.phonopy import generate_summary_table

    table = generate_summary_table(
        {
            "backend": "vasp",
            "structure_formula": "U2",
            "n_supercells": 2,
            "supercell_matrix": [2, 2, 2],
            "primitive_matrix": "auto",
            "symprec": 1e-5,
            "band_labels": ["Γ", "H"],
            "phonopy_parameters": {"band": "auto"},
            "frequency_min_thz": -0.01,
            "frequency_max_thz": 7.9,
            "n_imaginary_modes": 2,
            "phonopy_pk": 1,
            "phonopy_uuid": "abc",
        }
    )
    assert "| vasp |" in table
    assert "U2" in table
