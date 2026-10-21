"""Tests for the VASP elastic workflow and adapter.

The VASP elastic WorkChain follows the ABACUS stress method (24
deformed structures, final-ionic-step 3×3 stress tensor from each
child's ``misc`` output). VASP's "total stress" reports **positive =
compression** — the same convention as ABACUS (verified against real
OUTCAR/vasprun output: a compressed cell shows positive stress) — so
the tensors are fitted through the same compression→tension flip as the
ABACUS path. The end-to-end gather test below would return a *negative*
bulk modulus if that sign handling were ever regressed.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import numpy as np
import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.elastic.vasp import (
    VaspElasticWorkChain,
)

# aiida-vasp's validator requires an installed POTCAR family group in
# the database; tests patch the lookup to accept any name.
import aiida_vasp.workchains.v2.vasp as vasp_mod  # noqa: E402


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _known_cubic_tensor():
    """A known cubic elastic tensor (GPa, Voigt 6×6)."""
    c11, c12, c44 = 500.0, 200.0, 150.0
    tensor = np.zeros((6, 6))
    for i in range(3):
        tensor[i, i] = c11
        for j in range(3):
            if i != j:
                tensor[i, j] = c12
    tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = c44
    return tensor


def _patch_potential_family(monkeypatch):
    monkeypatch.setattr(
        vasp_mod.PotcarData,
        "get_potcar_group",
        classmethod(lambda cls, name: object()),
    )


def _instantiate(
    aiida_profile_clean,
    aiida_localhost,
    monkeypatch,
    relax_internal,
    relax_lattice=False,
    require_lattice_relax=False,
):
    _patch_potential_family(monkeypatch)
    orm.InstalledCode(
        label="elastic_vasp_test",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    runner = get_manager().get_runner()
    inputs = dict(
        structure=_structure(),
        code=orm.load_code("elastic_vasp_test"),
        parameters=orm.Dict(dict={"incar": {"encut": 400, "ismear": 1, "sigma": 0.1}}),
        potential_family=orm.Str("PBE"),
        potential_mapping=orm.Dict(dict={"U": "U"}),
        kpoints=orm.KpointsData(),
        norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        relax_internal=orm.Bool(relax_internal),
        relax_lattice=orm.Bool(relax_lattice),
    )
    if require_lattice_relax:
        inputs["require_lattice_relax"] = orm.Bool(True)
    return instantiate_process(runner, VaspElasticWorkChain, **inputs)


class TestDefinition:
    def test_spec_has_relax_internal(self):
        assert "relax_internal" in VaspElasticWorkChain.spec().inputs
        assert "relax_lattice" in VaspElasticWorkChain.spec().inputs
        assert "relax_settings" in VaspElasticWorkChain.spec().inputs
        assert "norm_strains" in VaspElasticWorkChain.spec().inputs
        assert "shear_strains" in VaspElasticWorkChain.spec().inputs

    def test_exit_codes(self):
        assert VaspElasticWorkChain.exit_codes.SUCCESS.status == 0
        assert VaspElasticWorkChain.exit_codes.ERROR_CHILD.status == 300
        assert VaspElasticWorkChain.exit_codes.ERROR_PARSER.status == 305
        assert VaspElasticWorkChain.exit_codes.ERROR_RELAX.status == 310


class TestStrainedChildren:
    """The strained cells are ``vasp.relax`` children with RelaxOptions.

    The ionic settings of the 24 strained cells are expressed as
    RelaxOptions (``positions`` / ``steps`` / ``force_cutoff`` / …) and
    handed to ``VaspRelaxWorkChain`` — the same vocabulary as
    ``parameters/relax.yml`` — instead of raw INCAR tags. The cell is held
    at the strained lattice (``shape`` / ``volume`` off), so no final
    single-point calculation is run.
    """

    @staticmethod
    def _defaults():
        from aiida_uranium_workflow.workflows.elastic.vasp import (
            _DEFAULT_CHILD_RELAX_SETTINGS,
        )

        return _DEFAULT_CHILD_RELAX_SETTINGS

    def test_default_settings_are_positions_only(self):
        defaults = self._defaults()
        assert defaults["positions"] is True
        assert defaults["shape"] is False and defaults["volume"] is False
        # tight enough for the soft internal modes of α-U (~0.03 eV/Å
        # develops under a ±1 % shear)
        assert defaults["force_cutoff"] <= 0.005
        assert defaults["steps"] >= 100
        assert defaults["convergence_on"] is False

    def test_default_settings_validate_against_the_plugin(self):
        from aiida_vasp.utils.opthold import RelaxOptions

        # RelaxOptions is a pydantic model: raises on a bad/typed field.
        RelaxOptions(**self._defaults())

    def test_incar_tags_are_stripped_for_the_children(self):
        """The preset's relax tags must not override ``relax_settings``."""
        params = VaspElasticWorkChain._relax_child_parameters(
            {"incar": {"encut": 400, "nsw": 0, "ibrion": -1, "isif": 4, "ediffg": -0.1}}
        )
        assert params.get_dict()["incar"] == {"encut": 400}


class TestRelaxChildParameters:
    """The INCAR handed to the ``vasp.relax`` lattice-relaxation child.

    Regression: the SCF preset carries ``nsw: 0``; aiida-vasp applies the
    ``incar`` namespace *after* its relax set-functions, so the tag won over
    ``relax_settings`` — VASP ran a static calculation, omitted the
    ``structure`` output (``omit_structure``) and ``VaspRelaxWorkChain``
    exited 300 before a single strained cell was submitted.
    """

    def test_controlled_tags_are_stripped(self):
        params = VaspElasticWorkChain._relax_child_parameters(
            {
                "incar": {
                    "encut": 400,
                    "nsw": 0,
                    "ibrion": -1,
                    "isif": 4,
                    "ediffg": -0.1,
                }
            }
        )
        incar = params.get_dict()["incar"]
        assert incar == {"encut": 400}

    def test_strip_list_matches_the_relax_adapter(self):
        from aiida_uranium_workflow.input_builders.relax.vasp import (
            RELAX_CONTROLLED_INCAR_TAGS,
        )
        from aiida_uranium_workflow.workflows.elastic.vasp import (
            _RELAX_CONTROLLED_INCAR_TAGS,
        )

        assert tuple(_RELAX_CONTROLLED_INCAR_TAGS) == tuple(RELAX_CONTROLLED_INCAR_TAGS)

    def test_accepts_orm_dict(self):
        params = VaspElasticWorkChain._relax_child_parameters(
            orm.Dict(dict={"incar": {"encut": 400, "nsw": 0}})
        )
        assert params.get_dict()["incar"] == {"encut": 400}

    def test_final_incar_is_the_relaxation(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """End-to-end: the INCAR VASP would receive for the relax child.

        The SCF preset's ``nsw: 0`` must not survive; ``relax_settings``
        (full cell + ions, 60 steps) decides NSW / IBRION / ISIF / EDIFFG.
        """
        from aiida_vasp.assistant.parameters import ParametersMassage

        _patch_potential_family(monkeypatch)
        orm.InstalledCode(
            label="elastic_vasp_test",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        runner = get_manager().get_runner()
        process = instantiate_process(
            runner,
            VaspElasticWorkChain,
            structure=_structure(),
            code=orm.load_code("elastic_vasp_test"),
            parameters=orm.Dict(dict={"incar": {"encut": 400, "nsw": 0}}),
            potential_family=orm.Str("PBE"),
            potential_mapping=orm.Dict(dict={"U": "U"}),
            kpoints=orm.KpointsData(),
            norm_strains=orm.List(list=[-0.01, 0.01]),
            shear_strains=orm.List(list=[-0.01, 0.01]),
            relax_internal=orm.Bool(True),
            relax_lattice=orm.Bool(True),
        )
        calls = []

        def fake_submit(wc_cls, **inputs):
            calls.append(inputs)
            return SimpleNamespace(pk=999, process_label="VaspRelaxWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.initialize()

        # VaspRelaxWorkChain injects the relax settings as parameters['relax']
        # before aiida-vasp massages the INCAR (see ``_init_relax_input_additions``).
        parameters = dict(calls[0]["vasp"]["parameters"].get_dict())
        parameters["relax"] = calls[0]["relax_settings"].get_dict()
        incar = ParametersMassage(parameters).parameters["incar"]
        assert incar["nsw"] == 60  # steps from RelaxOptions, not 0
        assert incar["ibrion"] == 2  # cg
        assert incar["isif"] == 3  # positions + shape + volume
        assert incar["ediffg"] == pytest.approx(-0.03)


class TestGather:
    def test_generate_deformations_24(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()
        assert len(process.ctx.deformed_structures) == 24
        assert len(process.ctx.strain_voigts) == 24

    def test_submit_scfs_submits_24_children(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """One ``vasp.relax`` child per strained cell — 24 total.

        Regression: the workflow used to abort at ``generate_deformations``
        (exit 310) and submitted **zero** children.
        """
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()

        submitted = []

        def fake_submit(wc_cls, **inputs):
            submitted.append((wc_cls, inputs))
            return SimpleNamespace(
                pk=1000 + len(submitted), process_label="VaspRelaxWorkChain"
            )

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.submit_scfs()

        assert len(submitted) == 24
        from aiida_vasp.workchains.v2.relax import VaspRelaxWorkChain

        assert all(cls is VaspRelaxWorkChain for cls, _ in submitted)
        # Each child is a positions-only relax at the strained lattice, with
        # the ionic settings expressed as RelaxOptions (not raw INCAR tags),
        # and the SCF INCAR has the relax-controlled tags stripped.
        first = submitted[0][1]
        settings = first["relax_settings"].get_dict()
        assert settings["positions"] is True
        assert settings["shape"] is False and settings["volume"] is False
        assert settings["force_cutoff"] <= 0.005
        assert first["vasp"]["parameters"].get_dict()["incar"].get("nsw") is None
        assert "structure" not in first["vasp"]
        cell_0 = np.asarray(submitted[0][1]["structure"].cell)
        cell_1 = np.asarray(submitted[1][1]["structure"].cell)
        assert not np.allclose(cell_0, cell_1)

    def test_gather_fits_compression_positive_stress(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """End-to-end sign check: VASP reports positive = compression, so
        the raw kbar stresses (the negated tension-positive response) must
        fit back to the known *positive* tensor through the same
        compression→tension flip as the ABACUS path."""
        from types import SimpleNamespace

        from pymatgen.analysis.elasticity import Stress

        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()

        tensor = _known_cubic_tensor()
        stresses_kbar = []
        for sv in process.ctx.strain_voigts:
            sigma6_gpa = tensor @ np.asarray(sv, dtype=float)
            sigma3 = Stress.from_voigt(sigma6_gpa)
            # VASP reports compression positive → negate the tension
            # response (verified against real output: a compressed cell
            # shows positive stress).
            stresses_kbar.append((-sigma3 * 10.0).tolist())  # GPa -> kbar

        for idx in range(24):
            setattr(
                process.ctx,
                f"scf_{idx}",
                SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(
                        misc=SimpleNamespace(
                            get_dict=lambda s=stresses_kbar[idx]: {"stress": s}
                        )
                    ),
                ),
            )

        exit_code = process.gather_results()
        assert exit_code is None
        params = process._outputs["output_parameters"].get_dict()
        assert params["method"] == "stress"
        fitted = np.asarray(params["elastic_tensor_gpa"])
        np.testing.assert_allclose(fitted, tensor, atol=1.0)
        # No lattice relaxation in this setup: the tensor is fitted on the
        # input cell and the output says so.
        assert params["equilibrium_structure_source"] == "input"
        assert params["lattice_relaxed"] is False

    def test_gather_reports_missing_stress(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        from types import SimpleNamespace

        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()

        for idx in range(24):
            setattr(
                process.ctx,
                f"scf_{idx}",
                SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(misc=SimpleNamespace(get_dict=lambda: {})),
                ),
            )

        assert process.gather_results() == VaspElasticWorkChain.exit_codes.ERROR_CHILD


class TestLatticeRelax:
    """The optional full lattice relaxation (vasp.relax) before the deformations."""

    def test_initialize_skips_when_false(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=False
        )
        calls = []

        def fake_submit(wc_cls, **inputs):
            calls.append(wc_cls)
            return SimpleNamespace(pk=999, process_label="VaspRelaxWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        process.initialize()
        assert calls == []  # no relax submitted
        assert not hasattr(process.ctx, "lattice_relax")

    def test_initialize_submits_vasp_relax(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True
        )
        calls, ctx_calls = [], []

        def fake_submit(wc_cls, **inputs):
            calls.append((wc_cls, inputs))
            return SimpleNamespace(pk=999, process_label="VaspRelaxWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        # ``to_context`` requires a real ProcessNode; record it instead.
        monkeypatch.setattr(process, "to_context", lambda **kw: ctx_calls.append(kw))
        process.initialize()

        assert len(calls) == 1
        wc_cls, relax_inputs = calls[0]
        from aiida_vasp.workchains.v2.relax import VaspRelaxWorkChain

        assert wc_cls is VaspRelaxWorkChain
        # vasp namespace carries the SCF inputs, structure at top level.
        assert relax_inputs["vasp"]["code"].label == "elastic_vasp_test"
        assert relax_inputs["vasp"]["potential_family"].value == "PBE"
        assert "structure" not in relax_inputs["vasp"]
        assert relax_inputs["structure"] is process.inputs.structure
        assert relax_inputs["metadata"]["label"] == "lattice_relax"
        # relax_settings is a required VaspRelaxWorkChain input and is
        # consumed verbatim as its runtime config — the full RelaxOptions
        # defaults must always be sent (perform=True, ISIF=3 dof).
        rs = relax_inputs["relax_settings"].get_dict()
        assert rs["perform"] is True
        assert rs["positions"] is True and rs["volume"] is True and rs["shape"] is True
        assert rs["force_cutoff"] == 0.03
        assert rs["steps"] == 60
        assert rs["algo"] == "cg"
        # The running child is registered in the context under
        # ``lattice_relax`` so the engine waits for it before the next
        # outline step.
        assert list(ctx_calls[0]) == ["lattice_relax"]

    def test_initialize_forwards_relax_settings(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        _patch_potential_family(monkeypatch)
        orm.InstalledCode(
            label="elastic_vasp_test",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        runner = get_manager().get_runner()
        process = instantiate_process(
            runner,
            VaspElasticWorkChain,
            structure=_structure(),
            code=orm.load_code("elastic_vasp_test"),
            parameters=orm.Dict(dict={"incar": {"encut": 400}}),
            potential_family=orm.Str("PBE"),
            potential_mapping=orm.Dict(dict={"U": "U"}),
            kpoints=orm.KpointsData(),
            norm_strains=orm.List(list=[-0.01, 0.01]),
            shear_strains=orm.List(list=[-0.01, 0.01]),
            relax_internal=orm.Bool(True),
            relax_lattice=orm.Bool(True),
            relax_settings=orm.Dict(dict={"force_cutoff": 0.05, "steps": 40}),
        )
        calls = []

        def fake_submit(wc_cls, **inputs):
            calls.append(inputs)
            return SimpleNamespace(pk=999, process_label="VaspRelaxWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        monkeypatch.setattr(process, "to_context", lambda **kw: None)
        process.initialize()
        rs = calls[0]["relax_settings"].get_dict()
        # User keys override the defaults; untouched keys keep the defaults.
        assert rs["force_cutoff"] == 0.05
        assert rs["steps"] == 40
        assert rs["perform"] is True
        assert rs["positions"] is True and rs["shape"] is True and rs["volume"] is True

    def test_equilibrium_structure_falls_back_to_input(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True
        )
        assert process._equilibrium_structure() is process.inputs.structure

    def test_generate_deformations_uses_relaxed_structure(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        from ase.build import bulk

        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True
        )
        # A relaxed structure with a visibly different lattice constant.
        relaxed = orm.StructureData(ase=bulk("U", "bcc", a=4.0))
        process.ctx.lattice_relax = SimpleNamespace(
            is_finished_ok=True,
            outputs={"relax.structure": relaxed},
        )
        assert process._equilibrium_structure() is relaxed

        assert process.generate_deformations() is None
        # First normal strain is ε1 = -0.01: the deformation gradient
        # F = Strain.from_index_amount((0,0), -0.01).get_deformation_matrix()
        # scales the x-direction of the *relaxed* lattice (input lattice
        # is a=3.45, not a=4.0).
        from aiida_uranium_workflow.utils.elastic import generate_deformations

        defs = generate_deformations(
            [-0.01, -0.005, 0.005, 0.01], [-0.01, -0.005, 0.005, 0.01]
        )
        expected = np.asarray(relaxed.cell) @ np.asarray(defs[0][1]).T
        np.testing.assert_allclose(process.ctx.deformed_structures[0].cell, expected)

    def test_generate_deformations_continues_when_relax_failed(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """A failed lattice relaxation must not swallow the whole workflow.

        The 24 strained cells are still computed, from the *input*
        structure, and the result is flagged as not being on the
        equilibrium lattice.
        """
        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True
        )
        process.ctx.lattice_relax = SimpleNamespace(
            pk=1, is_finished_ok=False, exit_status=300, outputs={}
        )
        assert process.generate_deformations() is None
        assert len(process.ctx.deformed_structures) == 24
        assert process.ctx.equilibrium_structure_source == "input"

        from aiida_uranium_workflow.utils.elastic import generate_deformations

        defs = generate_deformations(
            [-0.01, -0.005, 0.005, 0.01], [-0.01, -0.005, 0.005, 0.01]
        )
        expected = np.asarray(process.inputs.structure.cell) @ np.asarray(defs[0][1]).T
        np.testing.assert_allclose(process.ctx.deformed_structures[0].cell, expected)

    def test_generate_deformations_aborts_when_required(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        """``require_lattice_relax=True`` restores the strict behaviour."""
        process = _instantiate(
            aiida_profile_clean,
            aiida_localhost,
            monkeypatch,
            True,
            relax_lattice=True,
            require_lattice_relax=True,
        )
        process.ctx.lattice_relax = SimpleNamespace(
            pk=1, is_finished_ok=False, exit_status=300, outputs={}
        )
        assert (
            process.generate_deformations()
            == VaspElasticWorkChain.exit_codes.ERROR_RELAX
        )
        assert not hasattr(process.ctx, "deformed_structures")

    def test_generate_deformations_records_relaxed_source(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        from ase.build import bulk

        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True
        )
        process.ctx.lattice_relax = SimpleNamespace(
            is_finished_ok=True,
            outputs={"relax.structure": orm.StructureData(ase=bulk("U", "bcc", a=4.0))},
        )
        assert process.generate_deformations() is None
        assert process.ctx.equilibrium_structure_source == "lattice_relax"

    def test_generate_deformations_no_relax_when_false(
        self, aiida_profile_clean, aiida_localhost, monkeypatch
    ):
        process = _instantiate(
            aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=False
        )
        assert process.generate_deformations() is None
        from aiida_uranium_workflow.utils.elastic import generate_deformations

        defs = generate_deformations(
            [-0.01, -0.005, 0.005, 0.01], [-0.01, -0.005, 0.005, 0.01]
        )
        expected = np.asarray(process.inputs.structure.cell) @ np.asarray(defs[0][1]).T
        np.testing.assert_allclose(process.ctx.deformed_structures[0].cell, expected)


class TestFitSignConvention:
    """The ``compression_positive`` switch of the shared fit helper."""

    @staticmethod
    def _full_24_strain_stress(tensor):
        """The Materials Project 24-state (strain, stress) set (GPa → kbar).

        Returns ``(voigts, stresses_kbar)`` with the stresses in the
        tension-positive convention (``sigma = C @ strain``); negate for
        the ABACUS / VASP compression-positive convention.
        """
        from pymatgen.analysis.elasticity import Stress

        voigts, stresses_kbar = [], []
        for idx in [(0, 0), (1, 1), (2, 2)]:
            for a in [-0.01, -0.005, 0.005, 0.01]:
                sv = np.zeros(6)
                sv[idx[0]] = a
                voigts.append(sv)
        for idx in [(0, 1), (0, 2), (1, 2)]:
            for a in [-0.01, -0.005, 0.005, 0.01]:
                sv = np.zeros(6)
                sv[{(0, 1): 5, (0, 2): 4, (1, 2): 3}[idx]] = 2 * a
                voigts.append(sv)
        for sv in voigts:
            sigma6_gpa = tensor @ np.asarray(sv, dtype=float)
            stresses_kbar.append((Stress.from_voigt(sigma6_gpa) * 10.0).tolist())
        return voigts, stresses_kbar

    def test_compression_positive_default_flip(self):
        """ABACUS / VASP report positive = compression; the default
        ``compression_positive=True`` negates the raw kbar data and
        recovers the known positive tensor."""
        from aiida_uranium_workflow.utils.elastic import fit_elastic_from_stress

        tensor = _known_cubic_tensor()
        voigts, stresses_kbar = self._full_24_strain_stress(tensor)

        # compression-positive data (as ABACUS and VASP report it).
        res = fit_elastic_from_stress(
            voigts,
            [-np.asarray(s) for s in stresses_kbar],
            crystal_system="cubic",
        )
        np.testing.assert_allclose(res["elastic_tensor_gpa"], tensor, atol=1.0)

    def test_tension_positive_needs_no_flip(self):
        """Only data already tension-positive uses ``compression_positive=False``."""
        from aiida_uranium_workflow.utils.elastic import fit_elastic_from_stress

        tensor = _known_cubic_tensor()
        voigts, stresses_kbar = self._full_24_strain_stress(tensor)

        res = fit_elastic_from_stress(
            voigts,
            stresses_kbar,
            crystal_system="cubic",
            compression_positive=False,
        )
        np.testing.assert_allclose(res["elastic_tensor_gpa"], tensor, atol=1.0)


class TestAdapter:
    def test_vasp_inputs(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        computer = aiida_localhost
        orm.InstalledCode(
            label="elastic_vasp_adapter",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = VaspElasticAdapter(
            code_label=f"elastic_vasp_adapter@{computer.label}",
            software_params={
                "parameters": {"incar": {"encut": 400, "ismear": 1, "sigma": 0.1}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
                "kpoints_mesh": [7, 7, 7],
            },
            metadata={
                "options": {
                    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}
                }
            },
            workflow_data={
                "vasp": {
                    "norm_strains": [-0.01, 0.01],
                    "shear_strains": [-0.01, 0.01],
                    "relax_internal": False,
                    "relax_lattice": False,
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)

        assert inputs["code"].label == "elastic_vasp_adapter"
        assert inputs["structure"] is structure
        assert inputs["parameters"].get_dict()["incar"]["encut"] == 400
        assert inputs["potential_family"].value == "PBE"
        assert inputs["potential_mapping"].get_dict() == {"U": "U"}
        assert inputs["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]
        assert inputs["norm_strains"].get_list() == [-0.01, 0.01]
        assert inputs["shear_strains"].get_list() == [-0.01, 0.01]
        assert inputs["relax_internal"].value is False
        assert inputs["relax_lattice"].value is False
        # Strict abort is off by default: a failed lattice relax still
        # computes the 24 strained cells on the input structure.
        assert inputs["require_lattice_relax"].value is False
        assert "relax_settings" not in inputs
        assert "child_relax_settings" not in inputs  # not given in this preset

    def test_child_relax_forwarded(self, aiida_profile, aiida_localhost):
        """``vasp.child_relax`` (relax.yml vocabulary) -> RelaxOptions Dict."""
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        orm.InstalledCode(
            label="elastic_vasp_child_relax",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = VaspElasticAdapter(
            code_label=f"elastic_vasp_child_relax@{aiida_localhost.label}",
            software_params={
                "parameters": {"incar": {}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
            },
            metadata={},
            workflow_data={
                "vasp": {
                    "relax_lattice": False,
                    "child_relax": {
                        "relax_type": "positions",
                        "relax_method": "cg",
                        "max_ionic_steps": 100,
                        "force_cutoff": 0.005,
                        "meta_convergence": False,
                    },
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        settings = inputs["child_relax_settings"].get_dict()
        # positions only, tight force criterion, single relax run
        assert settings["positions"] is True
        assert settings["shape"] is False and settings["volume"] is False
        assert settings["steps"] == 100
        assert settings["force_cutoff"] == pytest.approx(0.005)
        assert settings["algo"] == "cg"
        assert settings["convergence_on"] is False

    def test_default_strains_and_relax_internal(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        orm.InstalledCode(
            label="elastic_vasp_defaults",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = VaspElasticAdapter(
            code_label=f"elastic_vasp_defaults@{aiida_localhost.label}",
            software_params={
                "parameters": {"incar": {}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
            },
            metadata={},
            workflow_data={"vasp": {}},
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["norm_strains"].get_list() == [-0.010, -0.005, 0.005, 0.010]
        assert inputs["shear_strains"].get_list() == [-0.010, -0.005, 0.005, 0.010]
        assert inputs["relax_internal"].value is True
        # Lattice relaxation defaults on (the ABACUS example's
        # ``prepare_elastic.sh`` cell-relax counterpart).
        assert inputs["relax_lattice"].value is True
        assert inputs["require_lattice_relax"].value is False
        assert "relax_settings" not in inputs
        assert "child_relax_settings" not in inputs  # not given in this preset

    def test_require_lattice_relax_forwarded(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        orm.InstalledCode(
            label="elastic_vasp_require_relax",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = VaspElasticAdapter(
            code_label=f"elastic_vasp_require_relax@{aiida_localhost.label}",
            software_params={
                "parameters": {"incar": {}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
            },
            metadata={},
            workflow_data={"vasp": {"require_lattice_relax": True}},
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["require_lattice_relax"].value is True

    def test_relax_settings_forwarded(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        orm.InstalledCode(
            label="elastic_vasp_rsettings",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = VaspElasticAdapter(
            code_label=f"elastic_vasp_rsettings@{aiida_localhost.label}",
            software_params={
                "parameters": {"incar": {}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
            },
            metadata={},
            workflow_data={
                "vasp": {
                    "relax_lattice": True,
                    "lattice_relax": {
                        "relax_type": "positions_cell",
                        "force_cutoff": 0.05,
                        "max_ionic_steps": 40,
                    },
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["relax_lattice"].value is True
        settings = inputs["relax_settings"].get_dict()
        assert settings["force_cutoff"] == 0.05
        assert settings["steps"] == 40


class TestRelaxSettingsFromProtocol:
    """The elastic protocol's vasp block uses the relax.yml vocabulary.

    elastic.yml spells out the relaxation of its ``vasp.relax`` children in
    the *same* keys as ``parameters/relax.yml`` (relax_type / relax_method /
    max_ionic_steps / force_cutoff / meta_convergence / volume_convergence /
    max_meta_convergence_iterations), one sub-block per role:
    ``lattice_relax`` (ISIF=3 pre-relaxation) and ``child_relax`` (the 24
    strained cells) — no duplicated top-level keys.
    """

    def _adapter(self, aiida_localhost, vasp_block, label):
        from aiida_uranium_workflow.input_builders.elastic.vasp import (
            VaspElasticAdapter,
        )

        orm.InstalledCode(
            label=label,
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        return VaspElasticAdapter(
            code_label=f"{label}@{aiida_localhost.label}",
            software_params={
                "parameters": {"incar": {"encut": 400}},
                "potential_family": "PBE",
                "potential_mapping": {"U": "U"},
                "kpoints_mesh": [7, 7, 7],
            },
            metadata={},
            workflow_data={"vasp": vasp_block},
            extra_codes={},
        )

    def test_relax_type_and_method_are_translated(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        adapter = self._adapter(
            aiida_localhost,
            {
                "relax_lattice": True,
                "lattice_relax": {
                    "relax_type": "positions_cell",
                    "relax_method": "cg",
                    "max_ionic_steps": 50,
                    "force_cutoff": 0.05,
                    "meta_convergence": True,
                    "volume_convergence": 0.01,
                    "max_meta_convergence_iterations": 5,
                },
            },
            "elastic_relax_spec",
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        settings = adapter._build_workchain_inputs(structure)[
            "relax_settings"
        ].get_dict()

        # positions_cell -> ISIF=3 degrees of freedom
        assert settings["positions"] is True
        assert settings["shape"] is True
        assert settings["volume"] is True
        assert settings["algo"] == "cg"  # IBRION=2
        assert settings["steps"] == 50  # NSW
        assert settings["force_cutoff"] == 0.05  # EDIFFG = -0.05
        assert settings["convergence_on"] is True
        assert settings["convergence_max_iterations"] == 5
        # relative threshold × V0 -> absolute Å³
        assert settings["convergence_volume"] == pytest.approx(
            0.01 * structure.get_cell_volume(), rel=1e-9
        )
        # the dict is complete (VaspRelaxWorkChain reads .perform etc.)
        assert settings["perform"] is True

    def test_raw_relax_settings_still_override(self, aiida_profile, aiida_localhost):
        """A raw ``relax_settings`` sub-dict overrides the translated keys."""
        from ase.build import bulk

        adapter = self._adapter(
            aiida_localhost,
            {
                "relax_lattice": True,
                "lattice_relax": {
                    "relax_type": "positions_cell",
                    "relax_settings": {"steps": 20},
                },
            },
            "elastic_relax_raw",
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        settings = adapter._build_workchain_inputs(structure)[
            "relax_settings"
        ].get_dict()
        assert settings["steps"] == 20

    def test_no_relax_spec_means_no_override(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        adapter = self._adapter(
            aiida_localhost, {"relax_lattice": True}, "elastic_relax_none"
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        assert "relax_settings" not in adapter._build_workchain_inputs(structure)


class TestAbacusRelaxSettings:
    """The abacus block can override the internal-coordinate relaxation."""

    def test_relax_settings_forwarded(self, aiida_profile, aiida_localhost):
        from ase.build import bulk

        from aiida_uranium_workflow.input_builders.elastic.abacus import (
            AbacusElasticAdapter,
        )

        orm.InstalledCode(
            label="elastic_abacus_relax",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = AbacusElasticAdapter(
            code_label=f"elastic_abacus_relax@{aiida_localhost.label}",
            software_params={
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints_mesh": [4, 4, 4],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "abacus": {
                    "norm_strains": [-0.01, 0.01],
                    "shear_strains": [-0.01, 0.01],
                    "relax_internal": True,
                    "child_relax": {
                        "relax_type": "positions",
                        "relax_method": "cg_bfgs",
                        "max_ionic_steps": 40,
                        "force_cutoff": 0.03,
                        "stress_cutoff": 1.0,
                    },
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        settings = inputs["child_relax_settings"].get_dict()
        assert settings["relax_method"] == "cg_bfgs"
        assert settings["max_ionic_steps"] == 40
        assert settings["force_cutoff"] == 0.03
        # the lattice pre-relaxation is off unless the protocol asks for it
        assert inputs["relax_lattice"].value is False
        assert "relax_settings" not in inputs
