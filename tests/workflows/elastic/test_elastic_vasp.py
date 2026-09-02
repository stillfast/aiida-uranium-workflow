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
    _CLAMPED_INCAR,
    _RELAX_INCAR,
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
    aiida_profile_clean, aiida_localhost, monkeypatch, relax_internal, relax_lattice=False
):
    _patch_potential_family(monkeypatch)
    orm.InstalledCode(
        label="elastic_vasp_test",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    runner = get_manager().get_runner()
    return instantiate_process(
        runner,
        VaspElasticWorkChain,
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


class TestRelaxParameters:
    """The INCAR injected for each relax mode."""

    def test_relax_internal_incar(self):
        params = VaspElasticWorkChain._relaxed_parameters(
            {"incar": {"encut": 400}}, relax_internal=True
        )
        incar = params["incar"]
        for key, value in _RELAX_INCAR.items():
            assert incar[key] == value  # ISIF=2 / IBRION=2 / NSW=50
        assert incar["ediffg"] == -0.02  # force criterion injected
        assert incar["encut"] == 400  # preset survives

    def test_relax_internal_keeps_preset_ediffg(self):
        params = VaspElasticWorkChain._relaxed_parameters(
            {"incar": {"ediffg": -0.05}}, relax_internal=True
        )
        assert params["incar"]["ediffg"] == -0.05

    def test_clamped_incar(self):
        params = VaspElasticWorkChain._relaxed_parameters(
            {"incar": {"encut": 400}}, relax_internal=False
        )
        incar = params["incar"]
        for key, value in _CLAMPED_INCAR.items():
            assert incar[key] == value  # ISIF=4 / IBRION=-1 / NSW=0
        assert "ediffg" not in incar  # single point: no force criterion
        assert incar["encut"] == 400

    def test_accepts_orm_dict(self):
        params = VaspElasticWorkChain._relaxed_parameters(
            orm.Dict(dict={"incar": {"encut": 400}}), relax_internal=True
        )
        assert params["incar"]["isif"] == 2


class TestGather:
    def test_generate_deformations_24(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()
        assert len(process.ctx.deformed_structures) == 24
        assert len(process.ctx.strain_voigts) == 24

    def test_gather_fits_compression_positive_stress(self, aiida_profile_clean, aiida_localhost, monkeypatch):
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

    def test_gather_reports_missing_stress(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        from types import SimpleNamespace

        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True)
        process.generate_deformations()

        for idx in range(24):
            setattr(
                process.ctx,
                f"scf_{idx}",
                SimpleNamespace(
                    is_finished_ok=True,
                    outputs=SimpleNamespace(
                        misc=SimpleNamespace(get_dict=lambda: {})
                    ),
                ),
            )

        assert (
            process.gather_results() == VaspElasticWorkChain.exit_codes.ERROR_CHILD
        )


class TestLatticeRelax:
    """The optional full lattice relaxation (vasp.relax) before the deformations."""

    def test_initialize_skips_when_false(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=False)
        calls = []

        def fake_submit(wc_cls, **inputs):
            calls.append(wc_cls)
            return SimpleNamespace(pk=999, process_label="VaspRelaxWorkChain")

        monkeypatch.setattr(process, "submit", fake_submit)
        process.initialize()
        assert calls == []  # no relax submitted
        assert not hasattr(process.ctx, "lattice_relax")

    def test_initialize_submits_vasp_relax(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True)
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

    def test_initialize_forwards_relax_settings(self, aiida_profile_clean, aiida_localhost, monkeypatch):
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

    def test_equilibrium_structure_falls_back_to_input(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True)
        assert process._equilibrium_structure() is process.inputs.structure

    def test_generate_deformations_uses_relaxed_structure(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        from ase.build import bulk

        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True)
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

        defs = generate_deformations([-0.01, -0.005, 0.005, 0.01], [-0.01, -0.005, 0.005, 0.01])
        expected = np.asarray(defs[0][1]) @ np.asarray(relaxed.cell)
        np.testing.assert_allclose(process.ctx.deformed_structures[0].cell, expected)

    def test_generate_deformations_fails_when_relax_failed(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=True)
        process.ctx.lattice_relax = SimpleNamespace(is_finished_ok=False, outputs={})
        assert process.generate_deformations() == VaspElasticWorkChain.exit_codes.ERROR_RELAX

    def test_generate_deformations_no_relax_when_false(self, aiida_profile_clean, aiida_localhost, monkeypatch):
        process = _instantiate(aiida_profile_clean, aiida_localhost, monkeypatch, True, relax_lattice=False)
        assert process.generate_deformations() is None
        from aiida_uranium_workflow.utils.elastic import generate_deformations

        defs = generate_deformations([-0.01, -0.005, 0.005, 0.01], [-0.01, -0.005, 0.005, 0.01])
        expected = np.asarray(defs[0][1]) @ np.asarray(process.inputs.structure.cell)
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
                "options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}
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
        assert inputs["parameters"]["incar"]["encut"] == 400
        assert inputs["potential_family"].value == "PBE"
        assert inputs["potential_mapping"].get_dict() == {"U": "U"}
        assert inputs["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]
        assert inputs["norm_strains"].get_list() == [-0.01, 0.01]
        assert inputs["shear_strains"].get_list() == [-0.01, 0.01]
        assert inputs["relax_internal"].value is False
        assert inputs["relax_lattice"].value is False
        assert "relax_settings" not in inputs

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
        assert "relax_settings" not in inputs

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
                    "relax_settings": {"force_cutoff": 0.05, "steps": 40},
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["relax_lattice"].value is True
        assert inputs["relax_settings"].get_dict() == {
            "force_cutoff": 0.05,
            "steps": 40,
        }
