"""Tests for the elastic workflows' internal-coordinate relaxation option.

``relax_internal`` switches the strained-cell children from a
fixed-lattice SCF (clamped-ion constants) to a position relaxation with
the cell fixed by the strain (relaxed constants, the method of the
official ABACUS elastic example).
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.elastic.abacus import AbacusElasticWorkChain
from aiida_uranium_workflow.workflows.elastic.fleur import FleurElasticWorkChain


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _instantiate_abacus(aiida_profile_clean, relax_internal):
    runner = get_manager().get_runner()
    return instantiate_process(
        runner,
        AbacusElasticWorkChain,
        structure=_structure(),
        norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        relax_internal=orm.Bool(relax_internal),
    )


def _instantiate_fleur(aiida_profile_clean, relax_internal, aiida_localhost):
    orm.InstalledCode(
        label="elastic_fleur_test",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    orm.InstalledCode(
        label="elastic_inpgen_test",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    runner = get_manager().get_runner()
    return instantiate_process(
        runner,
        FleurElasticWorkChain,
        structure=_structure(),
        fleur=orm.load_code("elastic_fleur_test"),
        inpgen=orm.load_code("elastic_inpgen_test"),
        wf_parameters=orm.Dict(dict={"mode": "density"}),
        calc_parameters=orm.Dict(dict={"comp": {"kmax": 7.0}}),
        options=orm.Dict(dict={"resources": {"num_machines": 1}}),
        options_inpgen=orm.Dict(dict={}),
        norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        relax_internal=orm.Bool(relax_internal),
    )




def _fake_strain_voigts():
    """The 24 DeformedStructureSet Voigt strains (norm + shear, ±1/±0.5%)."""
    import numpy as np
    voigts = []
    for idx in [(0, 0), (1, 1), (2, 2)]:
        for a in [-0.01, -0.005, 0.005, 0.01]:
            sv = np.zeros(6); sv[idx[0]] = a; voigts.append(sv)
    for idx in [(0, 1), (0, 2), (1, 2)]:
        for a in [-0.01, -0.005, 0.005, 0.01]:
            sv = np.zeros(6); sv[{(0, 1): 5, (0, 2): 4, (1, 2): 3}[idx]] = 2 * a
            voigts.append(sv)
    return voigts

def _run_generate(process):
    process.generate_deformations()
    assert len(process.ctx.deformed_structures) == 24
    return process.ctx.deformed_structures[0]


# ---------------------------------------------------------------------------
# ABACUS
# ---------------------------------------------------------------------------


class TestAbacusRelaxInternal:
    def test_spec_has_relax_internal(self):
        assert "relax_internal" in AbacusElasticWorkChain.spec().inputs

    def test_relax_inputs_use_abacus_relax(self, aiida_profile_clean):
        process = _instantiate_abacus(aiida_profile_clean, True)
        structure = _run_generate(process)
        base = {
            "abacus": {
                "code": orm.Str("code"),
                "parameters": orm.Dict(dict={"input": {"ecutwfc": 65}}),
            },
            "kpoints": orm.Str("kpoints"),
        }
        inputs = process._relax_inputs_for(structure, base)

        # abacus.relax layout: structure at top level, SCF base in the
        # ``base`` namespace without a structure, relax_settings present.
        assert inputs["structure"] is structure
        assert "structure" not in inputs["base"]["abacus"]
        assert inputs["base"]["abacus"]["code"] is base["abacus"]["code"]
        assert inputs["base"]["kpoints"] is base["kpoints"]
        assert inputs["meta_convergence"].value is False
        assert inputs["clean_workdir"].value is False

        rs = inputs["relax_settings"].get_dict()
        assert rs["relax_type"] == "positions"  # atoms only, cell fixed
        # The SCF parameters must run ``calculation relax`` with stress on.
        params = inputs["base"]["abacus"]["parameters"].get_dict()["input"]
        assert params["calculation"] == "relax"
        assert params["cal_stress"] == 1
        # The base ecutwfc survives the override.
        assert params["ecutwfc"] == 65

    def test_scf_inputs_keep_structure_in_base(self, aiida_profile_clean):
        process = _instantiate_abacus(aiida_profile_clean, False)
        structure = _run_generate(process)
        base = {"abacus": {"parameters": orm.Dict(dict={"input": {}})}}
        inputs = process._scf_inputs_for(structure, base)
        assert inputs["abacus"]["structure"] is structure


# ---------------------------------------------------------------------------
# FLEUR
# ---------------------------------------------------------------------------


class TestFleurRelaxInternal:
    def test_spec_has_relax_internal(self):
        assert "relax_internal" in FleurElasticWorkChain.spec().inputs
        assert "relax_wf_parameters" in FleurElasticWorkChain.spec().inputs

    def test_relax_child_layout(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate_fleur(aiida_profile_clean, True, aiida_localhost)
        structure = _run_generate(process)
        inputs = process._child_inputs(structure, relax_internal=True)

        # FleurRelaxWorkChain layout: SCF base in the ``scf`` namespace,
        # the relax wf parameters at the top level.
        assert inputs["scf"]["structure"] is structure
        assert inputs["scf"]["wf_parameters"].get_dict() == {"mode": "density"}
        assert inputs["scf"]["calc_parameters"].get_dict() == {"comp": {"kmax": 7.0}}
        assert inputs["scf"]["options"].get_dict() == {"resources": {"num_machines": 1}}
        relax_wf = inputs["wf_parameters"].get_dict()
        assert relax_wf["relaxation_type"] == "atoms"  # positions only, cell fixed
        assert relax_wf["run_final_scf"] is False

    def test_relax_wf_parameters_override(self, aiida_profile_clean, aiida_localhost):
        runner = get_manager().get_runner()
        orm.InstalledCode(
            label="elastic_fleur_test",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        orm.InstalledCode(
            label="elastic_inpgen_test",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        process = instantiate_process(
            runner,
            FleurElasticWorkChain,
            structure=_structure(),
            fleur=orm.load_code("elastic_fleur_test"),
            inpgen=orm.load_code("elastic_inpgen_test"),
            wf_parameters=orm.Dict(dict={"mode": "density"}),
            calc_parameters=orm.Dict(dict={"comp": {"kmax": 7.0}}),
            options=orm.Dict(dict={"resources": {"num_machines": 1}}),
            options_inpgen=orm.Dict(dict={}),
            norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
            shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
            relax_internal=orm.Bool(True),
            relax_wf_parameters=orm.Dict(
                dict={"run_final_scf": True, "force_criterion": 0.0005}
            ),
        )
        structure = _run_generate(process)
        inputs = process._child_inputs(structure, relax_internal=True)
        relax_wf = inputs["wf_parameters"].get_dict()
        # User overrides win; untouched defaults survive.
        assert relax_wf["run_final_scf"] is True
        assert relax_wf["force_criterion"] == 0.0005
        assert relax_wf["relaxation_type"] == "atoms"
        assert relax_wf["relax_iter"] == 20

    def test_scf_child_layout_without_relax(self, aiida_profile_clean, aiida_localhost):
        process = _instantiate_fleur(aiida_profile_clean, False, aiida_localhost)
        structure = _run_generate(process)
        inputs = process._child_inputs(structure, relax_internal=False)
        # Plain FleurScfWorkChain inputs, no ``scf`` namespace wrapper.
        assert inputs["structure"] is structure
        assert "scf" not in inputs
        assert "wf_parameters" in inputs

    def test_gather_reads_relax_last_energy(self, aiida_profile_clean, aiida_localhost):
        from types import SimpleNamespace

        process = _instantiate_fleur(aiida_profile_clean, True, aiida_localhost)
        process.ctx.strain_voigts = _fake_strain_voigts()
        process.ctx.deformed_structures = [None] * 24

        def stub(energy_ev):
            return SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    output_relax_wc_para=SimpleNamespace(
                        get_dict=lambda: {"last_energy": energy_ev}
                    )
                ),
            )

        for idx in range(24):
            setattr(process.ctx, f"scf_{idx}", stub(-1528354.0))

        exit_code = process.gather_results()
        assert exit_code is None
        assert "output_parameters" in getattr(process, "_outputs", {})

    def test_gather_reads_scf_energy_without_relax(self, aiida_profile_clean, aiida_localhost):
        from types import SimpleNamespace

        process = _instantiate_fleur(aiida_profile_clean, False, aiida_localhost)
        process.ctx.strain_voigts = _fake_strain_voigts()
        process.ctx.deformed_structures = [None] * 24

        def stub(energy_htr):
            return SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    output_scf_wc_para=SimpleNamespace(
                        get_dict=lambda: {"total_energy": energy_htr}
                    )
                ),
            )

        for idx in range(24):
            setattr(process.ctx, f"scf_{idx}", stub(-100.0))

        exit_code = process.gather_results()
        assert exit_code is None
        assert "output_parameters" in getattr(process, "_outputs", {})
