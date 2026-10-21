"""Tests for the elastic workflows' internal-coordinate relaxation option.

``relax_internal`` switches the strained-cell children from a
fixed-lattice SCF (clamped-ion constants) to a position relaxation with
the cell fixed by the strain (relaxed constants, the method of the
official ABACUS elastic example).
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.elastic.abacus import AbacusElasticWorkChain
from aiida_uranium_workflow.workflows.elastic.fleur import FleurElasticWorkChain


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _instantiate_abacus(
    aiida_profile_clean,
    relax_internal,
    relax_lattice=False,
    require_lattice_relax=False,
    relax_settings=None,
    child_relax_settings=None,
    base=None,
):
    runner = get_manager().get_runner()
    inputs = dict(
        structure=_structure(),
        norm_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        shear_strains=orm.List(list=[-0.01, -0.005, 0.005, 0.01]),
        relax_internal=orm.Bool(relax_internal),
    )
    if relax_lattice:
        inputs["relax_lattice"] = orm.Bool(True)
    if require_lattice_relax:
        inputs["require_lattice_relax"] = orm.Bool(True)
    if relax_settings is not None:
        inputs["relax_settings"] = orm.Dict(dict=relax_settings)
    if child_relax_settings is not None:
        inputs["child_relax_settings"] = orm.Dict(dict=child_relax_settings)
    if base is not None:
        inputs["base"] = base
    return instantiate_process(runner, AbacusElasticWorkChain, **inputs)


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
            sv = np.zeros(6)
            sv[idx[0]] = a
            voigts.append(sv)
    for idx in [(0, 1), (0, 2), (1, 2)]:
        for a in [-0.01, -0.005, 0.005, 0.01]:
            sv = np.zeros(6)
            sv[{(0, 1): 5, (0, 2): 4, (1, 2): 3}[idx]] = 2 * a
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

    def test_child_relax_settings_reach_the_children(self, aiida_profile_clean):
        """``child_relax_settings`` is the RelaxOptions of the strained cells."""
        process = _instantiate_abacus(
            aiida_profile_clean,
            True,
            child_relax_settings={"relax_type": "positions", "force_cutoff": 0.001},
        )
        structure = _run_generate(process)
        base = {"abacus": {"parameters": orm.Dict(dict={"input": {}})}}
        inputs = process._relax_inputs_for(structure, base)
        assert inputs["relax_settings"].get_dict()["force_cutoff"] == 0.001


class TestAbacusLatticeRelax:
    """``relax_lattice`` — the abacus.relax pre-relaxation of the input cell."""

    def test_spec_has_the_lattice_relax_inputs(self):
        ports = AbacusElasticWorkChain.spec().inputs
        for name in ("relax_lattice", "require_lattice_relax", "relax_settings"):
            assert name in ports
        assert AbacusElasticWorkChain.exit_codes.ERROR_RELAX.status == 310

    def test_initialize_skips_when_false(self, aiida_profile_clean):
        process = _instantiate_abacus(aiida_profile_clean, True, relax_lattice=False)
        calls = []

        def fake_submit(wc_cls, **inputs):
            calls.append(wc_cls)
            return SimpleNamespace(pk=999, process_label="AbacusRelaxWorkChain")

        process.submit = fake_submit
        process.initialize()
        assert calls == []
        assert not hasattr(process.ctx, "lattice_relax")

    def test_initialize_submits_abacus_relax(self, aiida_profile_clean):
        process = _instantiate_abacus(
            aiida_profile_clean,
            True,
            relax_lattice=True,
            relax_settings={
                "relax_type": "positions_cell",
                "relax_method": "cg",
                "max_ionic_steps": 50,
                "force_cutoff": 0.05,
                "stress_cutoff": 1.0,
            },
            base={
                "abacus": {
                    "code": orm.Str("code"),
                    "parameters": orm.Dict(dict={"input": {}}),
                }
            },
        )
        calls, ctx_calls = [], []

        def fake_submit(wc_cls, **inputs):
            calls.append((wc_cls, inputs))
            return SimpleNamespace(pk=999, process_label="AbacusRelaxWorkChain")

        process.submit = fake_submit
        process.to_context = lambda **kw: ctx_calls.append(kw)
        process.initialize()

        assert len(calls) == 1
        wc_cls, relax_inputs = calls[0]
        from aiida_abacus.workflows.relax import AbacusRelaxWorkChain

        assert wc_cls is AbacusRelaxWorkChain
        # structure at the top level, SCF base in the ``base`` namespace
        assert relax_inputs["structure"] is process.inputs.structure
        assert "structure" not in relax_inputs["base"]["abacus"]
        assert relax_inputs["metadata"]["label"] == "lattice_relax"
        settings = relax_inputs["relax_settings"].get_dict()
        assert settings["relax_type"] == "positions_cell"  # cell + ions
        assert settings["force_cutoff"] == 0.05
        assert settings["max_ionic_steps"] == 50
        # the running child is registered under ``lattice_relax``
        assert list(ctx_calls[0]) == ["lattice_relax"]

    def test_generate_deformations_uses_the_relaxed_structure(
        self, aiida_profile_clean
    ):
        from ase.build import bulk

        process = _instantiate_abacus(aiida_profile_clean, True, relax_lattice=True)
        relaxed = orm.StructureData(ase=bulk("U", "bcc", a=3.5))
        process.ctx.lattice_relax = SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(structure=relaxed),
        )
        assert process._equilibrium_structure() is relaxed
        assert process.generate_deformations() is None
        assert process.ctx.equilibrium_structure_source == "lattice_relax"
        assert len(process.ctx.deformed_structures) == 24

    def test_generate_deformations_continues_when_relax_failed(
        self, aiida_profile_clean
    ):
        process = _instantiate_abacus(aiida_profile_clean, True, relax_lattice=True)
        process.ctx.lattice_relax = SimpleNamespace(
            pk=1, is_finished_ok=False, exit_status=300
        )
        assert process.generate_deformations() is None
        assert process.ctx.equilibrium_structure_source == "input"
        assert len(process.ctx.deformed_structures) == 24

    def test_generate_deformations_aborts_when_required(self, aiida_profile_clean):
        process = _instantiate_abacus(
            aiida_profile_clean,
            True,
            relax_lattice=True,
            require_lattice_relax=True,
        )
        process.ctx.lattice_relax = SimpleNamespace(
            pk=1, is_finished_ok=False, exit_status=300
        )
        assert (
            process.generate_deformations()
            == AbacusElasticWorkChain.exit_codes.ERROR_RELAX
        )


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

    def test_gather_reads_scf_energy_without_relax(
        self, aiida_profile_clean, aiida_localhost
    ):
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
