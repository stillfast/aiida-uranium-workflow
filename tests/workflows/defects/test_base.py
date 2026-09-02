"""WorkChain-level tests for the defect workflows (stub host/defect children)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.defects.abacus import AbacusDefectsWorkChain
from aiida_uranium_workflow.workflows.defects.fleur import FleurDefectsWorkChain


def _structure():
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _instantiate(aiida_profile_clean, cls, mode="scf", defect=None, base=None):
    if defect is None:
        defect = {
            "type": "vacancy", "site_index": 0, "element": "U", "label": "vac_U_0",
        }
    kwargs = dict(
        structure=_structure(),
        supercell_matrix=orm.List(list=[2, 2, 2]),
        defect=orm.Dict(dict=defect),
        wf_parameters=orm.Dict(dict={"mode": mode}),
    )
    if base is not None:
        kwargs["base"] = base
    runner = get_manager().get_runner()
    return instantiate_process(runner, cls, **kwargs)


def _stub_abacus_child(energy):
    return SimpleNamespace(
        is_finished_ok=True,
        outputs=SimpleNamespace(
            misc=SimpleNamespace(get_dict=lambda: {"total_energy": energy})
        ),
    )


def _stub_fleur_child(energy_htr):
    return SimpleNamespace(
        is_finished_ok=True,
        outputs=SimpleNamespace(
            output_scf_wc_para=SimpleNamespace(
                get_dict=lambda: {"total_energy": energy_htr}
            )
        ),
    )


def test_generate_structures_abacus(aiida_profile_clean):
    """generate_structures builds an 8-atom host and a 7-atom vacancy."""
    process = _instantiate(aiida_profile_clean, AbacusDefectsWorkChain)
    exit_code = process.generate_structures()
    assert exit_code is None
    assert len(process.ctx.host.sites) == 8
    assert len(process.ctx.defect.sites) == 7
    assert process.ctx.defect_dict["type"] == "vacancy"


def test_abacus_relax_mode_uses_abacus_relax(aiida_profile, aiida_localhost):
    """relax mode must submit ``abacus.relax`` with positions-only
    settings (relax_type='positions', meta_convergence off)."""
    from aiida.plugins import WorkflowFactory

    computer = aiida_localhost
    code = orm.InstalledCode(
        label="defect_relax_test_code",
        computer=computer,
        filepath_executable="/bin/true",
    ).store()
    base = {
        "abacus": {
            "code": code,
            "parameters": orm.Dict(
                dict={"input": {"calculation": "scf", "scf_thr": 1e-7}}
            ),
        }
    }
    process = _instantiate(
        aiida_profile, AbacusDefectsWorkChain, mode="relax",
        defect={
            "type": "interstitial", "element": "U",
            "position": [0.25, 0.25, 0.25], "label": "int_U",
        },
        base=base,
    )

    assert process._wc_cls() is WorkflowFactory("abacus.relax")
    inputs = process._make_calc_inputs(process.inputs.structure, "host")

    assert inputs["relax_settings"].get_dict()["relax_type"] == "positions"
    assert inputs["meta_convergence"].value is False
    # Top-level structure for abacus.relax; the base namespace carries no
    # structure (the plugin's base namespace excludes abacus.structure).
    assert "structure" in inputs
    assert "structure" not in inputs["base"]["abacus"]
    assert (
        inputs["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"]
        == "relax"
    )

    # scf mode still uses abacus.base.
    process_scf = _instantiate(aiida_profile, AbacusDefectsWorkChain, base=base)
    assert process_scf._wc_cls() is WorkflowFactory("abacus.base")


def test_gather_results_abacus_vacancy(aiida_profile_clean):
    """gather_results computes E_f for a vacancy (μ_U = −10 eV)."""
    process = _instantiate(aiida_profile_clean, AbacusDefectsWorkChain)
    process.generate_structures()
    process.ctx.host_wc = _stub_abacus_child(-110.0)
    process.ctx.defect_wc = _stub_abacus_child(-100.0)

    exit_code = process.gather_results()
    assert exit_code is None, f"gather_results failed: {exit_code}"

    outputs = getattr(process, "_outputs", {})
    para_node = outputs["output_parameters"]
    # The output node must be stored (created by a calcfunction) — a bare
    # orm.Dict built in the WorkChain would be rejected by AiiDA's
    # update_outputs (regression: "tried returning an unstored Data node").
    assert para_node.is_stored
    para = para_node.get_dict()
    # E_f = E_def − E_host × (N_def/N_host) = -100 - (-110)·7/8 = -3.75.
    assert para["formation_energy_ev"] == pytest.approx(-3.75)
    assert para["host_natoms"] == 8
    assert para["defect_natoms"] == 7
    assert para["mode"] == "scf"
    assert "host_structure" in outputs and "defect_structure" in outputs


def test_gather_results_fleur_interstitial(aiida_profile_clean):
    """gather_results computes E_f for an interstitial (FLEUR, Hartree)."""
    process = _instantiate(
        aiida_profile_clean,
        FleurDefectsWorkChain,
        defect={
            "type": "interstitial", "element": "U",
            "position": [0.25, 0.25, 0.25], "label": "int_U",
        },
    )
    process.generate_structures()
    assert len(process.ctx.host.sites) == 8
    assert len(process.ctx.defect.sites) == 9
    # Host E = −110 eV → −110/27.2114 Htr; defect E = −115 eV.
    process.ctx.host_wc = _stub_fleur_child(-110.0 / 27.211386245988)
    process.ctx.defect_wc = _stub_fleur_child(-115.0 / 27.211386245988)

    exit_code = process.gather_results()
    assert exit_code is None

    outputs = getattr(process, "_outputs", {})
    para_node = outputs["output_parameters"]
    assert para_node.is_stored
    para = para_node.get_dict()
    # E_f = E_def − E_host × (N_def/N_host) = -115 - (-110)·9/8 = +8.75.
    assert para["formation_energy_ev"] == pytest.approx(8.75)
    assert para["host_natoms"] == 8
    assert para["defect_natoms"] == 9
