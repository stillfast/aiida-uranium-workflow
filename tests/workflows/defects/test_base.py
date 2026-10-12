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


def _instantiate(aiida_profile_clean, cls, mode="scf", defect=None, base=None,
                 supercell_matrices=None):
    if defect is None:
        defect = {
            "type": "vacancy", "site_index": 0, "element": "U", "label": "vac_U_0",
        }
    kwargs = dict(
        structure=_structure(),
        defect=orm.Dict(dict=defect),
        wf_parameters=orm.Dict(dict={"mode": mode}),
    )
    if supercell_matrices is None:
        kwargs["supercell_matrix"] = orm.List(list=[2, 2, 2])
    elif supercell_matrices != []:
        kwargs["supercell_matrices"] = orm.List(list=supercell_matrices)
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
    process.ctx.host_0_wc = _stub_abacus_child(-110.0)
    process.ctx.defect_0_wc = _stub_abacus_child(-100.0)

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
    process.ctx.host_0_wc = _stub_fleur_child(-110.0 / 27.211386245988)
    process.ctx.defect_0_wc = _stub_fleur_child(-115.0 / 27.211386245988)

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


def test_multi_supercell_scan_abacus(aiida_profile_clean):
    """Two supercell matrices in one WorkChain → per-cell formation energies."""
    process = _instantiate(
        aiida_profile_clean,
        AbacusDefectsWorkChain,
        supercell_matrices=[
            [2, 2, 2],
            [3, 3, 3],
        ],
    )
    process.generate_structures()
    assert len(process.ctx.cells) == 2
    assert [c["label"] for c in process.ctx.cells] == ["2x2x2", "3x3x3"]
    assert len(process.ctx.cells[0]["host"].sites) == 8
    assert len(process.ctx.cells[1]["host"].sites) == 27

    # per-cell stub children: host always −1.0 eV/atom, defect −1.0 eV/atom
    # minus a 1 eV penalty ⇒ E_f = −1 − N_host_scale ... use explicit values
    for i, n_host in enumerate((8, 27)):
        setattr(process.ctx, f"host_{i}_wc", _stub_abacus_child(-1.0 * n_host))
        setattr(process.ctx, f"defect_{i}_wc", _stub_abacus_child(-1.0 * (n_host - 1) - 1.0))

    exit_code = process.gather_results()
    assert exit_code is None, f"gather_results failed: {exit_code}"

    para = getattr(process, "_outputs", {})["output_parameters"].get_dict()
    assert para["n_cells"] == 2
    assert [c["label"] for c in para["cells"]] == ["2x2x2", "3x3x3"]
    # E_f = E_defect − E_host × N_defect/N_host
    for cell, n_host in zip(para["cells"], (8, 27)):
        e_host = -1.0 * n_host
        e_def = -1.0 * (n_host - 1) - 1.0
        assert cell["formation_energy_ev"] == pytest.approx(
            e_def - e_host * (n_host - 1) / n_host
        )
    assert para["formation_energies_ev"] == [
        c["formation_energy_ev"] for c in para["cells"]
    ]
    # legacy flat keys are only added for a single cell
    assert "host_natoms" not in para


def test_single_matrix_keeps_flat_output(aiida_profile_clean):
    """The legacy single-matrix preset still yields the flat output keys."""
    process = _instantiate(aiida_profile_clean, AbacusDefectsWorkChain)
    process.generate_structures()
    assert len(process.ctx.cells) == 1
    process.ctx.host_0_wc = _stub_abacus_child(-110.0)
    process.ctx.defect_0_wc = _stub_abacus_child(-100.0)
    assert process.gather_results() is None
    para = getattr(process, "_outputs", {})["output_parameters"].get_dict()
    assert para["host_natoms"] == 8 and para["defect_natoms"] == 7
    assert para["n_cells"] == 1


def _stub_failed_child(pk, exit_status, exit_message="failed"):
    """A finished-but-unsuccessful child (e.g. abacus.base exit 300/401)."""
    return SimpleNamespace(
        pk=pk,
        is_finished_ok=False,
        exit_status=exit_status,
        exit_message=exit_message,
        outputs=SimpleNamespace(),
    )


def test_per_cell_scf_override_lands_in_child(aiida_profile_clean):
    """A per-cell `mixing_ndim` in the protocol reaches the child INPUT.

    The larger supercells need a longer mixing history (measured: 16 for
    249 atoms, 20-24 for 431 atoms), so each cell must be able to override
    the shared SCF base preset.
    """
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    from aiida_uranium_workflow.workflows.defects.abacus import AbacusDefectsWorkChain

    base = {
        "abacus": {
            "code": orm.Str("code"),
            "parameters": orm.Dict(
                dict={"input": {"calculation": "scf", "mixing_ndim": 8}}
            ),
        },
    }
    process = instantiate_process(
        get_manager().get_runner(),
        AbacusDefectsWorkChain,
        structure=_structure(),
        defect=orm.Dict(
            dict={"type": "vacancy", "site_index": 0, "element": "U", "label": "v"}
        ),
        wf_parameters=orm.Dict(dict={"mode": "scf"}),
        supercell_cells=orm.List(
            list=[
                {"matrix": [2, 2, 2], "mixing_ndim": 8},
                {"matrix": [6, 6, 6], "mixing_ndim": 24, "scf_nmax": 250},
            ]
        ),
        base=base,
    )
    process.generate_structures()
    assert [c["scf"] for c in process.ctx.cells] == [
        {"mixing_ndim": 8},
        {"mixing_ndim": 24, "scf_nmax": 250},
    ]

    small = process._make_calc_inputs(
        process.ctx.cells[0]["host"], "host", scf_overrides=process.ctx.cells[0]["scf"]
    )
    large = process._make_calc_inputs(
        process.ctx.cells[1]["host"], "host", scf_overrides=process.ctx.cells[1]["scf"]
    )
    small_in = small["abacus"]["parameters"].get_dict()["input"]
    large_in = large["abacus"]["parameters"].get_dict()["input"]
    assert small_in["mixing_ndim"] == 8
    assert large_in["mixing_ndim"] == 24
    assert large_in["scf_nmax"] == 250
    # the SCF base preset is untouched for the cells that do not override
    assert "scf_nmax" not in small_in
    # and the shared preset node itself is not mutated
    shared = process.inputs.base["abacus"]["parameters"].get_dict()["input"]
    assert shared["mixing_ndim"] == 8
    assert "scf_nmax" not in shared


def test_per_cell_scf_override_rejected_for_fleur(aiida_profile_clean):
    """Per-cell SCF keys are ABACUS INPUT keys — FLEUR must fail loudly."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    from aiida_uranium_workflow.workflows.defects.fleur import FleurDefectsWorkChain

    process = instantiate_process(
        get_manager().get_runner(),
        FleurDefectsWorkChain,
        structure=_structure(),
        defect=orm.Dict(
            dict={"type": "vacancy", "site_index": 0, "element": "U", "label": "v"}
        ),
        wf_parameters=orm.Dict(dict={"mode": "scf"}),
        supercell_cells=orm.List(list=[{"matrix": [2, 2, 2], "mixing_ndim": 16}]),
    )
    process.generate_structures()
    with pytest.raises(ValueError, match="ABACUS INPUT keys"):
        process._make_calc_inputs(
            process.ctx.cells[0]["host"], "host", scf_overrides={"mixing_ndim": 16}
        )


def test_child_max_iterations_defaults_to_zero(aiida_profile, aiida_localhost):
    """Children get max_iterations=0 (no auto-restart) unless overridden.

    The plugin's retry ladder (scf_nmax → mixing_beta) is counter-productive
    once the per-cell SCF settings are tuned, and every retry re-runs the
    whole supercell. ``abacus.base`` takes the port at the top level, while
    ``abacus.relax`` exposes it as ``base.max_iterations``.
    """
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    from aiida_uranium_workflow.workflows.defects.abacus import AbacusDefectsWorkChain

    code = orm.InstalledCode(
        label="defect_maxiter_test_code",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    base = {
        "abacus": {
            "code": code,
            "parameters": orm.Dict(dict={"input": {"calculation": "scf"}}),
        }
    }
    structure = _structure()
    defect = orm.Dict(
        dict={"type": "vacancy", "site_index": 0, "element": "U", "label": "v"}
    )

    def _inputs(mode, **extra):
        proc = instantiate_process(
            get_manager().get_runner(),
            AbacusDefectsWorkChain,
            structure=structure,
            defect=defect,
            wf_parameters=orm.Dict(dict={"mode": mode}),
            base=base,
            **extra,
        )
        return proc._make_calc_inputs(structure, "host")

    # scf mode → top-level port on abacus.base
    assert _inputs("scf")["max_iterations"].value == 0
    # relax mode → exposed base namespace on abacus.relax
    assert _inputs("relax")["base"]["max_iterations"].value == 0

    # …and an explicit workflow input overrides it in both modes
    assert _inputs("scf", max_iterations=orm.Int(3))["max_iterations"].value == 3
    assert _inputs("relax", max_iterations=orm.Int(3))["base"][
        "max_iterations"
    ].value == 3


def test_stub_failed_child_helper():
    """Guard the helper used by the partial-failure tests."""
    child = _stub_failed_child(1, 300)
    assert child.is_finished_ok is False and child.exit_status == 300


def test_gather_results_partial_failure(aiida_profile_clean):
    """One failed cell must not throw away the cells that succeeded."""
    process = _instantiate(
        aiida_profile_clean,
        AbacusDefectsWorkChain,
        supercell_matrices=[[2, 2, 2], [3, 3, 3]],
    )
    process.generate_structures()
    # cell 0 fine, cell 1's defect child failed (e.g. SCF did not converge)
    process.ctx.host_0_wc = _stub_abacus_child(-110.0)
    process.ctx.defect_0_wc = _stub_abacus_child(-100.0)
    process.ctx.host_1_wc = _stub_abacus_child(-200.0)
    process.ctx.defect_1_wc = _stub_failed_child(999999, 401, "max iterations exceeded")

    exit_code = process.gather_results()
    assert exit_code is not None and exit_code.status == 310, (
        f"expected PARTIAL (310), got {exit_code}"
    )

    para = getattr(process, "_outputs", {})["output_parameters"].get_dict()
    assert para["status"] == "partial"
    assert para["n_cells"] == 1
    assert para["n_cells_requested"] == 2
    assert para["n_cells_failed"] == 1
    # the surviving cell keeps its formation energy (-100 - (-110)·7/8)
    assert [c["label"] for c in para["cells"]] == ["2x2x2"]
    assert para["cells"][0]["formation_energy_ev"] == pytest.approx(-3.75)
    assert para["formation_energy_ev"] == pytest.approx(-3.75)  # legacy flat keys
    # the failure is traceable from the outputs alone
    assert para["failed_cells"] == ["3x3x3"]
    failure = para["failures"][0]
    assert failure["reason"] == "child_failed"
    assert failure["defect_pk"] == 999999
    assert failure["defect_exit_status"] == 401
    assert failure["host_ok"] is True and failure["defect_ok"] is False
    # legacy structure outputs point at the first *computed* cell
    assert len(getattr(process, "_outputs", {})["host_structure"].sites) == 8


def test_gather_results_all_failed_still_errors(aiida_profile_clean):
    """With no successful cell the WorkChain must still fail (exit 300)."""
    process = _instantiate(aiida_profile_clean, AbacusDefectsWorkChain)
    process.generate_structures()
    process.ctx.host_0_wc = _stub_failed_child(111, 300, "child failed")
    process.ctx.defect_0_wc = _stub_abacus_child(-100.0)

    exit_code = process.gather_results()
    assert exit_code is not None and exit_code.status == 300
    assert "output_parameters" not in getattr(process, "_outputs", {})


def test_gather_results_partial_energy_read_failure(aiida_profile_clean):
    """An unreadable energy skips that cell only (not the whole scan)."""

    def _broken_child():
        return SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(misc=SimpleNamespace(get_dict=lambda: {})),
        )

    process = _instantiate(
        aiida_profile_clean,
        AbacusDefectsWorkChain,
        supercell_matrices=[[2, 2, 2], [3, 3, 3]],
    )
    process.generate_structures()
    process.ctx.host_0_wc = _broken_child()
    process.ctx.defect_0_wc = _stub_abacus_child(-100.0)
    process.ctx.host_1_wc = _stub_abacus_child(-200.0)
    process.ctx.defect_1_wc = _stub_abacus_child(-190.0)

    exit_code = process.gather_results()
    assert exit_code is not None and exit_code.status == 310
    para = getattr(process, "_outputs", {})["output_parameters"].get_dict()
    assert [c["label"] for c in para["cells"]] == ["3x3x3"]
    assert para["failures"][0]["reason"] == "energy_read_failed"
    assert para["failures"][0]["label"] == "2x2x2"


def test_generate_structures_without_matrix(aiida_profile_clean):
    """No supercell matrix → the input structure is the host cell (cloned,
    so the calcfunction output does not create a provenance cycle)."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    structure = _structure()
    process = instantiate_process(
        get_manager().get_runner(),
        AbacusDefectsWorkChain,
        structure=structure,
        defect=orm.Dict(
            dict={"type": "vacancy", "site_index": 0, "element": "U", "label": "v"}
        ),
        wf_parameters=orm.Dict(dict={"mode": "scf"}),
    )
    process.generate_structures()
    assert len(process.ctx.cells) == 1
    cell = process.ctx.cells[0]
    assert cell["label"] == "host_cell"
    assert cell["matrix"] is None
    assert cell["host"].pk != structure.pk  # clone, not the input node
    assert len(cell["host"].sites) == len(structure.sites)
    assert len(cell["defect"].sites) == len(structure.sites) - 1


def test_per_cell_kpoints_applied_to_child(aiida_profile_clean):
    """Each cell's k-mesh ends up in that cell's child inputs."""
    from aiida.plugins import DataFactory

    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    base = {
        "abacus": {
            "code": orm.Str("code"),
            "parameters": orm.Dict(dict={"input": {"calculation": "scf"}}),
        },
        "kpoints": DataFactory("core.array.kpoints")(),
    }
    process = instantiate_process(
        get_manager().get_runner(),
        AbacusDefectsWorkChain,
        structure=_structure(),
        defect=orm.Dict(
            dict={"type": "vacancy", "site_index": 0, "element": "U", "label": "v"}
        ),
        wf_parameters=orm.Dict(dict={"mode": "scf"}),
        supercell_cells=orm.List(
            list=[
                {"matrix": [2, 2, 2], "kpoints_mesh": [7, 7, 7]},
                {"matrix": [3, 3, 3], "kpoints_mesh": [5, 5, 5]},
            ]
        ),
        base=base,
    )
    process.generate_structures()
    assert [c["kpoints_mesh"] for c in process.ctx.cells] == [[7, 7, 7], [5, 5, 5]]
    assert "kpoints" in process.ctx.cells[1]

    inputs0 = process._make_calc_inputs(
        process.ctx.cells[0]["host"], "host_2x2x2", kpoints=process.ctx.cells[0]["kpoints"]
    )
    inputs1 = process._make_calc_inputs(
        process.ctx.cells[1]["host"], "host_3x3x3", kpoints=process.ctx.cells[1]["kpoints"]
    )
    assert inputs0["kpoints"].get_kpoints_mesh()[0] == [7, 7, 7]
    assert inputs1["kpoints"].get_kpoints_mesh()[0] == [5, 5, 5]
    # the cell k-mesh overrides the base k-points
    assert inputs1["kpoints"].uuid != process.inputs.base.kpoints.uuid
