"""Tests for the VASP ``relax`` backend (adapter + protocol translation).

The relax method used to support ABACUS / FLEUR / QE only — the ``vasp:``
block of ``parameters/relax.yml`` was an empty placeholder and the CLI's
``RELAX_CLASS_TO_BACKEND`` had no VASP entry. VASP is now wired through
aiida-vasp's ``vasp.relax`` (``VaspRelaxWorkChain``): the protocol block
keeps the ABACUS vocabulary and
:func:`relax_settings_from_protocol` translates it into the plugin's
``RelaxOptions`` (ISIF / IBRION / NSW / EDIFFG).
"""

from __future__ import annotations

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.relax import (
    relax_settings_from_protocol,
    VaspRelaxAdapter,
)

#: ``parameters/vasp/scf.yml`` -> ``standard`` preset (bcc-U, SCF base).
VASP_SCF = {
    "parameters": {
        "incar": {
            "system": "bcc-uranium",
            "ispin": 1,
            "encut": 400,
            "ismear": 1,
            "sigma": 0.2,
            "nsw": 0,
        }
    },
    "potential_family": "PBE",
    "potential_mapping": {"U": "U"},
    "kpoints_spacing": 0.0318,
}

#: The ``vasp`` block of ``parameters/relax.yml``'s ``test`` preset.
VASP_BLOCK = {
    "relax_type": "volume",
    "relax_method": "cg",
    "max_ionic_steps": 50,
    "force_cutoff": 0.05,
    "meta_convergence": True,
    "volume_convergence": 0.01,
    "max_meta_convergence_iterations": 5,
}

#: bcc-U cell volume at a = 3.45 Å (≈ 20.5 Å³) used to convert the
#: relative ``volume_convergence`` into aiida-vasp's absolute Å³.
V0 = 20.5


def _adapter(block=VASP_BLOCK, scf=None, code_label="vasp_relax"):
    return VaspRelaxAdapter(
        code_label=code_label,
        software_params=dict(scf or VASP_SCF),
        metadata={"options": {"queue_name": "q_test", "max_wallclock_hours": 1}},
        workflow_data={"vasp": block},
        extra_codes={},
    )


# ---------------------------------------------------------------------------
# Protocol block -> RelaxOptions translation (pure function, no profile)
# ---------------------------------------------------------------------------


def test_volume_relax_maps_to_isif7():
    settings = relax_settings_from_protocol(VASP_BLOCK, structure_volume=V0)
    # volume-only: ISIF=7 (shape + positions fixed)
    assert settings["positions"] is False
    assert settings["shape"] is False
    assert settings["volume"] is True
    assert settings["algo"] == "cg"
    assert settings["steps"] == 50
    assert settings["force_cutoff"] == pytest.approx(0.05)
    assert settings["convergence_on"] is True
    assert settings["convergence_max_iterations"] == 5
    # relative |ΔV|/V0 -> absolute Å³
    assert settings["convergence_volume"] == pytest.approx(0.01 * V0)


def test_positions_cell_maps_to_isif3():
    settings = relax_settings_from_protocol({"relax_type": "positions_cell"})
    assert (settings["positions"], settings["shape"], settings["volume"]) == (
        True,
        True,
        True,
    )


def test_full_is_an_alias_of_positions_cell():
    """``full`` (the qe block's spelling for vc-relax) = ISIF=3 here."""
    settings = relax_settings_from_protocol({"relax_type": "full"})
    assert (settings["positions"], settings["shape"], settings["volume"]) == (
        True,
        True,
        True,
    )


@pytest.mark.parametrize(
    "relax_type, dof",
    [
        ("positions", (True, False, False)),
        ("shape", (False, True, False)),
        ("cell", (False, True, True)),
        ("positions_shape", (True, True, False)),
    ],
)
def test_relax_type_dof_table(relax_type, dof):
    settings = relax_settings_from_protocol({"relax_type": relax_type})
    assert (
        settings["positions"],
        settings["shape"],
        settings["volume"],
    ) == dof


def test_relax_type_is_required():
    # aiida-vasp's RelaxOptions default is a FULL relaxation; silently
    # accepting an empty block would relax far more than the preset asks.
    with pytest.raises(ValueError, match="'relax_type' is required"):
        relax_settings_from_protocol({})


def test_unknown_relax_type_is_rejected():
    with pytest.raises(ValueError, match="relax_type must be one of"):
        relax_settings_from_protocol({"relax_type": "ions"})


def test_positions_volume_has_no_vasp_counterpart():
    with pytest.raises(ValueError, match="no VASP counterpart"):
        relax_settings_from_protocol({"relax_type": "positions_volume"})


def test_unknown_protocol_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        relax_settings_from_protocol({"relax_type": "volume", "force_cutof": 0.05})


def test_relax_method_mapping():
    assert (
        relax_settings_from_protocol({"relax_type": "volume", "relax_method": "rd"})[
            "algo"
        ]
        == "rd"
    )
    with pytest.raises(ValueError, match="relax_method must be one of"):
        relax_settings_from_protocol(
            {"relax_type": "volume", "relax_method": "cg_bfgs"}
        )


def test_force_and_energy_cutoff_are_exclusive():
    with pytest.raises(ValueError, match="only one of 'force_cutoff'"):
        relax_settings_from_protocol(
            {
                "relax_type": "volume",
                "force_cutoff": 0.05,
                "energy_cutoff": 1e-4,
            }
        )


def test_energy_cutoff_alone_is_rejected():
    # aiida-vasp's RelaxOptions.force_cutoff is a non-optional float (default
    # 0.03) and set_ediffg refuses both criteria, so an energy-only criterion
    # cannot be expressed through vasp.relax.
    with pytest.raises(ValueError, match="'energy_cutoff' cannot be used"):
        relax_settings_from_protocol({"relax_type": "volume", "energy_cutoff": 1e-4})


def test_stress_cutoff_is_rejected():
    # VASP stops on EDIFFG (force or energy) only — no stress criterion.
    with pytest.raises(ValueError, match="stress_cutoff"):
        relax_settings_from_protocol({"relax_type": "volume", "stress_cutoff": 1.0})


def test_volume_convergence_needs_a_structure_volume():
    with pytest.raises(ValueError, match="structure volume is unavailable"):
        relax_settings_from_protocol(
            {"relax_type": "volume", "volume_convergence": 0.01}
        )


def test_volume_convergence_can_be_bypassed():
    settings = relax_settings_from_protocol({"relax_type": "volume"})
    # No conversion happened: the plugin's own default (0.01 Å³ absolute)
    # survives because every RelaxOptions field is filled in.
    assert settings["convergence_volume"] == pytest.approx(0.01)


def test_none_skips_the_relaxation():
    settings = relax_settings_from_protocol({"relax_type": "none"})
    assert settings["perform"] is False
    # The remaining fields keep the plugin defaults (perform=False short
    # circuits perform_relaxation, so the dof flags are not consulted).
    assert settings["positions"] is True
    assert settings["shape"] is True
    assert settings["volume"] is True


def test_convergence_positions_passthrough():
    settings = relax_settings_from_protocol(
        {"relax_type": "positions", "convergence_positions": 0.02}
    )
    assert settings["convergence_positions"] == pytest.approx(0.02)


def test_raw_relax_settings_override_translated_keys():
    settings = relax_settings_from_protocol(
        {
            "relax_type": "volume",
            "force_cutoff": 0.05,
            "relax_settings": {
                "force_cutoff": 0.02,
                "convergence_mode": "inout",
                "reuse": True,
            },
        },
        structure_volume=V0,
    )
    assert settings["force_cutoff"] == pytest.approx(0.02)
    assert settings["convergence_mode"] == "inout"
    assert settings["reuse"] is True
    assert settings["volume"] is True  # untouched


def test_unknown_raw_relax_settings_key_is_rejected():
    with pytest.raises(ValueError, match="unknown relax_settings key"):
        relax_settings_from_protocol(
            {"relax_type": "volume", "relax_settings": {"convergence_volum": 0.1}}
        )


def test_translated_settings_pass_the_plugin_validation():
    """The dict must validate against the real VaspRelaxWorkChain port."""
    from aiida_vasp.utils.opthold import RelaxOptions

    settings = relax_settings_from_protocol(VASP_BLOCK, structure_volume=V0)
    # RelaxOptions is a pydantic model: this raises on a bad/typed field.
    RelaxOptions(**settings)


# ---------------------------------------------------------------------------
# AiiDA inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def code_label(aiida_profile, aiida_localhost, request):
    from aiida import orm

    label = "vrelax_" + request.node.name.replace("[", "_").replace("]", "")
    orm.InstalledCode(
        label=label,
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    return f"{label}@{aiida_localhost.label}"


@pytest.fixture
def bcc_uranium():
    from aiida import orm
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def test_adapt_builds_vasp_namespace(code_label, bcc_uranium):
    inputs = _adapter(code_label=code_label).adapt(bcc_uranium).inputs
    entry_point = _adapter(code_label=code_label)._workchain_entry_point()
    assert entry_point == "vasp.relax"

    vasp = inputs["vasp"]
    assert inputs["structure"] is bcc_uranium
    assert vasp["potential_family"].value == "PBE"
    assert vasp["potential_mapping"].get_dict() == {"U": "U"}
    assert vasp["kpoints_spacing"].value == pytest.approx(0.0318)
    # Scheduler options live in the exposed ``vasp.calc`` namespace.
    assert vasp["calc"]["metadata"]["options"]["queue_name"] == "q_test"
    # Foreign keys of a shared metadata preset are dropped (filter).
    assert "max_wallclock_hours" not in vasp["calc"]["metadata"]["options"]


def test_relax_controlled_incar_tags_are_stripped(code_label, bcc_uranium):
    """``nsw: 0`` from the SCF preset must not defeat the relaxation.

    aiida-vasp applies the ``incar`` namespace after the relax
    set-functions, so a leftover NSW would win over ``relax_settings``.
    """
    from aiida_vasp.assistant.parameters import ParametersMassage

    inputs = _adapter(code_label=code_label).adapt(bcc_uranium).inputs
    incar = inputs["vasp"]["parameters"].get_dict()["incar"]
    assert "nsw" not in incar

    # End-to-end: what VASP would actually receive. VaspRelaxWorkChain
    # injects the relax settings as parameters['relax'] before the
    # ParametersMassage conversion (see ``_init_relax_input_additions``).
    parameters = dict(inputs["vasp"]["parameters"].get_dict())
    parameters["relax"] = inputs["relax_settings"].get_dict()
    massaged = ParametersMassage(parameters)
    final = massaged.parameters["incar"]
    assert final["nsw"] == 50  # steps, from max_ionic_steps
    assert final["ibrion"] == 2  # cg
    assert final["isif"] == 7  # volume-only
    assert final["ediffg"] == pytest.approx(-0.05)


def test_relax_settings_dict(code_label, bcc_uranium):
    inputs = _adapter(code_label=code_label).adapt(bcc_uranium).inputs
    settings = inputs["relax_settings"].get_dict()
    assert settings["volume"] is True
    assert settings["positions"] is False
    assert settings["shape"] is False
    assert settings["steps"] == 50
    assert settings["convergence_on"] is True
    # converted with the *input* cell volume (bcc-U, a = 3.45 Å)
    assert settings["convergence_volume"] == pytest.approx(
        0.01 * bcc_uranium.get_cell_volume()
    )


def test_missing_relax_type_fails_loudly(code_label, bcc_uranium):
    with pytest.raises(ValueError, match="'relax_type' is required"):
        _adapter(block={}, code_label=code_label).adapt(bcc_uranium)


def test_missing_incar_is_rejected(code_label, bcc_uranium):
    scf = {k: v for k, v in VASP_SCF.items() if k != "parameters"}
    with pytest.raises(ValueError, match="parameters.incar"):
        _adapter(scf=scf, code_label=code_label).adapt(bcc_uranium)


def test_kpoints_mesh_alternative(code_label, bcc_uranium):
    scf = {k: v for k, v in VASP_SCF.items() if k != "kpoints_spacing"}
    scf["kpoints_mesh"] = [18, 18, 18]
    inputs = _adapter(scf=scf, code_label=code_label).adapt(bcc_uranium).inputs
    assert "kpoints_spacing" not in inputs["vasp"]
    assert inputs["vasp"]["kpoints"].get_kpoints_mesh()[0] == [18, 18, 18]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered_in_relax_orchestrator():
    from aiida_uranium_workflow.schedulers.relax import (
        parse_relax_protocol,
        RelaxWorkflowOrchestrator,
    )

    assert RelaxWorkflowOrchestrator.ADAPTERS["vasp"] is VaspRelaxAdapter
    assert "vasp" in RelaxWorkflowOrchestrator.BACKENDS
    assert RelaxWorkflowOrchestrator.PRESET_SUBKEYS["vasp"] == "scf"

    parsed = parse_relax_protocol({"vasp": VASP_BLOCK, "abacus": {"a": 1}})
    assert parsed["vasp"] == VASP_BLOCK
    assert parsed["abacus"] == {"a": 1}


def test_registered_in_cli_class_map():
    from aiida_uranium_workflow.cli._common import (
        METHOD_SPECS,
        RELAX_CLASS_TO_BACKEND,
    )

    assert RELAX_CLASS_TO_BACKEND["VaspRelaxWorkChain"] == "vasp"
    assert METHOD_SPECS["relax"].backend_to_key["vasp"] == "scf"


# ---------------------------------------------------------------------------
# Report extraction
# ---------------------------------------------------------------------------


def _fake_vasp_node(structure, energies, relax_settings):
    import numpy as np

    class _Outs(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    class _Dict:
        def __init__(self, data):
            self._data = data

        def get_dict(self):
            return self._data

    outs = _Outs(
        structure=structure,
        misc=_Dict({"total_energies": {"energy_extrapolated": energies}}),
    )
    node = type("Node", (), {"outputs": outs})()
    node.inputs = type("Ins", (), {"relax_settings": _Dict(relax_settings)})()
    assert np.asarray(structure.cell).shape == (3, 3)
    return node


def test_report_derive_from_vasp_node():
    from aiida_uranium_workflow.utils.report.relax import _derive_from_node
    from ase.build import bulk

    import numpy as np

    cell = np.asarray(bulk("U", "bcc", a=3.45).cell)

    class _Struct:
        def get_cell_volume(self):
            return float(np.linalg.det(cell))

    _Struct.cell = cell
    node = _fake_vasp_node(
        _Struct(),
        [-10.1, -10.2, -10.3],  # three ionic steps -> last one is the result
        {"positions": False, "shape": False, "volume": True},
    )
    data = _derive_from_node(node, "vasp")
    assert data["energy"] == pytest.approx(-10.3)  # eV, last ionic step
    assert data["volume"] == pytest.approx(float(np.linalg.det(cell)))
    assert data["lattice_constant"] == pytest.approx(float(np.linalg.norm(cell[0])))
    assert data["mode"] == "volume-only relax (ISIF=7)"


def test_report_mode_for_no_relax():
    from aiida_uranium_workflow.utils.report.relax import _vasp_mode

    assert _vasp_mode({"perform": False}) == "no relaxation (static calculation only)"
    assert _vasp_mode({"positions": True, "shape": True, "volume": True}) == (
        "full relax (positions + cell, ISIF=3)"
    )
    assert _vasp_mode(None) is None


def test_settings_are_complete_for_the_workchain_attribute_reads():
    """``VaspRelaxWorkChain.initialize`` builds an ``AttributeDict`` from the
    ``relax_settings`` dict and then reads fields as plain attributes
    (``self.ctx.relax_settings.perform`` in ``perform_relaxation``), so a
    partial dict — e.g. only the positions/shape/volume flags derived from the
    protocol — excepts at start-up with:

        AttributeError: 'AttributeDict' object has no attribute 'perform'

    (regression: run 522755, VaspRelaxWorkChain, excepted before submitting
    any child). Every RelaxOptions field must therefore be present.
    """
    settings = relax_settings_from_protocol(VASP_BLOCK, structure_volume=100.0)
    assert settings["perform"] is True
    for key in (
        "algo",
        "steps",
        "force_cutoff",
        "convergence_on",
        "convergence_max_iterations",
        "double_relax_mode",
        "residual_forces_check",
    ):
        assert key in settings, key
    # Explicit protocol values still win over the plugin defaults.
    assert settings["positions"] is False
    assert settings["shape"] is False
    assert settings["volume"] is True
    assert settings["steps"] == 50
    assert settings["force_cutoff"] == 0.05


def test_none_relax_type_keeps_perform_false():
    settings = relax_settings_from_protocol({"relax_type": "none"}, structure_volume=100.0)
    assert settings["perform"] is False
