"""Tests for the VASP ``banddos`` backend (adapter + plot extractors).

The banddos method used to support ABACUS and FLEUR only — the ``vasp:``
stubs in ``parameters/banddos.yml`` were empty placeholders. VASP is now
wired through aiida-vasp's ``vasp.v2.bands`` (``VaspBandsWorkChain``),
whose ``band_settings`` vocabulary already matches the protocol file
(``band_mode: seekpath-aiida``, ``band_kpoints_distance``,
``dos_kpoints_distance``, ``symprec``, ``additional_band_analysis_parameters``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.banddos.vasp import (
    DEFAULT_BAND_SETTINGS,
    VaspBandAdapter,
)

VASP_SCF = {
    "parameters": {"incar": {"encut": 400, "ispin": 1}},
    "potential_family": "PBE",
    "potential_mapping": {"U": "U"},
    "kpoints_spacing": 0.0318,
}


def _adapter(vasp_block: dict | None, scf: dict | None = None, code_label="x"):
    return VaspBandAdapter(
        code_label=code_label,
        software_params=dict(scf or VASP_SCF),
        metadata={"options": {"queue_name": "q_test", "max_wallclock_seconds": 600}},
        workflow_data={"vasp": vasp_block} if vasp_block is not None else {},
        extra_codes={},
    )


# ---------------------------------------------------------------------------
# Protocol block -> band_settings mapping
# ---------------------------------------------------------------------------


def test_bands_only():
    settings = _adapter({"run_bands": True, "run_dos": False})._band_settings()
    assert settings["only_dos"] is False
    assert settings["run_dos"] is False


def test_bands_and_dos():
    settings = _adapter({"run_bands": True, "run_dos": True})._band_settings()
    assert settings["only_dos"] is False
    assert settings["run_dos"] is True


def test_dos_only():
    settings = _adapter({"run_bands": False, "run_dos": True})._band_settings()
    assert settings["only_dos"] is True
    assert settings["run_dos"] is True


def test_proj_dos_is_rejected():
    # LORBIT=11 makes vasprun.xml's <partial> block 1+16 columns wide for
    # f-electron POTCARs (U), which the pinned parsevasp cannot read — the
    # whole vasprun parse then fails and the WorkChain excepts. Refuse loudly.
    with pytest.raises(ValueError, match="run_proj_dos is not supported"):
        _adapter({"run_bands": True, "run_proj_dos": True})._band_settings()


def test_defaults_when_block_is_empty():
    settings = _adapter({})._band_settings()
    # default: bands only
    assert settings["run_dos"] is False
    assert settings["only_dos"] is False
    for key, value in DEFAULT_BAND_SETTINGS.items():
        assert settings[key] == value


def test_nothing_to_do_is_rejected():
    with pytest.raises(ValueError, match="nothing to compute"):
        _adapter({"run_bands": False, "run_dos": False})._band_settings()


def test_user_overrides_are_forwarded():
    settings = _adapter(
        {
            "run_bands": True,
            "band_mode": "sumo",
            "band_kpoints_distance": 0.05,
            "dos_kpoints_distance": 0.02,
            "symprec": 1e-3,
            "kpoints_per_split": 40,
            "additional_band_analysis_parameters": {"reference_distance": 0.03},
        }
    )._band_settings()
    assert settings["band_mode"] == "sumo"
    assert settings["band_kpoints_distance"] == 0.05
    assert settings["dos_kpoints_distance"] == 0.02
    assert settings["symprec"] == 1e-3
    assert settings["kpoints_per_split"] == 40
    assert settings["additional_band_analysis_parameters"] == {"reference_distance": 0.03}


# ---------------------------------------------------------------------------
# AiiDA inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def code_label(aiida_profile, aiida_localhost, request):
    from aiida import orm

    label = "bd_" + request.node.name.replace("[", "_").replace("]", "")
    orm.InstalledCode(
        label=label, computer=aiida_localhost, filepath_executable="/bin/true"
    ).store()
    return f"{label}@{aiida_localhost.label}"


@pytest.fixture
def u_structure(aiida_profile):
    from aiida import orm
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45, cubic=True))


def test_adapt_builds_scf_namespace(code_label, u_structure):
    adapted = _adapter({"run_bands": True}, code_label=code_label).adapt(u_structure)
    inputs = adapted.inputs
    assert adapted.workchain_cls.__name__ == "VaspBandsWorkChain"
    assert inputs["structure"] is u_structure

    scf = inputs["scf"]
    assert scf["code"].label == code_label.split("@")[0]
    assert scf["parameters"].get_dict()["incar"]["encut"] == 400
    assert scf["potential_family"].value == "PBE"
    assert scf["potential_mapping"].get_dict() == {"U": "U"}
    assert scf["kpoints_spacing"].value == pytest.approx(0.0318)
    # Scheduler options survive under the scf calc namespace.
    assert scf["calc"]["metadata"]["options"]["queue_name"] == "q_test"

    band_settings = inputs["band_settings"].get_dict()
    assert band_settings["only_dos"] is False
    assert band_settings["run_dos"] is False


def test_adapt_kpoints_mesh(code_label, u_structure):
    scf = dict(VASP_SCF)
    scf.pop("kpoints_spacing")
    scf["kpoints_mesh"] = [18, 18, 18]
    adapted = _adapter({"run_bands": True}, scf=scf, code_label=code_label).adapt(u_structure)
    kpoints = adapted.inputs["scf"]["kpoints"]
    assert kpoints.get_kpoints_mesh() == ([18, 18, 18], [0.0, 0.0, 0.0])


def test_adapt_proj_dos_refused_before_building(code_label, u_structure):
    with pytest.raises(ValueError, match="run_proj_dos is not supported"):
        _adapter({"run_bands": False, "run_proj_dos": True}, code_label=code_label).adapt(
            u_structure
        )


def test_adapt_dos_sets_dense_nedos(code_label, u_structure):
    # VASP's default NEDOS (301 points over the whole band range) leaves the
    # Fermi-surface region almost empty in the plotted DOS.
    adapted = _adapter({"run_bands": False, "run_dos": True}, code_label=code_label).adapt(
        u_structure
    )
    assert adapted.inputs["dos"]["parameters"].get_dict()["incar"]["nedos"] == 3001


def test_adapt_dos_nedos_is_configurable(code_label, u_structure):
    adapted = _adapter(
        {"run_bands": False, "run_dos": True, "nedos": 9001}, code_label=code_label
    ).adapt(u_structure)
    assert adapted.inputs["dos"]["parameters"].get_dict()["incar"]["nedos"] == 9001


def test_adapt_plain_dos_keeps_constant_charge(code_label, u_structure):
    # aiida-vasp crashes in run_bands_dos when run_dos is true but no ``dos``
    # namespace is supplied (its fallback settings dict is a plain dict), so
    # the adapter always provides one — with ICHARG=11 bits and an ``incar``
    # block for the parameters validator.
    adapted = _adapter({"run_bands": True, "run_dos": True}, code_label=code_label).adapt(
        u_structure
    )
    dos = adapted.inputs["dos"]
    params = dos["parameters"].get_dict()
    assert params["charge"] == {"constant_charge": True}
    assert params["incar"]["nedos"] == 3001
    assert "lorbit" not in params["incar"]


def test_adapt_band_only_has_no_dos_namespace(code_label, u_structure):
    adapted = _adapter({"run_bands": True, "run_dos": False}, code_label=code_label).adapt(u_structure)
    assert "dos" not in adapted.inputs


# ---------------------------------------------------------------------------
# Protocol parsing + orchestrator registration
# ---------------------------------------------------------------------------


def test_parse_protocol_keeps_vasp_block():
    from aiida_uranium_workflow.schedulers.banddos import parse_banddos_protocol

    parsed = parse_banddos_protocol(
        {
            "abacus": {"run_bands": True},
            "vasp": {"run_bands": True, "run_dos": True},
        }
    )
    assert parsed["vasp"] == {"run_bands": True, "run_dos": True}
    assert parsed["band_settings"] == {"run_bands": True}


def test_parse_protocol_tolerates_empty_placeholder():
    from aiida_uranium_workflow.schedulers.banddos import parse_banddos_protocol

    # A bare ``vasp:`` YAML key loads as None — must not crash.
    parsed = parse_banddos_protocol({"abacus": None, "fleur": None, "vasp": None})
    assert parsed == {"band_settings": {}, "fleur": {}, "vasp": {}}


def test_orchestrator_registers_vasp():
    from aiida_uranium_workflow.schedulers.banddos import BanddosWorkflowOrchestrator

    assert "vasp" in BanddosWorkflowOrchestrator.BACKENDS
    assert BanddosWorkflowOrchestrator.ADAPTERS["vasp"] is VaspBandAdapter
    assert BanddosWorkflowOrchestrator.PRESET_SUBKEYS["vasp"] == "scf"


def test_cli_maps_vasp_bands_workchain():
    from aiida_uranium_workflow.cli._common import BANDDOS_CLASS_TO_BACKEND

    assert BANDDOS_CLASS_TO_BACKEND["VaspBandsWorkChain"] == "vasp"


# ---------------------------------------------------------------------------
# Plot extractors
# ---------------------------------------------------------------------------


def _make_wc(bands_data, dos_data=None):
    """A minimal WorkChain stand-in: the extractors only touch ``outputs``/``pk``."""
    outputs = SimpleNamespace(band_structure=bands_data)
    if dos_data is not None:
        outputs.dos = dos_data
    return SimpleNamespace(outputs=outputs, pk=42)


def _bands_node(energies, kpoints=None, fermi=1.0, labels=None):
    """Wrap an ``(nk, nbands)`` / ``(nspin, nk, nbands)`` array in a BandsData."""
    from aiida import orm

    arr = np.asarray(energies)
    nk = arr.shape[1] if arr.ndim == 3 else arr.shape[0]
    if kpoints is None:
        kpoints = [[0.5 * i, 0.0, 0.0] for i in range(nk)]
    node = orm.BandsData()
    node.set_kpoints(kpoints)
    node.set_bands(energies)
    node.base.attributes.set("fermi_level", fermi)
    if labels:
        node.base.attributes.set("labels", labels)
    return node


def _dos_node(energy, tdos, pdos=None):
    from aiida import orm

    node = orm.ArrayData()
    node.set_array("energy", np.asarray(energy, dtype=float))
    node.set_array("tdos", np.asarray(tdos, dtype=float))
    if pdos is not None:
        node.set_array("pdos", np.asarray(pdos, dtype=float))
    return node


def test_extract_vasp_band_only():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp

    energies = np.array([[0.0, 2.0], [0.5, 2.5]])  # (nk, nbands)
    wc = _make_wc(_bands_node(energies, labels=[[0, "G"], [1, "X"]]))
    data = extract_vasp(wc)
    band = data["band"]
    assert band.energies.shape == (2, 2)
    # BandData is (nbands, nk) — the transpose of aiida's (nk, nbands).
    np.testing.assert_allclose(band.energies, energies.T)
    assert band.labels == ["G", "X"]
    assert band.label_numbers == [0, 1]
    assert band.fermi_energy == 1.0
    assert band.workchain_pk == 42
    assert data["dos"] is None


def test_extract_vasp_spin_polarised_bands_are_averaged():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp

    # (nspin, nk, nbands) -> mean over the exchange-split channels
    energies = np.array(
        [
            [[0.0, 2.0], [0.5, 2.5]],
            [[-0.2, 1.8], [0.3, 2.3]],
        ]
    )
    wc = _make_wc(_bands_node(energies))
    band = extract_vasp(wc)["band"]
    np.testing.assert_allclose(band.energies, energies.mean(axis=0).T)


def test_extract_vasp_noncollinear_dos():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp

    energy = np.linspace(-2, 2, 5)
    tdos = np.array([[1.0, 2.0, 3.0, 2.0, 1.0], [0.5, 1.0, 1.5, 1.0, 0.5]])
    wc = _make_wc(_bands_node(np.zeros((5, 1))), _dos_node(energy, tdos))
    dos = extract_vasp(wc)["dos"]
    np.testing.assert_allclose(dos.total, tdos[0] + tdos[1])
    assert set(dos.spin) == {"dos_up", "dos_dn"}
    np.testing.assert_allclose(dos.spin["dos_up"], tdos[0])


def test_extract_vasp_spin_summed_dos():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp

    energy = np.linspace(-2, 2, 5)
    tdos = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    wc = _make_wc(_bands_node(np.zeros((5, 1))), _dos_node(energy, tdos))
    dos = extract_vasp(wc)["dos"]
    np.testing.assert_allclose(dos.total, tdos)
    assert dos.spin == {}


def test_extract_vasp_pdos():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp_pdos

    energy = np.linspace(-2, 2, 4)
    # (natoms=2, nenergy=4, 9) — l-major s / p / d / d / d / d / d / d / d
    pdos = np.zeros((2, 4, 9))
    pdos[:, :, 0] = 1.0  # s on both atoms
    pdos[:, :, 1:4] = 2.0  # p
    pdos[:, :, 4:] = 3.0  # d
    wc = _make_wc(_bands_node(np.zeros((4, 1))), _dos_node(energy, np.zeros(4), pdos))
    data = extract_vasp_pdos(wc)
    np.testing.assert_allclose(data.orbitals["s"], np.full(4, 2.0))
    np.testing.assert_allclose(data.orbitals["p"], np.full(4, 12.0))
    np.testing.assert_allclose(data.orbitals["d"], np.full(4, 30.0))
    np.testing.assert_allclose(data.total, np.full(4, 44.0))


def test_extract_vasp_pdos_without_lorbit_is_explained():
    from aiida_uranium_workflow.utils.plot.extract import extract_vasp_pdos

    energy = np.linspace(-2, 2, 4)
    wc = _make_wc(_bands_node(np.zeros((4, 1))), _dos_node(energy, np.zeros(4)))
    with pytest.raises(ValueError, match="LORBIT=11"):
        extract_vasp_pdos(wc)


def test_extractors_registry():
    from aiida_uranium_workflow.utils.plot.extract import EXTRACTORS, PDOS_EXTRACTORS

    assert EXTRACTORS["VaspBandsWorkChain"] is not None
    assert PDOS_EXTRACTORS["VaspBandsWorkChain"] is not None
