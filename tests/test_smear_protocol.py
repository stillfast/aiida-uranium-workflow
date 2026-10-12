"""Tests for the smear protocol parser and its two adapters.

``parameters/smear.yml`` may carry a preset in either of two layouts:

* **per-backend** — one section per backend, holding that backend's own
  keywords and units::

      test_simple:
        abacus:
          smearing_method_list: [mp, gauss]        # ABACUS keywords
          smearing_sigma_list:  [0.0225, 0.0075]   # the unit is Ry
        vasp:
          smear_list: [0, 1]                       # ISMEAR integers
          sigma_list: [0.3, 0.1]                   # the unit is eV

  Both lists reach the WorkChain verbatim — no keyword mapping, no unit
  conversion.

* **legacy shared** — canonical method names plus a sigma in eV, which
  each adapter translates itself (the ``test`` / ``gauss`` presets).

The tests below pin both paths, plus the invariant that the ports the
adapters fill really are the ports their WorkChain declares.
"""

from __future__ import annotations

import pytest

from aiida_uranium_workflow.input_builders.smear.abacus import AbacusAdapter
from aiida_uranium_workflow.input_builders.smear.vasp import VaspAdapter
from aiida_uranium_workflow.schedulers.smear import parse_smear_protocol

#: The ``test_simple`` section of ``parameters/smear.yml`` (per-backend layout).
NATIVE_PROTOCOL = {
    "abacus": {
        "smearing_method_list": ["mp", "gauss"],
        "smearing_sigma_list": [0.0225, 0.0150, 0.0100, 0.0075],
    },
    "vasp": {
        "smear_list": [0, 1],
        "sigma_list": [0.3, 0.2, 0.15, 0.1],
    },
}

#: The legacy shared layout (``test`` / ``gauss`` presets).
LEGACY_PROTOCOL = {
    "smear_list": ["gauss", "mp"],
    "sigma_list": [0.3, 0.1],
}

_ABACUS_SCF = {
    "parameters": {"input": {"calculation": "scf", "basis_type": "lcao"}},
    "kpoints_distance": 0.1,
    "pseudo_family": "sg15_sz",
}
_VASP_SCF = {
    "parameters": {"incar": {}},
    "potential_family": "test_family",
    "potential_mapping": {"U": "U"},
    "kpoints_spacing": 0.0159,
}


def _abacus_adapter(workflow_data):
    return AbacusAdapter(
        code_label="test@localhost",
        software_params=_ABACUS_SCF,
        metadata={},
        workflow_data=workflow_data,
    )


def _vasp_adapter(workflow_data):
    return VaspAdapter(
        code_label="test@localhost",
        software_params=_VASP_SCF,
        metadata={},
        workflow_data=workflow_data,
    )


# ---------------------------------------------------------------------------
# Protocol parser
# ---------------------------------------------------------------------------


class TestParseSmearProtocol:
    """``parse_smear_protocol`` keeps backend-native and legacy layouts apart."""

    def test_per_backend_sections_pass_through(self):
        parsed = parse_smear_protocol(NATIVE_PROTOCOL)
        assert parsed == {
            "smear_lists": {
                "abacus": {
                    "smearing_method_list": ["mp", "gauss"],
                    "smearing_sigma_list": [0.0225, 0.0150, 0.0100, 0.0075],
                },
                "vasp": {
                    "smear_list": [0, 1],
                    "sigma_list": [0.3, 0.2, 0.15, 0.1],
                },
            }
        }

    def test_single_backend_section_is_enough(self):
        """A protocol preset may drive only one backend."""
        parsed = parse_smear_protocol({"abacus": NATIVE_PROTOCOL["abacus"]})
        assert set(parsed["smear_lists"]) == {"abacus"}

    def test_sigma_values_are_floats(self):
        parsed = parse_smear_protocol(
            {"vasp": {"smear_list": [1], "sigma_list": ["0.2"]}}
        )
        assert parsed["smear_lists"]["vasp"]["sigma_list"] == [0.2]

    def test_empty_sigma_list_raises(self):
        with pytest.raises(ValueError, match="smearing_sigma_list"):
            parse_smear_protocol(
                {"abacus": {"smearing_method_list": ["mp"], "smearing_sigma_list": []}}
            )

    def test_legacy_shared_pair_keeps_historical_shape(self):
        parsed = parse_smear_protocol(LEGACY_PROTOCOL)
        assert parsed == {"smear_lists": {"smear": ["gauss", "mp"], "sigma": [0.3, 0.1]}}

    def test_empty_protocol_raises(self):
        with pytest.raises(ValueError, match="per-backend section"):
            parse_smear_protocol({})

    def test_packaged_test_simple_preset_parses(self):
        """Whatever layout the shipped ``test_simple`` preset carries, it parses.

        ``parameters/`` is intentionally **not** versioned with this code
        (AGENT.md §2.7), so the packaged file may hold either layout: the
        per-backend one (``abacus`` / ``vasp`` sections) or the legacy
        shared ``smear_list`` / ``sigma_list`` pair.  Both must yield
        non-empty lists — that is the contract the adapters rely on.
        """
        from aiida_uranium_workflow.schedulers.smear import _SMEAR_PROTOCOL_KEYS
        from aiida_uranium_workflow.utils.common import PROTOCOL_DIR
        from aiida_uranium_workflow.utils.yamlio import read_yaml

        table = read_yaml(PROTOCOL_DIR / "smear.yml")
        parsed = parse_smear_protocol(table["test_simple"])
        lists = parsed["smear_lists"]
        assert lists, "test_simple must carry at least one backend grid"

        if "smear" in lists:  # legacy shared pair
            assert lists["smear"]
            assert lists["sigma"]
            return

        for backend, entry in lists.items():
            method_key, sigma_key = _SMEAR_PROTOCOL_KEYS[backend]
            assert entry[method_key], f"{backend} {method_key} is empty"
            assert entry[sigma_key], f"{backend} {sigma_key} is empty"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class TestAbacusAdapterGrid:
    """ABACUS consumes its own section verbatim; the legacy pair is converted."""

    def test_native_section_needs_no_conversion(self):
        method, sigma = _abacus_adapter(
            parse_smear_protocol(NATIVE_PROTOCOL)
        )._prepare_workflow_inputs()
        assert method == ["mp", "gauss"]
        # Ry stays Ry — 0.0225 Ry must NOT be re-scaled by 13.6.
        assert sigma == [0.0225, 0.0150, 0.0100, 0.0075]

    def test_legacy_pair_is_translated(self):
        method, sigma = _abacus_adapter(
            parse_smear_protocol(LEGACY_PROTOCOL)
        )._prepare_workflow_inputs()
        assert method == ["gauss", "mp"]
        assert sigma == pytest.approx([0.3 / 13.605693, 0.1 / 13.605693])

    def test_missing_workflow_data_yields_empty_grid(self):
        assert _abacus_adapter({})._prepare_workflow_inputs() == ([], [])


class TestVaspAdapterGrid:
    """VASP consumes its own section verbatim; the legacy pair is converted."""

    def test_native_section_needs_no_conversion(self):
        smear, sigma = _vasp_adapter(
            parse_smear_protocol(NATIVE_PROTOCOL)
        )._prepare_workflow_inputs()
        assert smear == [0, 1]
        assert sigma == [0.3, 0.2, 0.15, 0.1]

    def test_legacy_pair_maps_method_names_to_ismear(self):
        smear, sigma = _vasp_adapter(
            parse_smear_protocol(LEGACY_PROTOCOL)
        )._prepare_workflow_inputs()
        assert smear == [0, 1]
        assert sigma == [0.3, 0.1]

    def test_missing_workflow_data_yields_empty_grid(self):
        assert _vasp_adapter({})._prepare_workflow_inputs() == ([], [])


class TestAdapterPortsMatchWorkchainSpec:
    """Every input the adapter emits must be a port its WorkChain declares.

    The generic :meth:`SoftwareAdapter.adapt` writes the hard-coded
    ``smear`` / ``sigma`` port names; both smear adapters override it so
    the grid reaches the backend-natively named ports
    (``smearing_method_list`` / ``smearing_sigma_list`` for ABACUS,
    ``smear_list`` / ``sigma_list`` for VASP).
    """

    @pytest.fixture(autouse=True)
    def _stub_code_loading(self, monkeypatch):
        """``adapt`` resolves an AiiDA ``Code`` — stub it out (no profile)."""
        monkeypatch.setattr("aiida.orm.load_code", lambda label: None, raising=True)

    @staticmethod
    def _spec_ports(entry_point):
        from aiida.plugins import WorkflowFactory

        return set(WorkflowFactory(entry_point).spec().inputs.keys())

    def test_abacus_ports(self):
        adapted = _abacus_adapter(parse_smear_protocol(NATIVE_PROTOCOL)).adapt(
            structure=None
        )
        assert set(adapted.inputs) <= self._spec_ports("uranium.smear.abacus")
        assert adapted.inputs["smearing_method_list"].get_list() == ["mp", "gauss"]
        assert adapted.inputs["smearing_sigma_list"].get_list() == [
            0.0225,
            0.015,
            0.01,
            0.0075,
        ]
        assert "smear" not in adapted.inputs and "sigma" not in adapted.inputs

    def test_vasp_ports(self):
        adapted = _vasp_adapter(parse_smear_protocol(NATIVE_PROTOCOL)).adapt(
            structure=None
        )
        assert set(adapted.inputs) <= self._spec_ports("uranium.smear.vasp")
        assert adapted.inputs["smear_list"].get_list() == [0, 1]
        assert adapted.inputs["sigma_list"].get_list() == [0.3, 0.2, 0.15, 0.1]
        assert "smear" not in adapted.inputs and "sigma" not in adapted.inputs
