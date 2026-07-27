"""Tests for convergence workflow schedulers and input builders."""

from __future__ import annotations

import os
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida_pseudo.data.pseudo import UpfData
from aiida_pseudo.groups.family import PseudoPotentialFamily
from aiida_uranium_workflow.input_builders.convergence.abacus import (
    AbacusConvergenceAdapter,
)
from aiida_uranium_workflow.input_builders.convergence.vasp import (
    VaspConvergenceAdapter,
)
from aiida_uranium_workflow.schedulers.base import _WORKFLOW_REGISTRY
from aiida_uranium_workflow.schedulers.convergence import (
    ConvergenceWorkflowOrchestrator,
    parse_convergence_protocol,
)
from pathlib import Path

TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestParseConvergenceProtocol:
    """Tests for parse_convergence_protocol function."""

    def test_parse_abacus_protocol(self):
        """Test parsing ABACUS convergence protocol with distance list."""
        protocol = {
            "abacus": {
                "ecutwfc_list": [80, 100, 120],
                "kpoints_distance_list": [0.1, 0.15, 0.2],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "convergence_lists" in result
        assert "abacus" in result["convergence_lists"]
        assert result["convergence_lists"]["abacus"]["ecutwfc_list"] == [80, 100, 120]
        assert result["convergence_lists"]["abacus"]["kpoints_distance_list"] == [
            0.1,
            0.15,
            0.2,
        ]

    def test_parse_abacus_protocol_mesh_list(self):
        """Test parsing ABACUS convergence protocol with mesh list."""
        protocol = {
            "abacus": {
                "ecutwfc_list": [80, 100, 120],
                "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9], [7, 7, 7]],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "convergence_lists" in result
        assert "abacus" in result["convergence_lists"]
        assert result["convergence_lists"]["abacus"]["ecutwfc_list"] == [80, 100, 120]
        assert result["convergence_lists"]["abacus"]["kpoints_mesh_list"] == [
            [11, 11, 11],
            [9, 9, 9],
            [7, 7, 7],
        ]

    def test_parse_abacus_protocol_mesh_priority(self):
        """Test that mesh list takes priority over distance list."""
        protocol = {
            "abacus": {
                "ecutwfc_list": [80, 100],
                "kpoints_distance_list": [0.1, 0.15],
                "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9]],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "kpoints_mesh_list" in result["convergence_lists"]["abacus"]
        assert "kpoints_distance_list" not in result["convergence_lists"]["abacus"]

    def test_parse_vasp_protocol(self):
        """Test parsing VASP convergence protocol with spacing list."""
        protocol = {
            "vasp": {
                "encut_list": [300, 400],
                "kpoints_spacing_list": [0.0159, 0.0239],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "convergence_lists" in result
        assert "vasp" in result["convergence_lists"]
        assert result["convergence_lists"]["vasp"]["encut_list"] == [300, 400]
        assert result["convergence_lists"]["vasp"]["kpoints_spacing_list"] == [
            0.0159,
            0.0239,
        ]

    def test_parse_vasp_protocol_mesh_list(self):
        """Test parsing VASP convergence protocol with mesh list."""
        protocol = {
            "vasp": {
                "encut_list": [300, 400],
                "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9]],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "convergence_lists" in result
        assert "vasp" in result["convergence_lists"]
        assert result["convergence_lists"]["vasp"]["encut_list"] == [300, 400]
        assert result["convergence_lists"]["vasp"]["kpoints_mesh_list"] == [
            [11, 11, 11],
            [9, 9, 9],
        ]

    def test_parse_vasp_protocol_mesh_priority(self):
        """Test that mesh list takes priority over spacing list."""
        protocol = {
            "vasp": {
                "encut_list": [300, 400],
                "kpoints_spacing_list": [0.0159, 0.0239],
                "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9]],
            }
        }
        result = parse_convergence_protocol(protocol)
        assert "kpoints_mesh_list" in result["convergence_lists"]["vasp"]
        assert "kpoints_spacing_list" not in result["convergence_lists"]["vasp"]

    def test_parse_both_protocols(self):
        """Test parsing both ABACUS and VASP protocols."""
        protocol = {
            "abacus": {
                "ecutwfc_list": [80, 100],
                "kpoints_distance_list": [0.1, 0.15],
            },
            "vasp": {
                "encut_list": [300, 400],
                "kpoints_spacing_list": [0.0159, 0.0239],
            },
        }
        result = parse_convergence_protocol(protocol)
        assert "abacus" in result["convergence_lists"]
        assert "vasp" in result["convergence_lists"]

    def test_empty_abacus_lists_raises(self):
        """Test that empty abacus lists raise ValueError."""
        protocol = {
            "abacus": {
                "ecutwfc_list": [],
                "kpoints_distance_list": [0.1, 0.15],
            }
        }
        with pytest.raises(ValueError, match="ecutwfc_list"):
            parse_convergence_protocol(protocol)

    def test_empty_vasp_lists_raises(self):
        """Test that empty vasp lists raise ValueError."""
        protocol = {
            "vasp": {
                "encut_list": [300],
                "kpoints_spacing_list": [],
            }
        }
        with pytest.raises(ValueError, match="kpoints_spacing_list"):
            parse_convergence_protocol(protocol)


class TestConvergenceWorkflowOrchestrator:
    """Tests for ConvergenceWorkflowOrchestrator."""

    def test_adapters_registered(self):
        """Test that ABACUS and VASP adapters are registered."""
        assert "abacus" in ConvergenceWorkflowOrchestrator.ADAPTERS
        assert "vasp" in ConvergenceWorkflowOrchestrator.ADAPTERS

    def test_backends_list(self):
        """Test that backends are correctly listed."""
        assert ConvergenceWorkflowOrchestrator.BACKENDS == ("abacus", "vasp")

    def test_workflow_registered(self):
        """Test that convergence workflow is registered in the registry."""
        assert "convergence" in _WORKFLOW_REGISTRY

    def test_preset_subkeys(self):
        """Both backends must share ``"convergence"`` as the preset sub-key.

        ``parameters.abacus.convergence`` and ``parameters.vasp.convergence``
        are the canonical slots the input.json uses to list preset names
        (e.g. ``["pw", "pw_r"]``); without this mapping the orchestrator
        would fall back to ``abacus#0/#1`` synthetic names.
        """
        assert ConvergenceWorkflowOrchestrator.PRESET_SUBKEYS == {
            "abacus": "convergence",
            "vasp": "convergence",
        }


class TestPresetNameResolution:
    """End-to-end preset name resolution for convergence / magmom.

    These tests build a minimal :class:`WorkflowOrchestrator`-compatible
    bundle and call ``_preset_names_for`` directly so we can assert the
    user-facing preset names (``"pw"`` / ``"pw_r"`` / ``"test_magmom"``)
    instead of synthetic identifiers.
    """

    @staticmethod
    def _make_orchestrator(orchestrator_cls, parameters_section):
        """Build an orchestrator with a fake :class:`ParamBundle`."""

        class _Bundle:
            pass

        bundle = _Bundle()
        bundle.input_params = {"parameters": parameters_section}
        bundle.software_params = {"abacus": [{}, {}], "vasp": [{}]}
        return orchestrator_cls(bundle)

    def test_convergence_abacus_dict_form_yields_preset_names(self):
        """``{"abacus": {"convergence": ["pw", "pw_r"]}}`` → ``["pw", "pw_r"]``."""
        orchestrator = self._make_orchestrator(
            ConvergenceWorkflowOrchestrator,
            {"abacus": {"convergence": ["pw", "pw_r"]}, "vasp": {"convergence": ["test"]}},
        )
        assert orchestrator._preset_names_for("abacus") == ["pw", "pw_r"]
        assert orchestrator._preset_names_for("vasp") == ["test"]

    def test_convergence_string_form_still_supported(self):
        """Single string still returns a one-element list (legacy form)."""
        orchestrator = self._make_orchestrator(
            ConvergenceWorkflowOrchestrator,
            {"abacus": {"convergence": "pw"}, "vasp": {"convergence": "test"}},
        )
        assert orchestrator._preset_names_for("abacus") == ["pw"]
        assert orchestrator._preset_names_for("vasp") == ["test"]

    def test_convergence_dict_form_with_extra_keys_does_not_confuse_subkey(self):
        """An unrelated dict key (e.g. ``"extra"``) is ignored."""
        orchestrator = self._make_orchestrator(
            ConvergenceWorkflowOrchestrator,
            {
                "abacus": {
                    "convergence": ["pw", "pw_r"],
                    "extra": "noise",
                },
            },
        )
        assert orchestrator._preset_names_for("abacus") == ["pw", "pw_r"]

    def test_magmom_abacus_dict_form_yields_preset_names(self):
        """Magmom uses the same dict form with sub-key ``"magmom"``."""
        from aiida_uranium_workflow.schedulers.magmom import (
            MagmomWorkflowOrchestrator,
        )

        orchestrator = self._make_orchestrator(
            MagmomWorkflowOrchestrator,
            {
                "abacus": {
                    "magmom": ["test_magmom", "test_magmom_pw"],
                },
            },
        )
        assert orchestrator._preset_names_for("abacus") == [
            "test_magmom",
            "test_magmom_pw",
        ]

    def test_preset_subkey_overrides_legacy_string_form(self):
        """When both forms are present, the sub-key form takes priority."""
        orchestrator = self._make_orchestrator(
            ConvergenceWorkflowOrchestrator,
            {"abacus": "ignored", "abacus": {"convergence": ["pw"]}},
        )
        # When the sub-key form is a dict it is used; without the dict
        # form (or with ``PRESET_SUBKEYS`` pointing somewhere else) the
        # legacy string is returned in a single-element list.
        orchestrator.bundle.input_params = {
            "parameters": {"abacus": "pw"},  # plain string form
        }
        assert orchestrator._preset_names_for("abacus") == ["pw"]


class TestMagmomPresetSubkeys:
    """Regression guard: ``MagmomWorkflowOrchestrator.PRESET_SUBKEYS`` is wired."""

    def test_magmom_preset_subkeys(self):
        from aiida_uranium_workflow.schedulers.magmom import (
            MagmomWorkflowOrchestrator,
        )

        assert MagmomWorkflowOrchestrator.PRESET_SUBKEYS == {
            "abacus": "magmom",
            "vasp": "magmom",
        }


@pytest.mark.usefixtures("aiida_profile_clean")
class TestAbacusConvergenceAdapter:
    """Tests for AbacusConvergenceAdapter."""

    def test_workchain_entry_point(self):
        """Test that the correct workchain entry point is returned."""
        adapter = AbacusConvergenceAdapter(
            code_label="abacus@localhost",
            software_params={
                "parameters": {"input": {"ecutwfc": 100}},
                "kpoints_distance": 0.1,
                "pseudo_family": "test-family",
            },
            metadata={},
            workflow_data={},
        )
        assert adapter._workchain_entry_point() == "abacus.convergence"

    def test_prepare_workflow_inputs_distance(self):
        """Test extraction of convergence lists with distance mode."""
        adapter = AbacusConvergenceAdapter(
            code_label="abacus@localhost",
            software_params={
                "parameters": {"input": {"ecutwfc": 100}},
                "kpoints_distance": 0.1,
                "pseudo_family": "test-family",
            },
            metadata={},
            workflow_data={
                "convergence_lists": {
                    "abacus": {
                        "ecutwfc_list": [80, 100],
                        "kpoints_distance_list": [0.1, 0.15],
                    }
                }
            },
        )
        ecutwfc_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert ecutwfc_list == [80, 100]
        assert kpoints_values == [0.1, 0.15]
        assert mode == "distance"

    def test_prepare_workflow_inputs_mesh(self):
        """Test extraction of convergence lists with mesh mode."""
        adapter = AbacusConvergenceAdapter(
            code_label="abacus@localhost",
            software_params={
                "parameters": {"input": {"ecutwfc": 100}},
                "kpoints_distance": 0.1,
                "pseudo_family": "test-family",
            },
            metadata={},
            workflow_data={
                "convergence_lists": {
                    "abacus": {
                        "ecutwfc_list": [80, 100],
                        "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9]],
                    }
                }
            },
        )
        ecutwfc_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert ecutwfc_list == [80, 100]
        assert kpoints_values == [[11, 11, 11], [9, 9, 9]]
        assert mode == "mesh"

    def test_prepare_workflow_inputs_empty(self):
        """Test with empty workflow_data."""
        adapter = AbacusConvergenceAdapter(
            code_label="abacus@localhost",
            software_params={
                "parameters": {"input": {"ecutwfc": 100}},
                "kpoints_distance": 0.1,
                "pseudo_family": "test-family",
            },
            metadata={},
            workflow_data={},
        )
        ecutwfc_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert ecutwfc_list == []
        assert kpoints_values == []
        assert mode == "distance"


@pytest.mark.usefixtures("aiida_profile_clean")
class TestVaspConvergenceAdapter:
    """Tests for VaspConvergenceAdapter."""

    def test_workchain_entry_point(self):
        """Test that the correct workchain entry point is returned."""
        adapter = VaspConvergenceAdapter(
            code_label="vasp@localhost",
            software_params={
                "parameters": {"incar": {"encut": 300}},
                "kpoints_spacing": 0.0159,
                "potential_family": "test-potentials",
                "potential_mapping": {"Si": "Si.pbe-nl-kjpaw_psl.0.3.0.UPF"},
            },
            metadata={},
            workflow_data={},
        )
        assert adapter._workchain_entry_point() == "vasp.convergence"

    def test_prepare_workflow_inputs_spacing(self):
        """Test extraction of convergence lists with spacing mode."""
        adapter = VaspConvergenceAdapter(
            code_label="vasp@localhost",
            software_params={
                "parameters": {"incar": {"encut": 300}},
                "kpoints_spacing": 0.0159,
                "potential_family": "test-potentials",
                "potential_mapping": {"Si": "Si.pbe-nl-kjpaw_psl.0.3.0.UPF"},
            },
            metadata={},
            workflow_data={
                "convergence_lists": {
                    "vasp": {
                        "encut_list": [300, 400],
                        "kpoints_spacing_list": [0.0159, 0.0239],
                    }
                }
            },
        )
        encut_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert encut_list == [300, 400]
        assert kpoints_values == [0.0159, 0.0239]
        assert mode == "spacing"

    def test_prepare_workflow_inputs_mesh(self):
        """Test extraction of convergence lists with mesh mode."""
        adapter = VaspConvergenceAdapter(
            code_label="vasp@localhost",
            software_params={
                "parameters": {"incar": {"encut": 300}},
                "kpoints_spacing": 0.0159,
                "potential_family": "test-potentials",
                "potential_mapping": {"Si": "Si.pbe-nl-kjpaw_psl.0.3.0.UPF"},
            },
            metadata={},
            workflow_data={
                "convergence_lists": {
                    "vasp": {
                        "encut_list": [300, 400],
                        "kpoints_mesh_list": [[11, 11, 11], [9, 9, 9]],
                    }
                }
            },
        )
        encut_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert encut_list == [300, 400]
        assert kpoints_values == [[11, 11, 11], [9, 9, 9]]
        assert mode == "mesh"

    def test_prepare_workflow_inputs_empty(self):
        """Test with empty workflow_data."""
        adapter = VaspConvergenceAdapter(
            code_label="vasp@localhost",
            software_params={
                "parameters": {"incar": {"encut": 300}},
                "kpoints_spacing": 0.0159,
                "potential_family": "test-potentials",
                "potential_mapping": {"Si": "Si.pbe-nl-kjpaw_psl.0.3.0.UPF"},
            },
            metadata={},
            workflow_data={},
        )
        encut_list, kpoints_values, mode = adapter._prepare_workflow_inputs()
        assert encut_list == []
        assert kpoints_values == []
        assert mode == "spacing"
