"""Edge-case tests for :class:`ConfigLoader`.

Covers the parts of ``utils/config.py`` not exercised by
``test_config_loader.py``:

* ``_validate_static`` (string / list / empty list / non-string elements /
  wrong type / missing key)
* ``_resolve_presets`` sibiling-key preservation (the nested-vs-flat
  layout convention)
* ``_coerce_preset_names`` rejection path for dict sub-categories
* ``_load_all_software_params`` skipping workflow-protocol slots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pytest

from aiida_uranium_workflow.utils.config import ConfigLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_input(tmp_path: Path, body: Dict[str, Any]) -> Path:
    p = tmp_path / "input.json"
    p.write_text(json.dumps(body))
    return p


def _minimal_input(**overrides: Any) -> Dict[str, Any]:
    """Build the minimum input.json that ``_validate_static`` accepts.

    The defaults are designed to make every override visible: change a
    single key to exercise one branch of validation, leaving the rest
    alone.
    """
    body: Dict[str, Any] = {
        "workflow": "convergence",
        "parameters": {
            "convergence": "test",
            "abacus": "test",
            "vasp": "test",
        },
        "static": {
            "structure": "bcc-uranium",
            "metadata": "yeesuan",
        },
        "profile": "aiida_profile",
        "code": {
            "abacus": "abacus@yeesuan",
            "vasp": "vaspstd@yeesuan",
        },
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# _validate_static
# ---------------------------------------------------------------------------


class TestValidateStatic:
    """``static.structure`` accepts a string or a list of strings."""

    def test_string_accepted(self, tmp_path: Path) -> None:
        """Legacy single-structure form: ``"structure": "bcc-uranium"``."""
        path = _write_input(tmp_path, _minimal_input())
        bundle = ConfigLoader(path).load_all()
        # bundle.software_params loaded successfully → no validation error.
        assert bundle.input_params["static"]["structure"] == "bcc-uranium"

    def test_list_of_strings_accepted(self, tmp_path: Path) -> None:
        """Multi-structure form: ``["si", "ge"]``."""
        path = _write_input(
            tmp_path,
            _minimal_input(static={"structure": ["si", "ge"], "metadata": "yeesuan"}),
        )
        bundle = ConfigLoader(path).load_all()
        assert bundle.input_params["static"]["structure"] == ["si", "ge"]

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        path = _write_input(
            tmp_path,
            _minimal_input(static={"structure": [], "metadata": "yeesuan"}),
        )
        with pytest.raises(ValueError, match="non-empty"):
            ConfigLoader(path).load_all()

    def test_non_string_element_raises(self, tmp_path: Path) -> None:
        path = _write_input(
            tmp_path,
            _minimal_input(
                static={"structure": ["si", 42], "metadata": "yeesuan"}
            ),
        )
        with pytest.raises(TypeError, match="string or a list of strings"):
            ConfigLoader(path).load_all()

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        path = _write_input(
            tmp_path,
            _minimal_input(static={"structure": 42, "metadata": "yeesuan"}),
        )
        with pytest.raises(TypeError, match="string or a list of strings"):
            ConfigLoader(path).load_all()

    def test_missing_structure_raises(self, tmp_path: Path) -> None:
        path = _write_input(
            tmp_path,
            _minimal_input(static={"metadata": "yeesuan"}),
        )
        with pytest.raises(KeyError, match="static.structure"):
            ConfigLoader(path).load_all()


# ---------------------------------------------------------------------------
# _resolve_presets — nested unwrap keeps sibiling keys
# ---------------------------------------------------------------------------


class TestResolvePresetsUnwrap:
    """The nested layout must keep keys at the same level as the backend."""

    def test_nested_form_preserves_sibling_keys(self, tmp_path: Path) -> None:
        """``{abacus: {parameters: ...}, kpoints_distance: ..., pseudo_family: ...}``
        → after unwrap, ``pseudo_family`` and ``kpoints_distance`` survive."""
        path = _write_input(
            tmp_path,
            _minimal_input(
                parameters={
                    "convergence": "test",
                    "abacus": "test",  # uses parameters/abacus/abacus.yml
                }
            ),
        )
        bundle = ConfigLoader(path).load_all()
        abacus = bundle.software_params["abacus"][0]

        # The nested form is unwrapped: ``parameters`` (and other
        # backend-native keys) move up to the top level, while sibling
        # keys like ``pseudo_family`` are preserved.
        assert "parameters" in abacus
        assert abacus["pseudo_family"] == "sg15_sz"
        assert abacus["kpoints_distance"] == 0.1

    def test_flat_form_kept_verbatim(self, tmp_path: Path) -> None:
        """A preset without the backend key is passed through unchanged.

        We exercise this via :func:`ConfigLoader._resolve_presets`
        directly so the test stays independent of any backend YAML
        layout changes.
        """
        flat = {
            "parameters": {"input": {"ecutwfc": 100}},
            "pseudo_family": "my_family",
        }
        out = ConfigLoader._resolve_presets(
            "abacus", {"test": flat}, ["test"], source="<inline>"
        )
        assert out == [flat]
        # Make sure we didn't accidentally inject an ``abacus`` key.
        assert "abacus" not in out[0]


# ---------------------------------------------------------------------------
# _coerce_preset_names — dict sub-category rejection path
# ---------------------------------------------------------------------------


class TestCoercePresetNames:
    """The helper used by the dict sub-category path rejects bad inputs."""

    def test_dict_subcategory_with_non_string_list_raises(self) -> None:
        """``{"abacus": {"magmom": [1, 2]}}`` must raise a TypeError."""
        with pytest.raises(TypeError, match="must be a string or list of strings"):
            ConfigLoader._coerce_preset_names("abacus", "magmom", [1, 2])

    def test_dict_subcategory_with_string_list_passes(self) -> None:
        out = ConfigLoader._coerce_preset_names(
            "abacus", "magmom", ["lcao", "pw"]
        )
        assert out == ["lcao", "pw"]

    def test_single_string_passes(self) -> None:
        out = ConfigLoader._coerce_preset_names("abacus", "abacus", "test")
        assert out == ["test"]


# ---------------------------------------------------------------------------
# _load_all_software_params — workflow-protocol slots are skipped
# ---------------------------------------------------------------------------


class TestWorkflowSlotSkipped:
    """Top-level ``parameters[<workflow>]`` is handled by ``_load_protocol``."""

    def test_workflow_protocol_slot_not_loaded_as_backend(
        self, tmp_path: Path
    ) -> None:
        """Only ``abacus`` / ``vasp`` slots populate ``software_params``."""
        path = _write_input(
            tmp_path,
            _minimal_input(
                parameters={
                    "convergence": "test",  # workflow slot — should be skipped
                    "abacus": "test",
                }
            ),
        )
        bundle = ConfigLoader(path).load_all()
        # ``convergence`` is not a backend: it must not appear in
        # ``software_params`` even though the input.json has a
        # ``parameters.convergence`` key.
        assert "convergence" not in bundle.software_params
        assert "abacus" in bundle.software_params
        # And the protocol section was loaded normally.
        assert "ecutwfc_list" in bundle.protocol["abacus"]
