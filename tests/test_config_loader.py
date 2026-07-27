"""Tests for :class:`ConfigLoader` against ``parameters/convergence.yml``.

The shared fixtures (input.json, ``ConfigLoader`` instance, expected
YAML snapshot) live in ``conftest.py`` so they can be reused by other
tests if needed.
"""

from __future__ import annotations

from aiida_uranium_workflow.utils.common import ParamBundle
from aiida_uranium_workflow.utils.config import ConfigLoader
from pathlib import Path

import json
import pytest

# ---------------------------------------------------------------------------
# __init__ / input.json round-trip
# ---------------------------------------------------------------------------


class TestBaseConfigLoader:
    def test_exact_example_base_json_loads_without_protocol(self) -> None:
        example = (
            Path(__file__).parents[1]
            / "src"
            / "aiida_uranium_workflow"
            / "example"
            / "base.json"
        )
        bundle = ConfigLoader(example).load_all()

        assert bundle.input_params["workflow"] == "base"
        assert bundle.protocol == {}
        assert bundle.workflow_data == {}
        assert len(bundle.software_params["abacus"]) == 1
        assert len(bundle.software_params["vasp"]) == 1
        assert bundle.software_params["abacus"][0]["pseudo_family"] == "sg15_sz"
        assert bundle.software_params["vasp"][0]["potential_family"] == "PBE"

    """``ConfigLoader.__init__`` reads the JSON file."""

    def test_reads_input_json(self, convergence_input_json: Path) -> None:
        loader = ConfigLoader(convergence_input_json)
        assert loader.input_json_path == convergence_input_json
        assert loader.input_params["workflow"] == "convergence"
        # ``parameters`` carries both the protocol-name slot and the
        # backend-preset slots.
        assert loader.input_params["parameters"]["convergence"] == "test"
        assert loader.input_params["parameters"]["abacus"] == "test"
        assert loader.input_params["parameters"]["vasp"] == "test"
        # ``static`` carries the metadata reference that drives
        # ``_load_metadata``.
        assert loader.input_params["static"]["metadata"] == "yeesuan"

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "input.json"
        bad.write_text(json.dumps({"workflow": "convergence"}))
        with pytest.raises(KeyError, match="missing required key"):
            ConfigLoader(bad).load_all()


# ---------------------------------------------------------------------------
# Convergence protocol parsing — exercises ``_load_protocol`` on
# ``parameters/convergence.yml``.
# ---------------------------------------------------------------------------


class TestConvergenceProtocol:
    """``load_all`` populates ``bundle.protocol`` from convergence.yml."""

    def test_protocol_matches_yaml_snapshot(
        self,
        loaded_bundle: ParamBundle,
        expected_convergence_yaml: dict,
    ) -> None:
        expected = expected_convergence_yaml["test"]
        assert loaded_bundle.protocol == expected

    def test_protocol_preserves_abacus_block(self, loaded_bundle: ParamBundle) -> None:
        abacus = loaded_bundle.protocol["abacus"]
        assert "ecutwfc_list" in abacus
        assert isinstance(abacus["ecutwfc_list"], list)
        assert all(isinstance(v, (int, float)) for v in abacus["ecutwfc_list"])
        assert "kpoints_distance_list" in abacus
        assert isinstance(abacus["kpoints_distance_list"], list)
        assert all(isinstance(v, (int, float)) for v in abacus["kpoints_distance_list"])

    def test_protocol_preserves_vasp_block(self, loaded_bundle: ParamBundle) -> None:
        vasp = loaded_bundle.protocol["vasp"]
        assert "encut_list" in vasp
        assert isinstance(vasp["encut_list"], list)
        assert all(isinstance(v, (int, float)) for v in vasp["encut_list"])
        assert "kpoints_spacing_list" in vasp
        assert isinstance(vasp["kpoints_spacing_list"], list)
        assert all(isinstance(v, (int, float)) for v in vasp["kpoints_spacing_list"])

    def test_workflow_data_contains_convergence_lists(
        self, loaded_bundle: ParamBundle
    ) -> None:
        # ``convergence`` is registered with ``parse_convergence_protocol``, so
        # ``_parse_protocol`` should return convergence_lists.
        assert "convergence_lists" in loaded_bundle.workflow_data
        assert "abacus" in loaded_bundle.workflow_data["convergence_lists"]
        assert "vasp" in loaded_bundle.workflow_data["convergence_lists"]


# ---------------------------------------------------------------------------
# Convergence backend-preset parsing — exercises ``_load_all_software_params``
# when the preset happens to be the convergence shared-preset file.
# ---------------------------------------------------------------------------


class TestConvergenceBackendParams:
    """``_load_all_software_params`` only touches real backends.

    Under the current schema ``convergence`` is a *workflow-protocol*
    slot (``parameters["convergence"]``) — it names the section to load
    from ``parameters/convergence.yml``. The backend YAMLs live in
    their own subdirectories (``parameters/abacus/abacus.yml`` and
    ``parameters/vasp/vasp.yml``), so the loader must ignore the
    ``convergence`` key here and only populate ``abacus`` / ``vasp``.
    """

    def test_convergence_slot_is_not_a_backend(
        self, loaded_bundle: ParamBundle
    ) -> None:
        assert "convergence" not in loaded_bundle.software_params

    def test_abacus_and_vasp_backends_loaded_from_their_own_yaml(
        self, loaded_bundle: ParamBundle
    ) -> None:
        # ``abacus.yml`` and ``vasp.yml`` carry their own ``test``
        # presets — these come from sibling files, not convergence.yml.
        assert "abacus" in loaded_bundle.software_params
        assert "vasp" in loaded_bundle.software_params

        abacus_list = loaded_bundle.software_params["abacus"]
        assert isinstance(abacus_list, list)
        assert len(abacus_list) == 1
        abacus = abacus_list[0]
        # Both backend presets use the nested layout (``abacus:`` /
        # ``vasp:`` keys); the loader unwraps them so only the
        # backend-native section survives.
        assert "parameters" in abacus
        assert abacus["parameters"]["input"]["ecutwfc"] == 100

        vasp_list = loaded_bundle.software_params["vasp"]
        assert isinstance(vasp_list, list)
        assert len(vasp_list) == 1
        vasp = vasp_list[0]
        assert "parameters" in vasp
        assert vasp["parameters"]["incar"]["encut"] == 300


# ---------------------------------------------------------------------------
# Metadata loading — ``_load_metadata``
# ---------------------------------------------------------------------------


class TestMetadata:
    """``_load_metadata`` reads ``static/metadata.yml``."""

    def test_metadata_section_populated(self, loaded_bundle: ParamBundle) -> None:
        md = loaded_bundle.metadata
        assert md["options"]["queue_name"] == "q_ysuan"
        assert md["options"]["resources"]["tot_num_mpiprocs"] == 56
        assert md["options"]["withmpi"] is True


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


class TestLoadAllEndToEnd:
    """A single end-to-end check that ``load_all`` returns a fully
    populated :class:`ParamBundle`."""

    def test_returns_param_bundle_with_all_sections(
        self, loaded_bundle: ParamBundle
    ) -> None:
        assert isinstance(loaded_bundle, ParamBundle)
        assert loaded_bundle.input_params["workflow"] == "convergence"
        # protocol
        assert "abacus" in loaded_bundle.protocol
        assert "vasp" in loaded_bundle.protocol
        # workflow_data
        assert "convergence_lists" in loaded_bundle.workflow_data
        # software_params
        assert set(loaded_bundle.software_params) >= {"abacus", "vasp"}
        # ``convergence`` is a workflow-protocol slot, not a backend
        # slot, so it must not appear under ``software_params``.
        assert "convergence" not in loaded_bundle.software_params
        # metadata
        assert "options" in loaded_bundle.metadata


# ---------------------------------------------------------------------------
# Negative-path tests — make sure the loader's error reporting still works.
# ---------------------------------------------------------------------------


class TestLoadProtocolErrors:
    """Missing protocol names should raise a descriptive ``KeyError``."""

    def test_unknown_protocol_name_raises(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.schedulers.base import _WORKFLOW_REGISTRY
        from aiida_uranium_workflow.utils.config import ConfigLoader

        # Make sure the convergence workflow is registered.
        assert "convergence" in _WORKFLOW_REGISTRY

        bad = tmp_path / "input.json"
        bad.write_text(
            json.dumps(
                {
                    "workflow": "convergence",
                    "parameters": {"convergence": "no-such-protocol"},
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {"abacus": "x", "vasp": "y"},
                }
            )
        )
        with pytest.raises(KeyError, match="no-such-protocol"):
            ConfigLoader(bad).load_all()


# ---------------------------------------------------------------------------
# Multiple presets per backend — ``"abacus": ["test", "test_soc"]``
# ---------------------------------------------------------------------------


class TestMultiplePresetsPerBackend:
    """A backend may be requested multiple times via a list of preset names."""

    def _write_input(self, tmp_path: Path, parameters: dict) -> Path:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "smear",
                    "parameters": parameters,
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {"abacus": "x", "vasp": "y"},
                }
            )
        )
        return path

    def test_backend_with_list_of_presets_loads_each_one(
        self, tmp_path: Path
    ) -> None:
        path = self._write_input(
            tmp_path,
            {"abacus": ["test", "test_soc"], "vasp": "test", "smear": "test"},
        )
        bundle = ConfigLoader(path).load_all()

        abacus_presets = bundle.software_params["abacus"]
        assert isinstance(abacus_presets, list)
        assert len(abacus_presets) == 2
        # Each preset is independently loaded (different dict objects).
        assert abacus_presets[0] is not abacus_presets[1]
        # ``test`` and ``test_soc`` carry different ``pseudo_family``
        # entries; the loader returns them as distinct presets.
        assert abacus_presets[0] != abacus_presets[1]

        # vasp still produces exactly one preset (string form).
        assert len(bundle.software_params["vasp"]) == 1

    def test_backend_with_unknown_preset_in_list_raises(
        self, tmp_path: Path
    ) -> None:
        path = self._write_input(
            tmp_path,
            {"abacus": ["test", "no-such-preset"], "smear": "test"},
        )
        with pytest.raises(KeyError, match="no-such-preset"):
            ConfigLoader(path).load_all()

    def test_backend_with_non_string_list_entry_raises(
        self, tmp_path: Path
    ) -> None:
        path = self._write_input(
            tmp_path,
            {"abacus": ["test", 123], "smear": "test"},
        )
        with pytest.raises(TypeError, match="must be a string or list of strings"):
            ConfigLoader(path).load_all()


# ---------------------------------------------------------------------------
# New ``parameters/<backend>/<backend>.yml`` directory layout.
# ---------------------------------------------------------------------------
# These tests verify that the on-disk layout matches what
# ``example/input.json`` advertises:
#
#   parameters/
#     abacus/abacus.yml     <- backend presets for abacus
#     vasp/vasp.yml         <- backend presets for vasp
#     smear.yml             <- smear workflow protocol
#     magmom.yml            <- magmom workflow protocol
#     convergence.yml       <- convergence workflow protocol


class TestBackendYAMLDirectoryLayout:
    """Backend YAMLs live in their own subdirectories, not at the top of
    ``parameters/``."""

    def test_backend_yaml_paths_exist(self) -> None:
        from aiida_uranium_workflow.utils.common import PARAMETERS_DIR

        for backend in ("abacus", "vasp"):
            yaml_path = PARAMETERS_DIR / backend / f"{backend}.yml"
            assert yaml_path.is_file(), (
                f"Expected backend preset file at {yaml_path}. "
                "The new schema requires parameters/<backend>/<backend>.yml."
            )

    def test_protocol_yamls_stay_at_parameters_root(self) -> None:
        """Workflow-protocol YAMLs (smear, magmom, convergence) are NOT
        moved into a subdirectory — they stay next to the backend
        directories."""
        from aiida_uranium_workflow.utils.common import PARAMETERS_DIR

        for protocol in ("smear", "magmom", "convergence"):
            yaml_path = PARAMETERS_DIR / f"{protocol}.yml"
            assert yaml_path.is_file(), (
                f"Expected protocol file at {yaml_path}. "
                "Protocol YAMLs stay at parameters/<name>.yml."
            )


class TestSmearWorkflowAgainstNewLayout:
    """End-to-end smoke test of the ``smear`` workflow using the new
    directory layout. Mirrors ``example/input.json``."""

    def _write_input(self, tmp_path: Path) -> Path:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "smear",
                    "parameters": {
                        "abacus": "test",
                        "vasp": "test",
                        "smear": "test",
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
            )
        )
        return path

    def test_smear_input_loads_abacus_and_vasp_presets(
        self, tmp_path: Path
    ) -> None:
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()

        # Backend presets come from the new subdirectories.
        assert set(bundle.software_params) == {"abacus", "vasp"}
        assert len(bundle.software_params["abacus"]) == 1
        assert len(bundle.software_params["vasp"]) == 1

        abacus = bundle.software_params["abacus"][0]
        # ``abacus.yml`` uses the nested layout; the loader unwraps the
        # ``abacus:`` key so only the backend-native section survives.
        assert "parameters" in abacus
        assert abacus["parameters"]["input"]["basis_type"] == "lcao"
        assert abacus["pseudo_family"] == "sg15_sz"

        vasp = bundle.software_params["vasp"][0]
        assert "parameters" in vasp
        assert vasp["parameters"]["incar"]["encut"] == 300

    def test_smear_protocol_loaded_separately(
        self, tmp_path: Path
    ) -> None:
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()
        # Protocol YAMLs live at parameters/<protocol>.yml (flat).
        assert "smear_list" in bundle.protocol
        assert "sigma_list" in bundle.protocol
        # Workflow-data hook (parse_smear_protocol) ran.
        assert "smear_lists" in bundle.workflow_data
        assert bundle.workflow_data["smear_lists"]["smear"]
        assert bundle.workflow_data["smear_lists"]["sigma"]

    def test_smear_with_multiple_abacus_presets(
        self, tmp_path: Path
    ) -> None:
        """``"abacus": ["test", "test_soc"]`` mirrors ``example/inputs.json``."""
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "smear",
                    "parameters": {
                        "abacus": ["test", "test_soc"],
                        "vasp": "test",
                        "smear": "test",
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
            )
        )
        bundle = ConfigLoader(path).load_all()
        abacus_presets = bundle.software_params["abacus"]
        assert len(abacus_presets) == 2
        # ``test`` uses sg15_sz, ``test_soc`` uses sg15_sz_soc.
        assert abacus_presets[0]["pseudo_family"] == "sg15_sz"
        assert abacus_presets[1]["pseudo_family"] == "sg15_sz_soc"


class TestMagmomWorkflowAgainstNewLayout:
    """End-to-end smoke test of the ``magmom`` workflow using the new
    directory layout.

    ``magmom.yml`` (workflow protocol) stays at the top of
    ``parameters/``; ``abacus.yml`` / ``vasp.yml`` (the base backend
    presets the magmom workflow also depends on) live in their own
    subdirectories.  The per-backend magmom preset table
    (``parameters/abacus/magmom.yml``) is loaded separately by the
    magmom workflow when needed — not by :class:`ConfigLoader`.
    """

    def _write_input(self, tmp_path: Path) -> Path:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "magmom",
                    "parameters": {
                        "abacus": "test",
                        "vasp": "test",
                        "magmom": "test",
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
            )
        )
        return path

    def test_magmom_input_loads_abacus_and_vasp_presets(
        self, tmp_path: Path
    ) -> None:
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()
        assert set(bundle.software_params) == {"abacus", "vasp"}
        assert bundle.software_params["abacus"][0]["pseudo_family"] == "sg15_sz"
        assert bundle.software_params["vasp"][0]["potential_family"] == "PBE"

    def test_magmom_protocol_loaded_separately(
        self, tmp_path: Path
    ) -> None:
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()
        assert "abacus" in bundle.protocol
        assert "vasp" in bundle.protocol
        assert "mag_list" in bundle.protocol["abacus"]
        assert "magmom_mapping_list" in bundle.protocol["vasp"]


# ---------------------------------------------------------------------------
# Sub-category backend slots: ``parameters[<backend>] = {"<category>": ...}``
# ---------------------------------------------------------------------------
# The magmom workflow's per-backend preset table lives in
# ``parameters/<backend>/magmom.yml``. To select it from input.json we
# use the dict form::
#
#     "parameters": {
#         "abacus": {"magmom": ["test_magmom_pw_pz", "test_magmom_pw_pbe"]},
#         "magmom": "test"
#     }
#
# The category key (here ``"magmom"``) names the YAML file under
# ``parameters/<backend>/`` to load. The dict form is the only way to
# reach those per-backend sub-tables without colliding with the
# workflow-protocol slot of the same name.


class TestSubCategoryBackendSlot:
    """Dict-form backend slots dispatch to ``parameters/<backend>/<category>.yml``."""

    def _write_input(self, tmp_path: Path) -> Path:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "magmom",
                    "parameters": {
                        "abacus": {
                            "magmom": [
                                "test_magmom_pw_pz",
                                "test_magmom_pw_pbe",
                            ]
                        },
                        "magmom": "test",
                    },
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {
                        "abacus": "abacus_lts@yeesuan",
                    },
                }
            )
        )
        return path

    def test_dict_form_routes_to_category_yaml(
        self, tmp_path: Path
    ) -> None:
        """``parameters.abacus.magmom`` reads from
        ``parameters/abacus/magmom.yml``, not ``abacus.yml``."""
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()

        abacus_presets = bundle.software_params["abacus"]
        assert len(abacus_presets) == 2

        # ``test_magmom_pw_pz`` uses qeU-pz, ``test_magmom_pw_pbe`` uses
        # qeU-pbe — both live only in parameters/abacus/magmom.yml.
        assert abacus_presets[0]["pseudo_family"] == "qeU-pz"
        assert abacus_presets[1]["pseudo_family"] == "qeU-pbe"

        # Both should be the unwrapped (backend-native) section after
        # the nested-key unwrap, not the raw ``abacus:`` block.
        for preset in abacus_presets:
            assert "parameters" in preset
            assert "kpoints_mesh" in preset

    def test_dict_form_protocol_still_loaded(
        self, tmp_path: Path
    ) -> None:
        """``parameters.magmom`` (top-level workflow-protocol slot) is
        unaffected by the dict-form backend slot."""
        bundle = ConfigLoader(self._write_input(tmp_path)).load_all()
        assert "mag_list" in bundle.protocol["abacus"]
        assert "magmom_mapping_list" in bundle.protocol["vasp"]

    def test_dict_form_uses_default_when_string_value(
        self, tmp_path: Path
    ) -> None:
        """A single string inside the dict still works."""
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "magmom",
                    "parameters": {
                        "abacus": {"magmom": "test_magmom"},
                        "magmom": "test",
                    },
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {"abacus": "x"},
                }
            )
        )
        bundle = ConfigLoader(path).load_all()
        assert len(bundle.software_params["abacus"]) == 1
        assert (
            bundle.software_params["abacus"][0]["pseudo_family"] == "sg15_sz"
        )

    def test_dict_form_unknown_preset_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "magmom",
                    "parameters": {
                        "abacus": {"magmom": ["no-such-preset"]},
                        "magmom": "test",
                    },
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {"abacus": "x"},
                }
            )
        )
        with pytest.raises(KeyError, match="no-such-preset"):
            ConfigLoader(path).load_all()

    def test_dict_form_non_string_value_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": "magmom",
                    "parameters": {
                        "abacus": {"magmom": 123},
                        "magmom": "test",
                    },
                    "static": {
                        "structure": "bcc-uranium",
                        "metadata": "yeesuan",
                    },
                    "profile": "aiida_profile",
                    "code": {"abacus": "x"},
                }
            )
        )
        with pytest.raises(TypeError, match="must be a string or list of strings"):
            ConfigLoader(path).load_all()
