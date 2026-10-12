"""Tests for the data-root resolution (``parameters/`` + ``static/``).

The preset tables are site- and project-specific, so the library resolves
them at run time instead of hard-wiring the installed package:

    ``--data-dir`` / ``data_root=``  →  ``$AIIDA_URANIUM_DATA``
      →  nearest ancestor of ``input.json`` carrying ``parameters/``
      →  packaged fallback snapshot

These tests pin that order, the loud failure modes (a half-populated root,
a missing explicit root), the "you edited the frozen copy" warning, and an
end-to-end :class:`ConfigLoader` run against a throwaway root — all of it
without an AiiDA profile (only YAML/JSON is read).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aiida_uranium_workflow.utils import common as common_module
from aiida_uranium_workflow.utils.common import (
    DATA_ROOT_ENV,
    DEFAULT_DATA_ROOT,
    DataRoot,
    find_data_root,
    resolve_data_root,
    snapshot_drift,
)
from aiida_uranium_workflow.utils.config import ConfigLoader

PROJECT_INPUT = {
    "workflow": "smear",
    "parameters": {"abacus": {"test": ["proj_preset"]}, "smear": "proj_protocol"},
    "static": {"structure": "bcc-uranium", "metadata": "sitemeta"},
    "profile": "aiida_profile",
    "code": {"abacus": "code@host"},
}

PROJECT_PARAMETERS_SMEAR = """
proj_protocol:
  abacus:
    smearing_method_list:
      - mp
    smearing_sigma_list:
      - 0.011
"""

PROJECT_ABACUS_TEST = """
proj_preset:
  abacus:
    parameters:
      input:
        calculation: scf
        smearing_method: mp
        smearing_sigma: 0.011
  kpoints_distance: 0.2
  pseudo_family: project_family
"""

PROJECT_METADATA = """
sitemeta:
  options:
    queue_name: project_queue
    max_wallclock_seconds: 60
"""


def _make_root(base: Path, *, parameters: bool = True, static: bool = True) -> Path:
    """Create a (possibly half-populated) data-root skeleton under ``base``."""
    if parameters:
        (base / "parameters").mkdir(parents=True, exist_ok=True)
    if static:
        (base / "static").mkdir(parents=True, exist_ok=True)
    return base


def _write_project_root(base: Path) -> Path:
    """Write a complete, self-contained data root + input.json under ``base``."""
    (base / "parameters" / "abacus").mkdir(parents=True, exist_ok=True)
    (base / "static").mkdir(parents=True, exist_ok=True)
    (base / "parameters" / "smear.yml").write_text(PROJECT_PARAMETERS_SMEAR)
    (base / "parameters" / "abacus" / "test.yml").write_text(PROJECT_ABACUS_TEST)
    (base / "static" / "metadata.yml").write_text(PROJECT_METADATA)
    (base / "static" / "structure.yml").write_text("bcc-uranium: {}\n")
    jobs = base / "jobs" / "smear"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "input.json").write_text(json.dumps(PROJECT_INPUT))
    return base


def _freeze_mtimes(root: Path, timestamp: float) -> None:
    """Set a fixed mtime on every file under ``root``."""
    for path in root.rglob("*"):
        os.utime(path, (timestamp, timestamp))


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


class TestResolveDataRoot:
    def test_explicit_override_wins(self, tmp_path, monkeypatch):
        explicit = _make_root(tmp_path / "explicit")
        env = _make_root(tmp_path / "env")
        monkeypatch.setenv(DATA_ROOT_ENV, str(env))
        root = resolve_data_root(explicit)
        assert root.path == explicit.resolve()
        assert root.source == "cli"

    def test_env_var_used_when_no_override(self, tmp_path, monkeypatch):
        env = _make_root(tmp_path / "env")
        monkeypatch.setenv(DATA_ROOT_ENV, str(env))
        root = resolve_data_root()
        assert root.path == env.resolve()
        assert root.source == f"env:{DATA_ROOT_ENV}"

    def test_nearest_ancestor_of_input_json_wins(self, tmp_path):
        _write_project_root(tmp_path)
        input_json = tmp_path / "jobs" / "smear" / "input.json"
        root = resolve_data_root(input_json=input_json)
        assert root.path == tmp_path.resolve()
        assert root.source.startswith("input-json:")

    def test_nested_root_beats_outer_root(self, tmp_path):
        _write_project_root(tmp_path)
        nested = _make_root(tmp_path / "jobs" / "smear")
        root = resolve_data_root(input_json=tmp_path / "jobs" / "smear" / "input.json")
        assert root.path == nested.resolve()

    def test_falls_back_to_packaged_snapshot_with_warning(self, tmp_path):
        input_json = tmp_path / "lonely" / "input.json"
        input_json.parent.mkdir(parents=True)
        input_json.write_text("{}")
        with pytest.warns(UserWarning, match="no data root next to"):
            root = resolve_data_root(input_json=input_json)
        assert root.path == DEFAULT_DATA_ROOT
        assert root.source == "package-default"

    def test_missing_explicit_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            resolve_data_root(tmp_path / "nope")

    def test_missing_env_root_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError, match=DATA_ROOT_ENV):
            resolve_data_root()


class TestHalfPopulatedRoot:
    """A root must carry both halves — no silent completion from the package."""

    def test_root_without_static_is_refused(self, tmp_path):
        root = _make_root(tmp_path / "params-only", static=False)
        with pytest.raises(ValueError, match="static/"):
            resolve_data_root(root)

    def test_root_without_parameters_is_skipped_by_the_search(self, tmp_path):
        """``parameters/`` is the marker — a directory without it is not a root."""
        (tmp_path / "static").mkdir()
        input_json = tmp_path / "input.json"
        input_json.write_text("{}")
        assert find_data_root(input_json) is None


class TestDataRootHelpers:
    def test_directories_and_missing_parts(self, tmp_path):
        root = DataRoot(_make_root(tmp_path / "ok"), "cli")
        assert root.parameters_dir == tmp_path / "ok" / "parameters"
        assert root.protocol_dir == root.parameters_dir
        assert root.static_dir == tmp_path / "ok" / "static"
        assert root.missing_parts() == []

        partial = DataRoot(_make_root(tmp_path / "partial", static=False), "cli")
        assert partial.missing_parts() == ["static"]

    def test_describe_names_the_source(self, tmp_path):
        assert "source: cli" in DataRoot(tmp_path, "cli").describe()


# ---------------------------------------------------------------------------
# Frozen-snapshot drift
# ---------------------------------------------------------------------------


class TestSnapshotDrift:
    SHARED = "parameters/abacus/test.yml"

    def test_identical_trees_have_no_drift(self, tmp_path):
        root = _write_project_root(tmp_path / "scripts")
        assert snapshot_drift(root, packaged=root)["newer_in_packaged"] == []

    def test_newer_packaged_file_is_reported(self, tmp_path):
        """Editing the frozen copy instead of the live root is the trap."""
        active = _write_project_root(tmp_path / "scripts")
        packaged = _write_project_root(tmp_path / "package")
        _freeze_mtimes(packaged, 1_000_000_000)
        _freeze_mtimes(active, 2_000_000_000)

        # Healthy: the live root is the one that was touched last.
        assert snapshot_drift(active, packaged=packaged)["newer_in_packaged"] == []

        os.utime(packaged / self.SHARED, (3_000_000_000, 3_000_000_000))
        assert snapshot_drift(active, packaged=packaged)["newer_in_packaged"] == [
            self.SHARED
        ]

    def test_files_only_on_one_side_are_reported_separately(self, tmp_path):
        active = _write_project_root(tmp_path / "scripts")
        packaged = _make_root(tmp_path / "package")
        (packaged / "parameters" / "extra.yml").write_text("x: 1\n")

        drift = snapshot_drift(active, packaged=packaged)
        assert drift["only_in_packaged"] == ["parameters/extra.yml"]
        assert "parameters/smear.yml" in drift["only_in_root"]
        assert drift["newer_in_packaged"] == []

    def test_no_warning_even_when_the_packaged_copy_is_newer(
        self, tmp_path, monkeypatch, recwarn
    ):
        """The run path deliberately does *not* warn about snapshot drift.

        The packaged copy is expected to move ahead of a project data root
        (git checkout / merge refresh its mtimes, and a tree such as
        ``aiida-uranium-scripts-v14`` is a deliberate fork of it), so the
        warning fired on every run and was pure noise.  The information is
        still available on request through :func:`snapshot_drift`.
        """
        import shutil

        active = _write_project_root(tmp_path / "scripts")
        packaged = tmp_path / "package"
        shutil.copytree(active, packaged)
        _freeze_mtimes(packaged, 1_000_000_000)
        _freeze_mtimes(active, 2_000_000_000)
        os.utime(packaged / self.SHARED, (3_000_000_000, 3_000_000_000))
        monkeypatch.setattr(common_module, "DEFAULT_DATA_ROOT", packaged)

        root = resolve_data_root(active)
        assert root.path == active.resolve()
        assert [str(w.message) for w in recwarn] == []  # no chatter
        # …while the drift itself is still detectable when asked for.
        assert snapshot_drift(active, packaged=packaged)["newer_in_packaged"] == [
            self.SHARED
        ]

    def test_no_warning_when_the_active_root_is_the_newest(
        self, tmp_path, monkeypatch, recwarn
    ):
        import shutil

        active = _write_project_root(tmp_path / "scripts")
        packaged = tmp_path / "package"
        shutil.copytree(active, packaged)
        _freeze_mtimes(packaged, 1_000_000_000)
        _freeze_mtimes(active, 2_000_000_000)
        monkeypatch.setattr(common_module, "DEFAULT_DATA_ROOT", packaged)

        resolve_data_root(active)
        assert [str(w.message) for w in recwarn] == []


# ---------------------------------------------------------------------------
# End to end: ConfigLoader reads the throwaway root
# ---------------------------------------------------------------------------


class TestConfigLoaderUsesTheResolvedRoot:
    def test_presets_protocol_and_metadata_come_from_the_project_root(self, tmp_path):
        root = _write_project_root(tmp_path)
        bundle = ConfigLoader(root / "jobs" / "smear" / "input.json").load_all()

        assert bundle.data_root is not None
        assert bundle.data_root.path == root.resolve()

        # The SCF preset, the protocol and the site metadata are the
        # project's — not the packaged snapshot's.
        assert bundle.software_params["abacus"][0]["pseudo_family"] == "project_family"
        assert bundle.software_preset_names["abacus"] == ["proj_preset"]
        assert bundle.workflow_data["smear_lists"]["abacus"]["smearing_sigma_list"] == [
            0.011
        ]
        assert bundle.metadata["options"]["queue_name"] == "project_queue"

    def test_explicit_data_dir_overrides_the_search(self, tmp_path):
        found = _write_project_root(tmp_path / "found")
        other = _write_project_root(tmp_path / "other")
        bundle = ConfigLoader(
            found / "jobs" / "smear" / "input.json", data_root=other
        ).load_all()
        assert bundle.data_root is not None
        assert bundle.data_root.path == other.resolve()

    def test_unknown_metadata_lists_the_available_entries(self, tmp_path):
        root = _write_project_root(tmp_path)
        input_json = root / "jobs" / "smear" / "input.json"
        raw = json.loads(input_json.read_text())
        raw["static"]["metadata"] = "not-there"
        input_json.write_text(json.dumps(raw))
        with pytest.raises(KeyError, match="sitemeta"):
            ConfigLoader(input_json).load_all()
