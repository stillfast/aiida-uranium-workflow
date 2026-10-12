"""Tests for the per-scripts-tree ``parameters/`` + ``static/`` resolution.

A scripts tree may carry its own copies (``aiida-uranium-scripts-v14``
ships 14-valence-electron presets), and every command must pick those up
instead of the package's bundled files.

Resolution order (first hit wins):

1. ``set_config_root()`` — the CLI's ``--config-root``
2. ``$AIIDA_URANIUM_CONFIG_ROOT``
3. discovery: the nearest ancestor of the ``input.json`` / CWD that carries
   ``parameters/`` or ``static/``
4. the package itself

Lookups are **per file**: a file missing from the override root falls back
to the package copy, so a tree only carries what it changes.
"""

from __future__ import annotations

import json

import pytest

from aiida_uranium_workflow.utils.common import (
    CONFIG_ROOT_ENV,
    PKG_ROOT,
    config_path,
    config_root,
    config_root_label,
    find_config_root,
    set_config_root,
)


@pytest.fixture(autouse=True)
def _reset_root(monkeypatch):
    """Leave the module-level root exactly as it was found."""
    monkeypatch.delenv(CONFIG_ROOT_ENV, raising=False)
    set_config_root(None)
    yield
    set_config_root(None)


def _make_tree(tmp_path, *, parameters=True, static=False):
    """Build a scripts-tree layout: ``<root>/parameters/<backend>/test.yml``."""
    root = tmp_path / "aiida-uranium-scripts-v14"
    root.mkdir(parents=True, exist_ok=True)
    if parameters:
        (root / "parameters" / "abacus").mkdir(parents=True)
    if static:
        (root / "static").mkdir(parents=True)
    return root


class TestFindConfigRoot:
    def test_finds_tree_from_nested_input_dir(self, tmp_path):
        root = _make_tree(tmp_path)
        nested = root / "test" / "smear"
        nested.mkdir(parents=True)
        assert find_config_root(nested / "input.json") == root.resolve()

    def test_finds_tree_from_file_and_dir(self, tmp_path):
        root = _make_tree(tmp_path)
        input_json = root / "input.json"
        input_json.write_text("{}")
        assert find_config_root(input_json) == root.resolve()
        assert find_config_root(root) == root.resolve()

    def test_none_without_parameters_or_static(self, tmp_path):
        (tmp_path / "plain" / "deep").mkdir(parents=True)
        assert find_config_root(tmp_path / "plain" / "deep") is None

    def test_static_alone_is_enough(self, tmp_path):
        root = _make_tree(tmp_path, parameters=False, static=True)
        assert find_config_root(root / "test") == root.resolve()


class TestConfigPath:
    def test_package_default_when_nothing_set(self):
        assert config_root() is None
        assert config_path("parameters", "abacus", "scf.yml") == (
            PKG_ROOT / "parameters" / "abacus" / "scf.yml"
        )
        assert "package default" in config_root_label()

    def test_override_wins_for_existing_files(self, tmp_path):
        root = _make_tree(tmp_path)
        override = root / "parameters" / "abacus" / "test.yml"
        override.write_text("mine:\n  abacus: {}\n")
        set_config_root(root, source="test")

        assert config_path("parameters", "abacus", "test.yml") == override.resolve()
        assert config_root_label().endswith("(test)")

    def test_missing_file_falls_back_to_package(self, tmp_path):
        """Per-file overlay: only the changed files need to exist."""
        root = _make_tree(tmp_path)
        set_config_root(root, source="test")
        # Not present in the tree → package copy (it exists there).
        assert config_path("parameters", "abacus", "scf.yml") == (
            PKG_ROOT / "parameters" / "abacus" / "scf.yml"
        )
        # …and a path that exists nowhere still resolves under the package.
        assert config_path("parameters", "nope.yml") == PKG_ROOT / "parameters" / "nope.yml"

    def test_env_var_is_honoured(self, tmp_path, monkeypatch):
        root = _make_tree(tmp_path)
        monkeypatch.setenv(CONFIG_ROOT_ENV, str(root))
        set_config_root(None)  # re-read the environment
        assert config_root() == root.resolve()
        assert CONFIG_ROOT_ENV in config_root_label()

    def test_explicit_root_beats_env(self, tmp_path, monkeypatch):
        env_root = _make_tree(tmp_path / "env")
        explicit = _make_tree(tmp_path / "explicit")
        monkeypatch.setenv(CONFIG_ROOT_ENV, str(env_root))
        assert set_config_root(explicit, source="--config-root") == explicit.resolve()
        assert config_root() == explicit.resolve()


class TestLoaderUsesTheTree:
    """End-to-end: a preset that only exists in the tree must load."""

    def test_loader_reads_the_tree_preset(self, tmp_path):
        from aiida_uranium_workflow.utils.config import ConfigLoader

        root = _make_tree(tmp_path)
        (root / "parameters" / "abacus" / "test.yml").write_text(
            "only_in_tree:\n"
            "  abacus:\n"
            "    parameters:\n"
            "      input: {nspin: 1, basis_type: lw}\n"
            "  kpoints_mesh: [3, 3, 3]\n"
            "  pseudo_family: tree_family\n"
        )
        (root / "static").mkdir(exist_ok=True)
        (root / "static" / "structure.yml").write_text(
            "tree-cell:\n"
            "  spacegroup: 229\n"
            "  elements: ['U']\n"
            "  wickoff_position: ['a']\n"
            "  x: [3.5]\n"
        )
        input_json = root / "test" / "input.json"
        input_json.parent.mkdir()
        input_json.write_text(
            json.dumps(
                {
                    "workflow": "base",
                    "parameters": {"abacus": {"test": ["only_in_tree"]}},
                    "static": {"structure": "tree-cell", "metadata": "yeesuan"},
                    "profile": "aiida_profile",
                    "code": {"abacus": "abacus@yeesuan"},
                }
            )
        )

        set_config_root(find_config_root(input_json), source="discovered")
        bundle = ConfigLoader(input_json).load_all()

        preset = bundle.software_params["abacus"][0]
        assert preset["parameters"]["input"]["basis_type"] == "lw"
        assert preset["kpoints_mesh"] == [3, 3, 3]
        assert preset["pseudo_family"] == "tree_family"

    def test_structure_registry_can_be_overridden(self, tmp_path):
        from aiida_uranium_workflow.utils.structure import structure_yml

        root = _make_tree(tmp_path, static=True)
        (root / "static" / "structure.yml").write_text("x:\n  spacegroup: 229\n")
        set_config_root(root, source="test")
        assert structure_yml() == (root / "static" / "structure.yml").resolve()
