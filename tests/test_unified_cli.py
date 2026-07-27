"""Tests for the unified ``aiida-uranium {run,report,archive}`` CLI.

These tests focus on the argparse plumbing — they intentionally avoid
spinning up AiiDA / scheduler code paths, since those have their own
dedicated tests elsewhere in the suite.
"""

from __future__ import annotations

from aiida_uranium_workflow.cli._common import (
    METHOD_SPECS,
    SUPPORTED_METHODS,
    _short_id,
    build_unified_parser,
    collect_pk_map,
    default_result_path,
    read_unique_node_identifiers,
    read_unique_pks,
    get_method_spec,
    parse_method,
)
from aiida_uranium_workflow.cli.main import main

import json
import pytest
from pathlib import Path


class TestMethodRegistry:
    """``METHOD_SPECS`` exposes one entry per supported method."""

    def test_supported_methods(self):
        assert set(SUPPORTED_METHODS) == {"base", "smear", "convergence", "magmom"}

    @pytest.mark.parametrize("method", ["base", "smear", "convergence", "magmom"])
    def test_get_method_spec_returns_valid_entry(self, method):
        spec = get_method_spec(method)
        assert spec.name == method
        assert spec.generate_report is not None
        assert spec.class_to_backend  # non-empty

    def test_get_method_spec_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_method_spec("nope")

    def test_parse_method_rejects_unknown(self):
        with pytest.raises(Exception):
            parse_method("nope")

    def test_parse_method_accepts_known(self):
        assert parse_method("smear") == "smear"

    def test_run_parser_accepts_base(self):
        args = build_unified_parser().parse_args(
            ["run", "--method", "base", "-i", "base.json"]
        )
        assert args.command == "run"
        assert args.method == "base"
        assert args.input_json == "base.json"


class TestBuildUnifiedParser:
    """``build_unified_parser`` produces a parser with the unified subcommands."""

    def test_top_level_lists_subcommands(self):
        parser = build_unified_parser()
        # Parsing without a subcommand should fail (subparsers are required).
        with pytest.raises(SystemExit):
            parser.parse_args([])

    @pytest.mark.parametrize(
        "argv",
        [
            ["run", "--help"],
            ["report", "--help"],
            ["archive", "--help"],
            ["copy", "--help"],
        ],
    )
    def test_help_exits_zero_and_lists_method(self, argv, capsys):
        parser = build_unified_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(argv)
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--method" in out

    def test_top_level_help_exits_zero(self, capsys):
        parser = build_unified_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        # argparse lists subcommands alphabetically inside ``{...}``;
        # check each subcommand appears individually to stay
        # order-independent.
        for sub in ("run", "report", "archive", "copy"):
            assert sub in out


class TestMainHelp:
    """``main()`` propagates ``--help`` and shows the three subcommands."""

    def test_main_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "aiida-uranium" in out
        # All four subcommands must appear (argparse alphabetises them,
        # so check individually rather than the literal braces).
        for sub in ("run", "report", "archive", "copy"):
            assert sub in out

    def test_main_unknown_method_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--method", "bogus", "-i", "x.json"])
        assert exc_info.value.code != 0

    def test_main_missing_input_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--method", "smear"])
        assert exc_info.value.code != 0


class TestCollectPkMap:
    """``collect_pk_map`` parses output.json into the nested dict shape."""

    def test_round_trip(self, tmp_path):
        p = tmp_path / "output.json"
        p.write_text(
            json.dumps(
                {
                    "abacus": {"smear": {"lcao": 1, "pw": 2}},
                    "vasp": {"vasp": {"test": 3}},
                }
            )
        )
        data = collect_pk_map(p)
        assert data["abacus"]["smear"]["lcao"] == 1
        assert data["vasp"]["vasp"]["test"] == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            collect_pk_map(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(ValueError, match="Failed to parse"):
            collect_pk_map(p)

    def test_non_object_raises(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="expected a JSON object"):
            collect_pk_map(p)


class TestDefaultResultPath:
    """``default_result_path`` returns ``<input_dir>/output.json``."""

    def test_default_path_is_next_to_input(self, tmp_path):
        inp = tmp_path / "input.json"
        out = default_result_path(inp)
        assert out == (tmp_path / "output.json").resolve()


class TestMethodSpecBackendToKey:
    """``backend_to_key`` is wired up for every method in the registry."""

    @pytest.mark.parametrize("method", list(METHOD_SPECS))
    def test_backend_to_key_covers_abacus_and_vasp(self, method):
        spec = METHOD_SPECS[method]
        assert "abacus" in spec.backend_to_key
        assert "vasp" in spec.backend_to_key


class TestReadUniqueNodeIdentifiers:
    """``read_unique_node_identifiers`` handles both PK and UUID payloads."""

    def test_legacy_pk_payload(self, tmp_path, capsys):
        p = tmp_path / "output.json"
        p.write_text(json.dumps({"abacus": {"smear": {"lcao": 1, "pw": 2}}}))
        ids = read_unique_node_identifiers(p, source="smear-report")
        # The output JSON's integer pks are stringified in the result.
        assert ids == ["1", "2"]
        # No warnings printed for a successful read.
        captured = capsys.readouterr()
        assert "No node identifiers" not in captured.err

    def test_modern_uuid_payload(self, tmp_path):
        p = tmp_path / "output.json"
        uuids = {
            "pw": "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f",
            "pw_r": "33e15b7c-9f8a-4b6c-8123-0f9e8d7c6b5a",
        }
        p.write_text(json.dumps({"abacus": {"convergence": uuids}}))
        ids = read_unique_node_identifiers(p, source="convergence-report")
        assert ids == sorted(uuids.values())

    def test_mixed_payload_yields_everything_as_strings(self, tmp_path):
        p = tmp_path / "output.json"
        p.write_text(
            json.dumps(
                {
                    "abacus": {
                        "smear": {
                            "lcao": 1,
                            "pw": "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f",
                        }
                    }
                }
            )
        )
        ids = read_unique_node_identifiers(p, source="smear-report")
        assert "1" in ids
        assert "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f" in ids

    def test_missing_file_returns_empty(self, tmp_path, capsys):
        ids = read_unique_node_identifiers(
            tmp_path / "nope.json", source="smear-report"
        )
        assert ids == []
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        ids = read_unique_node_identifiers(p, source="smear-report")
        assert ids == []
        captured = capsys.readouterr()
        assert "Failed to parse" in captured.err


class TestReadUniquePksBackwardCompat:
    """``read_unique_pks`` still picks up integer pks in legacy payloads."""

    def test_pk_payload(self, tmp_path, capsys):
        p = tmp_path / "output.json"
        p.write_text(json.dumps({"abacus": {"smear": {"lcao": 1, "pw": 2}}}))
        pks = read_unique_pks(p, source="smear-archive")
        assert pks == [1, 2]

    def test_uuid_payload_returns_empty(self, tmp_path):
        p = tmp_path / "output.json"
        p.write_text(
            json.dumps(
                {"abacus": {"convergence": {"pw": "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f"}}}
            )
        )
        # Legacy helper doesn't understand UUIDs: returns empty.
        assert read_unique_pks(p, source="smear-archive") == []


class TestShortIdPassthrough:
    """``_short_id`` is exposed and handles pks + UUIDs."""

    def test_int(self):
        assert _short_id(42) == "42"

    def test_uuid_prefix(self):
        uuid = "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f"
        assert _short_id(uuid) == uuid[:8] == "8c0fe1a9"
