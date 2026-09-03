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
        assert set(SUPPORTED_METHODS) == {
            "base", "smear", "convergence", "magmom", "banddos", "relax",
            "elastic", "phonopy", "eos", "defects", "supercell",
        }

    @pytest.mark.parametrize(
        "method",
        ["base", "smear", "convergence", "magmom", "banddos", "relax",
         "elastic", "phonopy", "eos", "defects", "supercell"],
    )
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

    @pytest.mark.parametrize(
        "method",
        [m for m in METHOD_SPECS
         if m not in ("banddos", "relax", "elastic", "phonopy", "eos", "defects",
                      "supercell")],
    )
    def test_backend_to_key_covers_abacus_and_vasp(self, method):
        spec = METHOD_SPECS[method]
        assert "abacus" in spec.backend_to_key
        assert "vasp" in spec.backend_to_key

    def test_banddos_only_has_abacus(self):
        """banddos is ABACUS-only — the FLEUR / VASP paths use separate
        WorkChain entry-points (``fleur.banddos`` lives in
        ``workflows/banddos/fleur.py`` as a wrapper around the plugin
        ``FleurBandAndDosWorkChain``). The banddos method spec therefore
        lists only ``"abacus"`` in its ``backend_to_key`` mapping."""
        spec = METHOD_SPECS["banddos"]
        assert "abacus" in spec.backend_to_key
        assert "vasp" not in spec.backend_to_key

    def test_relax_has_abacus_and_fleur(self):
        """relax covers abacus + fleur (no VASP)."""
        spec = METHOD_SPECS["relax"]
        assert spec.backend_to_key == {"abacus": "scf", "fleur": "scf"}
        assert "vasp" not in spec.backend_to_key

    def test_phonopy_only_has_abacus(self):
        """phonopy covers abacus + fleur (no VASP)."""
        spec = METHOD_SPECS["phonopy"]
        assert spec.backend_to_key == {"abacus": "scf", "fleur": "scf"}
        assert "vasp" not in spec.backend_to_key

    def test_eos_has_abacus_and_fleur(self):
        """eos covers abacus + fleur (no VASP)."""
        spec = METHOD_SPECS["eos"]
        assert spec.backend_to_key == {"abacus": "scf", "fleur": "scf"}
        assert "vasp" not in spec.backend_to_key

    def test_defects_has_abacus_and_fleur(self):
        """defects covers abacus + fleur (no VASP)."""
        spec = METHOD_SPECS["defects"]
        assert spec.backend_to_key == {"abacus": "scf", "fleur": "scf"}
        assert "vasp" not in spec.backend_to_key


class TestShortIdPassthrough:
    """``_short_id`` is exposed and handles pks + UUIDs."""

    def test_int(self):
        assert _short_id(42) == "42"

    def test_uuid_prefix(self):
        uuid = "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f"
        assert _short_id(uuid) == uuid[:8] == "8c0fe1a9"


class TestGenerateOneReportSkips:
    """``generate_one_report`` must skip unfinished / failed WorkChains
    instead of crashing on them (regression: the ``process_class`` read
    used to run before the ``not_finished`` tuple was unpacked, raising
    ``AttributeError: 'tuple' object has no attribute 'process_class'``)."""

    @staticmethod
    def _call(monkeypatch, tmp_path, fake_result, fake_status):
        from aiida_uranium_workflow.cli import _common

        monkeypatch.setattr(
            _common,
            "load_finished_workchain",
            lambda node_identifier, profile: (fake_result, fake_status),
        )
        return _common.generate_one_report(
            node_identifier=123,
            output_path=tmp_path / "report.md",
            profile=None,
            class_to_backend=_common.EOS_CLASS_TO_BACKEND,
            generate_report=lambda *args, **kwargs: "# report",
        )

    def test_skips_not_finished(self, monkeypatch, tmp_path):
        """A still-running node yields a skip (not a crash)."""
        status = self._call(monkeypatch, tmp_path, (None, "running"), "not_finished")
        assert status.startswith("skipped:")
        assert "not finished" in status
        assert not (tmp_path / "report.md").exists()

    def test_skips_failed_exit_status(self, monkeypatch, tmp_path):
        """A finished WorkChain with a non-zero exit status (e.g. an EOS
        run whose SCF child failed) is ignored — no report is written."""

        class FakeWorkchain:
            process_class = type("FleurEosWorkChain", (), {"__name__": "FleurEosWorkChain"})
            is_finished_ok = False
            exit_status = 300

        status = self._call(monkeypatch, tmp_path, FakeWorkchain(), "ok")
        assert status.startswith("skipped:")
        assert "exit_status=300" in status
        assert not (tmp_path / "report.md").exists()

    def test_ok_writes_report(self, monkeypatch, tmp_path):
        """A finished, successful WorkChain still produces a report."""
        from types import SimpleNamespace

        class FakeOutputs:
            def __init__(self, para):
                self.output_parameters = para

        class FakeWorkchain:
            process_class = type("FleurEosWorkChain", (), {"__name__": "FleurEosWorkChain"})
            is_finished_ok = True
            exit_status = 0
            outputs = FakeOutputs(
                SimpleNamespace(get_dict=lambda: {"volume_gs": 41.0})
            )

        status = self._call(monkeypatch, tmp_path, FakeWorkchain(), "ok")
        assert status.startswith("ok ->")
        assert (tmp_path / "report.md").read_text() == "# report"


# ---------------------------------------------------------------------------
# check / example / plot subcommands (improve.md Phase C)
# ---------------------------------------------------------------------------


def test_parser_has_new_commands():
    from aiida_uranium_workflow.cli._common import build_unified_parser

    parser = build_unified_parser()
    # argparse exposes subcommands as choices of the 'command' dest
    choices = {
        a.dest: a
        for a in parser._actions
        if getattr(a, "dest", None) == "command"
    }
    sub = next(iter(choices.values()))
    for name in ("check", "example", "plot"):
        assert name in sub.choices


def test_example_templates_cover_all_methods():
    from aiida_uranium_workflow.cli._common import SUPPORTED_METHODS
    from aiida_uranium_workflow.cli.main import _EXAMPLE_INPUTS

    for method in SUPPORTED_METHODS:
        assert method in _EXAMPLE_INPUTS, method
        t = _EXAMPLE_INPUTS[method]
        assert t["workflow"] == method
        assert "static" in t and "code" in t


class TestCheckOutput:
    """``check`` prints the resolved per-preset parameters that reach
    the WorkChain (SCF preset content, parsed protocol, scheduler
    options), not just a coarse ``?``-labelled summary."""

    QE_MAGMOM_INPUT = {
        "workflow": "magmom",
        "parameters": {
            "qe": {"magmom": ["test_u"]},
            "magmom": "test_u_afm_qe",
        },
        "static": {"structure": "gamma-uranium", "metadata": "yeesuan"},
        "profile": "aiida_profile",
        "code": {"qe": "qe@yeesuan"},
    }

    def _write(self, tmp_path, payload=None):
        import json

        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(payload or self.QE_MAGMOM_INPUT, indent=2)
        )
        return path

    def test_prints_resolved_params(self, tmp_path, capsys):
        """The SCF preset name, its parameters, the protocol workflow
        data and the scheduler options all appear in the output."""
        path = self._write(tmp_path)
        rc = main(["check", "-i", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        # Preset name + code, not the old '?@code:...' placeholder.
        assert "[qe/test_u]" in out
        assert "qe@yeesuan" in out
        # Resolved SCF parameters reaching the WorkChain.
        assert "ecutwfc" in out and "60" in out
        assert "kpoints_mesh" in out
        assert "pseudo_family" in out
        # Parsed protocol (magmom sweep) + scheduler options.
        assert "magmom_list" in out
        assert "queue_name" in out
        assert "configuration OK" in out

    def test_multi_preset_lists_all_names(self, tmp_path, capsys):
        """Multiple presets appear under their own header block."""
        payload = dict(self.QE_MAGMOM_INPUT)
        payload["parameters"] = {
            "abacus": ["test", "test_soc"],
            "magmom": "test",
        }
        payload["code"] = {"abacus": "abacus@yeesuan"}
        path = self._write(tmp_path, payload)
        rc = main(["check", "-i", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[abacus/test]" in out
        assert "[abacus/test_soc]" in out

    def test_invalid_input_returns_nonzero(self, tmp_path, capsys):
        path = tmp_path / "input.json"
        path.write_text('{"workflow": "magmom"}')
        rc = main(["check", "-i", str(path)])
        assert rc != 0
        assert "invalid input" in capsys.readouterr().err
