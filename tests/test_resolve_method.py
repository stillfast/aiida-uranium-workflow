"""Tests for ``cli._common.resolve_method`` — method resolution from
CLI flag, input.json, or output.json. No key-inference fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_uranium_workflow.cli._common import resolve_method


# ---------------------------------------------------------------------------
# CLI flag takes priority
# ---------------------------------------------------------------------------


def test_cli_method_wins(tmp_path: Path) -> None:
    """``--method`` on the command line overrides input.json / output.json."""
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"workflow": "smear"}))
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"workflow": "smear", "abacus": {"smear": {"x": "y"}}}))

    assert (
        resolve_method(cli_method="convergence", input_json=inp, output_json=out)
        == "convergence"
    )


def test_cli_method_invalid_raises(tmp_path: Path) -> None:
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"workflow": "smear"}))
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_method(cli_method="nonsense", input_json=inp)


# ---------------------------------------------------------------------------
# output.json (report / copy / archive path)
# ---------------------------------------------------------------------------


def test_reads_modern_output_json(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    out.write_text(
        json.dumps(
            {
                "workflow": "magmom",
                "abacus": {"magmom": {"preset1": "uuid-1"}},
                "vasp": {"magmom": {"preset1": "uuid-2"}},
            }
        )
    )
    assert resolve_method(cli_method=None, output_json=out) == "magmom"


def test_legacy_output_json_without_workflow_raises(tmp_path: Path) -> None:
    """Legacy output.json without ``"workflow"`` field must error out.

    There is no key-inference fallback: callers must either pass
    ``--method`` explicitly or upgrade their output.json.
    """
    out = tmp_path / "output.json"
    out.write_text(
        json.dumps(
            {
                "abacus": {"convergence": {"encut_400": "uuid-1"}},
                "vasp": {"convergence": {"encut_400": "uuid-2"}},
            }
        )
    )
    with pytest.raises(ValueError, match="Cannot determine"):
        resolve_method(cli_method=None, output_json=out)


def test_modern_output_json_with_unknown_workflow_raises(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"workflow": "nonsense", "abacus": {}}))
    with pytest.raises(ValueError, match="not a known method"):
        resolve_method(cli_method=None, output_json=out)


# ---------------------------------------------------------------------------
# input.json (run path)
# ---------------------------------------------------------------------------


def test_reads_input_json(tmp_path: Path) -> None:
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"workflow": "smear"}))
    assert resolve_method(cli_method=None, input_json=inp) == "smear"


def test_input_json_unknown_workflow_raises(tmp_path: Path) -> None:
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"workflow": "made-up"}))
    with pytest.raises(ValueError, match="not a known method"):
        resolve_method(cli_method=None, input_json=inp)


def test_modern_output_json_priority_over_input_json(tmp_path: Path) -> None:
    """When both modern output.json and input.json exist, the output.json
    ``"workflow"`` field wins (it's the most specific recent signal)."""
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"workflow": "smear"}))
    out = tmp_path / "output.json"
    out.write_text(
        json.dumps({"workflow": "convergence", "abacus": {"convergence": {"x": "y"}}})
    )
    assert resolve_method(
        cli_method=None, input_json=inp, output_json=out
    ) == "convergence"


# ---------------------------------------------------------------------------
# Defensive: missing / malformed files
# ---------------------------------------------------------------------------


def test_missing_input_json_falls_through_to_output_json(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"workflow": "smear", "abacus": {"smear": {}}}))
    assert resolve_method(
        cli_method=None, input_json=tmp_path / "missing.json", output_json=out
    ) == "smear"


def test_malformed_input_json_is_tolerated(tmp_path: Path) -> None:
    inp = tmp_path / "input.json"
    inp.write_text("{ this is not valid json")
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"workflow": "smear", "abacus": {"smear": {}}}))
    assert resolve_method(
        cli_method=None, input_json=inp, output_json=out
    ) == "smear"


def test_no_sources_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot determine"):
        resolve_method(
            cli_method=None,
            input_json=tmp_path / "missing.json",
            output_json=tmp_path / "also_missing.json",
        )
