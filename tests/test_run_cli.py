"""Tests for ``aiida-uranium run`` — full pipeline with method resolution.

These tests mock at the ``cli.main.execute_workflow`` boundary, so they
don't need a real AiiDA profile / code / computer. The full method-
resolution path is exercised through the real
:func:`aiida_uranium_workflow.cli._common.resolve_method` and the real
:func:`utils.cal_json.write_cal_json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_uranium_workflow.cli.main import main
from aiida_uranium_workflow.schedulers import SubmittedJob


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_job(backend: str, preset: str, pk: int = 1) -> SubmittedJob:
    """Build a SubmittedJob with stable uuid so cal_json output is reproducible."""
    return SubmittedJob(
        backend=backend,
        preset_name=preset,
        pk=pk,
        structure_name="bcc-uranium",
        uuid=f"uuid-{backend}-{preset}-{pk}",
    )


@pytest.fixture
def fake_execute(monkeypatch):
    """Replace ``cli.main.execute_workflow`` with a recording fake.

    Returns a ``(calls, return_value_proxy)`` pair:

    * ``calls``  — list of ``dict`` (one entry per call) recording the
      keyword arguments the handler forwarded to ``execute_workflow``.
    * ``return_value_proxy`` — list-like the test mutates to control
      what the fake returns on the next call. Each handler call consumes
      ``return_value_proxy[0:0]`` (so multiple invocations behave as a
      FIFO queue, defaulting to ``[]`` if empty).
    """
    calls: list[dict] = []
    return_value: list[SubmittedJob] = []

    def _execute(*, input_json, profile, only):
        calls.append(
            {"input_json": input_json, "profile": profile, "only": only}
        )
        if return_value:
            return list(return_value)
        return []

    monkeypatch.setattr(
        "aiida_uranium_workflow.cli.main.execute_workflow", _execute
    )
    return calls, return_value


def _write_input(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "input.json"
    p.write_text(json.dumps(body))
    return p


# ---------------------------------------------------------------------------
# Method resolution (T-R1 ~ T-R5)
# ---------------------------------------------------------------------------


class TestRunMethodResolution:
    """How ``_run`` decides the workflow method."""

    def test_cli_method_wins_over_input_json(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R1: --method overrides input.json["workflow"]."""
        inp = _write_input(
            tmp_path, {"workflow": "convergence", "extra": "ignored"}
        )
        # fake_execute returns [] → handler exits 1. We only assert that
        # the *smear* log tag was emitted (proof the CLI took the
        # --method path, not the input.json path).
        rc = main(["run", "--method", "smear", "-i", str(inp)])

        assert rc == 1  # empty result branch
        err = capsys.readouterr().err
        assert "smear-run" in err
        # The handler still forwarded the call to execute_workflow with
        # the original input path.
        assert fake_execute[0][0]["input_json"] == str(inp)

    def test_input_json_workflow_used_when_method_omitted(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R2: no --method → use input.json['workflow']."""
        # `return_value` is empty → handler exits 1. We only care that
        # the handler got far enough to log with the correct tag.
        inp = _write_input(tmp_path, {"workflow": "convergence"})
        rc = main(["run", "-i", str(inp)])

        assert rc == 1  # empty return value path
        assert "convergence-run" in capsys.readouterr().err

    def test_output_json_workflow_fallback(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R3: when input.json has no workflow, output.json supplies it.

        In practice ``run`` writes output.json itself, so this codepath
        is only triggered if the same input.json is treated as both the
        input and (after a previous write) the output. We simulate by
        pointing ``-i`` at an output.json-shaped file with no input.json
        alongside.
        """
        out_like = _write_input(
            tmp_path,
            {
                "workflow": "magmom",
                "abacus": {"magmom": {"preset1": "uuid-1"}},
            },
        )
        rc = main(["run", "-i", str(out_like)])

        assert rc == 1
        assert "magmom-run" in capsys.readouterr().err

    def test_unresolvable_method_errors(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R4: no --method, no workflow field → ValueError, exit 1."""
        inp = _write_input(tmp_path, {"parameters": "anything"})
        rc = main(["run", "-i", str(inp)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "Cannot determine" in err
        # Handler must not have reached the execute_workflow call.
        assert fake_execute[0] == []

    def test_invalid_cli_method_errors(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R5: --method nonsense → argparse rejects before handler runs.

        ``parse_method`` is registered as the ``--method`` ``type=`` and
        fires inside ``argparse``, which raises ``SystemExit(2)``. This
        is the earliest line of defence — the handler is never reached.
        """
        inp = _write_input(tmp_path, {"workflow": "smear"})
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--method", "nonsense", "-i", str(inp)])

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "unknown method" in err
        # Handler never ran.
        assert fake_execute[0] == []


# ---------------------------------------------------------------------------
# Submission pipeline (T-R6 ~ T-R9)
# ---------------------------------------------------------------------------


class TestRunSubmission:
    """What ``_run`` does once the method is known."""

    def test_handler_forwards_args_to_execute_workflow(
        self, tmp_path: Path, fake_execute
    ) -> None:
        """T-R6: profile, only, input_json are passed through verbatim."""
        # Need a resolvable method so the handler reaches execute_workflow;
        # we also need a non-empty return so the call doesn't short-circuit.
        fake_execute[1].extend([_make_job("abacus", "smear_pbe")])
        inp = _write_input(tmp_path, {"workflow": "smear"})

        rc = main(
            [
                "run",
                "-i",
                str(inp),
                "-p",
                "my-profile",
                "--only",
                "abacus",
            ]
        )

        assert rc == 0
        assert len(fake_execute[0]) == 1
        call = fake_execute[0][0]
        assert call["input_json"] == str(inp)
        assert call["profile"] == "my-profile"
        assert call["only"] == "abacus"

    def test_output_json_written_with_workflow_field(
        self, tmp_path: Path, fake_execute
    ) -> None:
        """T-R7: successful run writes output.json with 'workflow' key."""
        fake_execute[1].extend(
            [
                _make_job("abacus", "smear_pbe", pk=1),
                _make_job("vasp", "smear_pbe", pk=2),
            ]
        )
        inp = _write_input(tmp_path, {"workflow": "smear"})

        rc = main(["run", "-i", str(inp)])

        assert rc == 0
        out_path = inp.parent / "output.json"
        assert out_path.is_file()
        data = json.loads(out_path.read_text())
        assert data["workflow"] == "smear"
        # Each backend's WorkChain UUID must be present.
        assert data["abacus"]["smear"]["smear_pbe"] == "uuid-abacus-smear_pbe-1"
        assert data["vasp"]["smear"]["smear_pbe"] == "uuid-vasp-smear_pbe-2"

    def test_no_workchain_submitted_exits_1(
        self, tmp_path: Path, fake_execute, capsys
    ) -> None:
        """T-R8: empty SubmittedJob list → exit 1, no output.json."""
        # fake_execute's return_value is empty by default.
        inp = _write_input(tmp_path, {"workflow": "convergence"})
        rc = main(["run", "-i", str(inp)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "no workchain submitted" in err
        # The handler short-circuits before write_cal_json.
        assert not (inp.parent / "output.json").exists()

    def test_only_filter_passed_through(
        self, tmp_path: Path, fake_execute
    ) -> None:
        """T-R9: --only abacus reaches execute_workflow."""
        fake_execute[1].extend([_make_job("abacus", "smear_pbe", pk=1)])
        inp = _write_input(tmp_path, {"workflow": "smear"})

        rc = main(["run", "-i", str(inp), "--only", "abacus"])

        assert rc == 0
        assert fake_execute[0][0]["only"] == "abacus"
