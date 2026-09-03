"""Tests for the report-side gather schema (``utils/report/schema.py``).

The schema is the *contract* between a workflow's gather calcfunction
and its report: whatever the gather step stores as
``output_parameters`` must round-trip through
:class:`GatherResult` (``to_dict`` → ``from_output_params``) without
losing a child's fields.  Legacy output dicts (pre-schema) must be
rejected (return ``None``) so the report falls back with a warning.
"""

from __future__ import annotations

from aiida_uranium_workflow.utils.report.schema import (
    SCHEMA_KEY,
    SCHEMA_VERSION,
    ChildRecord,
    GatherResult,
)

import pytest


class TestChildRecordRoundTrip:
    def test_to_from_dict_round_trip(self):
        record = ChildRecord(
            pk=101,
            status=0,
            finished_ok=True,
            energy_ev=-123.45,
            time_s=12.5,
            scf_steps=30,
            natoms=2,
            data={"final_magnetism": 0.0, "nspin": 2},
        )
        restored = ChildRecord.from_dict(record.to_dict())
        assert restored == record

    def test_defaults_survive(self):
        record = ChildRecord(pk=1)
        restored = ChildRecord.from_dict(record.to_dict())
        assert restored.status is None
        assert restored.finished_ok is False
        assert restored.data == {}


class TestGatherResultRoundTrip:
    def test_to_from_dict_round_trip(self):
        result = GatherResult(
            backend="abacus",
            children=[
                ChildRecord(
                    pk=1,
                    status=0,
                    finished_ok=True,
                    energy_ev=-1.0,
                    time_s=1.0,
                    scf_steps=5,
                    natoms=2,
                    data={"magnetism": [[0.0], [1.0]]},
                ),
                ChildRecord(pk=2, status=300, finished_ok=False),
            ],
            meta={"magmom_list": [[0.0, 0.0], [4.0, 4.0]]},
        )
        output = result.to_dict()
        assert output[SCHEMA_KEY] == SCHEMA_VERSION
        assert output["backend"] == "abacus"

        restored = GatherResult.from_output_params(output)
        assert restored is not None
        assert restored.backend == "abacus"
        assert restored.children == result.children
        assert restored.meta == result.meta


class TestLegacyDetection:
    def test_legacy_dict_returns_none(self):
        """A pre-schema gather dict has no schema marker → None."""
        legacy = {
            "magnetization": {1: 1.0},
            "final_energy": {1: -100.0},
            "status": {1: 0},
        }
        assert GatherResult.from_output_params(legacy) is None

    def test_wrong_version_returns_none(self):
        assert (
            GatherResult.from_output_params(
                {SCHEMA_KEY: 999, "backend": "x", "children": []}
            )
            is None
        )

    def test_non_dict_returns_none(self):
        assert GatherResult.from_output_params(None) is None
        assert GatherResult.from_output_params([1, 2]) is None

    def test_empty_children_valid(self):
        result = GatherResult.from_output_params(
            {SCHEMA_KEY: SCHEMA_VERSION, "backend": "qe", "children": []}
        )
        assert result is not None
        assert result.children == []


# ---------------------------------------------------------------------------
# Contract: schema dict → magmom report (new path renders, legacy warns)
# ---------------------------------------------------------------------------


def _abacus_gather_output() -> dict:
    """A realistic new-schema gather dict for an abacus magmom sweep."""
    return GatherResult(
        backend="abacus",
        children=[
            ChildRecord(
                pk=1,
                status=0,
                finished_ok=True,
                energy_ev=-123.45,
                time_s=12.5,
                scf_steps=30,
                natoms=2,
                data={"magnetism": [[0.0], [0.0]], "final_magnetism": 0.0},
            ),
            ChildRecord(
                pk=2,
                status=0,
                finished_ok=True,
                energy_ev=-123.5,
                time_s=14.0,
                scf_steps=32,
                natoms=2,
                data={"magnetism": [[4.0], [4.0]], "final_magnetism": 4.0},
            ),
        ],
    ).to_dict()


def _abacus_legacy_output() -> dict:
    """The equivalent legacy pk-keyed dict (pre-schema nodes)."""
    return {
        "magnetism": {1: [[0.0], [0.0]], 2: [[4.0], [4.0]]},
        "final_magnetism": {1: 0.0, 2: 4.0},
        "final_energy": {1: -123.45, 2: -123.5},
        "wall_time_seconds": {1: 12.5, 2: 14.0},
        "scf_steps": {1: 30, 2: 32},
        "status": {1: 0, 2: 0},
    }


class TestMagmomReportContract:
    def test_schema_path_renders_without_warning(self):
        from aiida_uranium_workflow.utils.report.magmom import generate_report

        with pytest.warns(None) as caught:
            report = generate_report(
                _abacus_gather_output(), pk=1, workflow_type="abacus"
            )
        # No legacy warning: schema path was taken.
        assert not any(
            "legacy" in str(w.message).lower() for w in caught.list
        )
        assert "## Magnetism Matrix" in report
        # The three legacy per-child sections are folded into Overview;
        # they must not be re-emitted as standalone sections.
        assert "## Calculation Status" not in report
        assert "## Final Energy" not in report
        assert "## Wall Time [s]" not in report

    def test_schema_and_legacy_render_same_body(self):
        """The same data must produce the same tables whether the node
        stored the new schema or the legacy layout (besides the warning
        on the legacy path)."""
        import warnings

        from aiida_uranium_workflow.utils.report.magmom import generate_report

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from_schema = generate_report(
                _abacus_gather_output(), pk=1, workflow_type="abacus"
            )
            from_legacy = generate_report(
                _abacus_legacy_output(), pk=1, workflow_type="abacus"
            )
        assert any("legacy" in str(w.message) for w in caught)
        assert from_schema == from_legacy
