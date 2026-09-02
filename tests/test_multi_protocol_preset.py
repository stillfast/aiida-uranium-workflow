"""Tests for multi-protocol-preset support.

``parameters[<workflow_key>]`` may be a **list** of protocol preset names
(e.g. ``"defects": ["vacancy_scf", "vacancy_relax"]``) — the
orchestrator then submits one WorkChain per requested preset, each with
its own parsed ``workflow_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import json
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@dataclass
class _FakeAdapted:
    workchain_cls: type
    inputs: dict


class _RecordingAdapter:
    """Adapter stub that records ``(scf_idx, workflow_preset)`` per submit."""

    instances: List[Tuple[Any, Any]] = []

    def __init__(
        self, code_label, software_params, metadata, workflow_data, *, extra_codes=None
    ) -> None:
        self.software_params = software_params
        self.workflow_data = workflow_data
        self.extra_codes = dict(extra_codes or {})
        _RecordingAdapter.instances.append(
            (
                software_params.get("__idx__"),
                workflow_data.get("__preset__"),
            )
        )

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def adapt(self, structure):
        return _FakeAdapted(workchain_cls=dict, inputs={})


class _FakeSubmit:
    counter: int = 0

    def __call__(self, workchain_cls, **inputs):
        type(self).counter += 1
        return type("FakeNode", (), {"pk": type(self).counter})()


@pytest.fixture
def multi_preset_env(monkeypatch, aiida_profile_clean):
    """Stub the AiiDA plumbing and record adapter construction."""
    _RecordingAdapter.reset()

    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.load_profile",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.build_structure",
        lambda *_a, **_kw: "fake-structure",
    )
    import aiida.orm as _aiida_orm

    monkeypatch.setattr(_aiida_orm, "StructureData", lambda **_kw: "fake-structure-data")

    submit = _FakeSubmit()
    import aiida.engine as _aiida_engine

    monkeypatch.setattr(_aiida_engine, "submit", submit)

    from aiida_uranium_workflow.schedulers.defects import (
        DefectsWorkflowOrchestrator,
    )

    original = DefectsWorkflowOrchestrator.ADAPTERS
    DefectsWorkflowOrchestrator.ADAPTERS = {"abacus": _RecordingAdapter}
    try:
        yield _RecordingAdapter, submit
    finally:
        DefectsWorkflowOrchestrator.ADAPTERS = original
        _RecordingAdapter.reset()


class TestConfigParsing:
    def test_protocol_list_populates_multi_preset(self, tmp_path):
        """``parameters['defects']`` as a list → per-preset workflow_data."""
        from aiida_uranium_workflow.utils.config import ConfigLoader

        inp = tmp_path / "input.json"
        inp.write_text(
            json.dumps(
                {
                    "workflow": "defects",
                    "parameters": {
                        "abacus": {"scf": ["lcao"]},
                        "defects": ["test", "test_relax"],
                    },
                    "static": {"structure": "bcc-uranium-0K", "metadata": "yunhe"},
                    "profile": "aiida_profile",
                    "code": {"abacus": "abacus@yunhe"},
                }
            )
        )
        bundle = ConfigLoader(inp).load_all()
        assert bundle.workflow_presets == ["test", "test_relax"]
        # Each preset has its own parsed workflow_data.
        assert bundle.workflow_data_map["test"]["abacus"]["defect"]["type"] == "vacancy"
        assert (
            bundle.workflow_data_map["test"]["abacus"]["wf_parameters"]["mode"]
            == "scf"
        )
        assert (
            bundle.workflow_data_map["test_relax"]["abacus"]["wf_parameters"]["mode"]
            == "relax"
        )
        # The single-preset default (backward compatible) is the first one.
        assert bundle.workflow_data["abacus"]["defect"]["type"] == "vacancy"

    def test_single_preset_keeps_empty_multi_fields(self, tmp_path):
        from aiida_uranium_workflow.utils.config import ConfigLoader

        inp = tmp_path / "input.json"
        inp.write_text(
            json.dumps(
                {
                    "workflow": "defects",
                    "parameters": {
                        "abacus": {"scf": ["lcao"]},
                        "defects": "test",
                    },
                    "static": {"structure": "bcc-uranium-0K", "metadata": "yunhe"},
                    "profile": "aiida_profile",
                    "code": {"abacus": "abacus@yunhe"},
                }
            )
        )
        bundle = ConfigLoader(inp).load_all()
        assert bundle.workflow_presets == []
        assert bundle.workflow_data_map == {}
        assert bundle.workflow_data["abacus"]["defect"]["type"] == "vacancy"


class TestOrchestratorMultiPreset:
    def _bundle(self, scf_presets, wf_presets, wf_data_map):
        from aiida_uranium_workflow.utils.common import ParamBundle

        return ParamBundle(
            input_params={
                "workflow": "defects",
                "parameters": {
                    "abacus": {"scf": scf_presets},
                    "defects": wf_presets,
                },
                "static": {"structure": "bcc-uranium-0K", "metadata": "yunhe"},
                "profile": "aiida_profile",
                "code": {"abacus": "abacus@yunhe"},
            },
            protocol={},
            workflow_data=wf_data_map[wf_presets[0]],
            software_params={
                "abacus": [
                    {"__idx__": i, **({"parameters": {"input": {}}} if False else {})}
                    for i in range(len(scf_presets))
                ]
            },
            metadata={},
            workflow_presets=wf_presets,
            workflow_data_map=wf_data_map,
        )

    def test_submits_per_scf_and_protocol_preset(self, multi_preset_env):
        """2 SCF presets × 2 protocol presets → 4 submissions, each with
        the right per-preset workflow_data."""
        from aiida_uranium_workflow.schedulers.defects import (
            DefectsWorkflowOrchestrator,
        )

        recording_adapter, submit = multi_preset_env
        wf_data_map = {
            "vacancy_scf": {"__preset__": "vacancy_scf"},
            "vacancy_relax": {"__preset__": "vacancy_relax"},
        }
        bundle = self._bundle(["lcao", "pw"], ["vacancy_scf", "vacancy_relax"], wf_data_map)
        orch = DefectsWorkflowOrchestrator(bundle)

        jobs = orch.run_with_jobs()
        assert len(jobs) == 4
        # Unique output.json keys combine the SCF preset and protocol preset.
        names = sorted(job.preset_name for job in jobs)
        assert names == sorted(
            [
                "lcao/vacancy_scf", "lcao/vacancy_relax",
                "pw/vacancy_scf", "pw/vacancy_relax",
            ]
        )
        # Each adapter saw the workflow_data of its protocol preset
        # (2 SCF presets × 2 protocol presets = 4 adapter constructions).
        seen = recording_adapter.instances
        from collections import Counter

        assert Counter(idx for idx, _ in seen) == {0: 2, 1: 2}
        assert Counter(preset for _, preset in seen) == {
            "vacancy_scf": 2,
            "vacancy_relax": 2,
        }

    def test_single_preset_behaviour_unchanged(self, multi_preset_env):
        """No workflow_presets → one submission per SCF preset (legacy)."""
        from aiida_uranium_workflow.schedulers.defects import (
            DefectsWorkflowOrchestrator,
        )

        recording_adapter, submit = multi_preset_env
        wf_data_map = {"test": {"__preset__": "test"}}
        bundle = self._bundle(["lcao"], ["test"], wf_data_map)
        bundle.workflow_presets = []
        bundle.workflow_data_map = {}
        orch = DefectsWorkflowOrchestrator(bundle)

        jobs = orch.run_with_jobs()
        assert len(jobs) == 1
        assert jobs[0].preset_name == "lcao"
