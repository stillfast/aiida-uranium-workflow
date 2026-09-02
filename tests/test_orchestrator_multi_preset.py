"""Tests for ``WorkflowOrchestrator`` when a backend has multiple presets.

These tests mock out the actual submission + adapter plumbing so we can
verify that ``run()`` invokes ``submit()`` once per (backend, preset)
pair and constructs each child inputs from the correct preset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeAdapted:
    workchain_cls: type
    inputs: dict


class _RecordingAdapter:
    """Adapter stub that records which preset was passed in."""

    instances: List[Tuple[str, int]] = []

    def __init__(
        self, code_label, software_params, metadata, workflow_data, *, extra_codes=None
    ) -> None:
        # ``software_params`` is a single preset dict; remember which
        # ``backend`` and which ``preset_index`` this adapter instance
        # corresponds to by reading a marker injected by the orchestrator.
        self.code_label = code_label
        self.software_params = software_params
        self.metadata = metadata
        self.workflow_data = workflow_data
        self.extra_codes = dict(extra_codes or {})
        _RecordingAdapter.instances.append(
            (software_params.get("__backend__"), software_params.get("__idx__"))
        )

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def adapt(self, structure):
        return _FakeAdapted(workchain_cls=dict, inputs=self.software_params)


class _FakeSubmit:
    """Mock for ``aiida.engine.submit`` that returns integer pks."""

    counter: int = 0
    calls: List[Tuple[type, dict]] = []

    @classmethod
    def reset(cls) -> None:
        cls.counter = 0
        cls.calls = []

    def __call__(self, workchain_cls, **inputs):
        type(self).counter += 1
        type(self).calls.append((workchain_cls, inputs))
        pk = type(self).counter
        return type("FakeNode", (), {"pk": pk})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator_env(monkeypatch, aiida_profile_clean):
    """Wire up a fake adapter + fake submit so we can inspect behaviour.

    Returns a helper ``make_bundle`` that builds a :class:`ParamBundle`
    whose ``software_params`` map carries fake presets carrying a
    ``__backend__`` / ``__idx__`` marker for the adapter to record.
    """
    _RecordingAdapter.reset()
    _FakeSubmit.reset()

    # Stub ``load_profile`` (the orchestrator calls it unconditionally).
    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.load_profile",
        lambda *_a, **_kw: None,
    )

    # Stub ``build_structure`` so we don't need a real AiiDA structure
    # registry to construct one — return a trivial marker.
    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.build_structure",
        lambda *_a, **_kw: "fake-structure",
    )

    # ``_build_structure`` then wraps the result in ``orm.StructureData``;
    # stub that out as well so we don't hit the real backend.
    import aiida.orm as _aiida_orm

    monkeypatch.setattr(_aiida_orm, "StructureData", lambda **_kw: "fake-structure-data")

    # Stub submit (the orchestrator imports it lazily inside
    # ``_submit_one`` via ``from aiida.engine import submit``).
    submit = _FakeSubmit()
    submit.reset()
    import aiida.engine as _aiida_engine

    monkeypatch.setattr(_aiida_engine, "submit", submit)

    # Replace the registry's adapters with our recorder.
    from aiida_uranium_workflow.schedulers.base import (
        WorkflowOrchestrator,
    )
    from aiida_uranium_workflow.schedulers.smear import (
        SmearWorkflowOrchestrator,
    )

    original_adapters = SmearWorkflowOrchestrator.ADAPTERS
    SmearWorkflowOrchestrator.ADAPTERS = {
        "abacus": _RecordingAdapter,
        "vasp": _RecordingAdapter,
    }
    WorkflowOrchestrator.ADAPTERS = SmearWorkflowOrchestrator.ADAPTERS

    try:
        yield _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator
    finally:
        SmearWorkflowOrchestrator.ADAPTERS = original_adapters
        WorkflowOrchestrator.ADAPTERS = original_adapters
        _RecordingAdapter.reset()
        _FakeSubmit.reset()


def _make_bundle(presets_per_backend: dict):
    """Build a minimal ParamBundle for the orchestrator."""
    from aiida_uranium_workflow.utils.common import ParamBundle

    # Inject ``__backend__`` / ``__idx__`` markers so the recorder adapter
    # can recover them after being passed an unwrapped preset dict.
    flat: dict[str, list[dict]] = {}
    for backend, presets in presets_per_backend.items():
        flat[backend] = [
            {**preset, "__backend__": backend, "__idx__": idx}
            for idx, preset in enumerate(presets)
        ]

    return ParamBundle(
        input_params={
            "workflow": "smear",
            "profile": "aiida_profile",
            "code": {"abacus": "fake-abacus", "vasp": "fake-vasp"},
            "static": {"structure": "fake-structure"},
            "parameters": {
                backend: (
                    [f"preset{i}" for i in range(len(presets))]
                    if len(presets) > 1
                    else "preset0"
                )
                for backend, presets in presets_per_backend.items()
            },
        },
        protocol={},
        workflow_data={"smear_lists": {"smear": ["gauss"], "sigma": [0.015]}},
        software_params=flat,
        metadata={"options": {"resources": {"num_machines": 1}}},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multiple_abacus_presets_submit_one_workchain_per_preset(orchestrator_env):
    """``abacus: [a, b]`` + ``vasp: v`` → 3 submissions total."""
    _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

    bundle = _make_bundle(
        {
            "abacus": [{"label": "a"}, {"label": "b"}],
            "vasp": [{"label": "v"}],
        }
    )

    orchestrator = SmearWorkflowOrchestrator(bundle)
    pks = orchestrator.run()

    # 2 abacus + 1 vasp = 3 submissions.
    assert pks == [1, 2, 3]
    assert len(_FakeSubmit.calls) == 3

    # Each call used a single preset (not the full list).
    submitted_inputs = [inputs for _wc, inputs in _FakeSubmit.calls]
    assert submitted_inputs[0]["__backend__"] == "abacus"
    assert submitted_inputs[0]["__idx__"] == 0
    assert submitted_inputs[0]["label"] == "a"

    assert submitted_inputs[1]["__backend__"] == "abacus"
    assert submitted_inputs[1]["__idx__"] == 1
    assert submitted_inputs[1]["label"] == "b"

    assert submitted_inputs[2]["__backend__"] == "vasp"
    assert submitted_inputs[2]["__idx__"] == 0
    assert submitted_inputs[2]["label"] == "v"

    # The recorder saw three adapter instances, one per preset.
    assert _RecordingAdapter.instances == [
        ("abacus", 0),
        ("abacus", 1),
        ("vasp", 0),
    ]


def test_single_preset_backend_submits_once(orchestrator_env):
    """Regression guard: string form still produces one submission per backend."""
    _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

    bundle = _make_bundle(
        {
            "abacus": [{"label": "a"}],
            "vasp": [{"label": "v"}],
        }
    )

    pks = SmearWorkflowOrchestrator(bundle).run()

    assert len(pks) == 2
    assert _RecordingAdapter.instances == [("abacus", 0), ("vasp", 0)]


def test_only_flag_filters_backends(orchestrator_env):
    """``--only abacus`` still respects multi-preset abacus list."""
    _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

    bundle = _make_bundle(
        {
            "abacus": [{"label": "a"}, {"label": "b"}],
            "vasp": [{"label": "v"}],
        }
    )

    orchestrator = SmearWorkflowOrchestrator(bundle)
    orchestrator.backends = ("abacus",)
    pks = orchestrator.run()

    # vasp skipped entirely.
    assert len(pks) == 2
    assert _RecordingAdapter.instances == [("abacus", 0), ("abacus", 1)]