"""Tests for the list-of-structures feature.

Two layers are covered:

* :class:`ConfigLoader` rejects malformed ``static.structure`` values
  early (string vs non-empty list of strings only).
* :class:`WorkflowOrchestrator` submits one WorkChain per structure
  when ``static.structure`` is a list, regardless of how many backends
  / presets are also requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import json
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
    """Adapter stub that records which (backend, preset_idx, structure)
    combination was used for every submission."""

    instances: List[Tuple[str, int, Any]] = []

    def __init__(
        self, code_label, software_params, metadata, workflow_data, *, extra_codes=None
    ) -> None:
        self.code_label = code_label
        self.software_params = software_params
        self.metadata = metadata
        self.workflow_data = workflow_data
        self.extra_codes = dict(extra_codes or {})

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def adapt(self, structure):
        _RecordingAdapter.instances.append(
            (
                self.software_params.get("__backend__"),
                self.software_params.get("__idx__"),
                structure,
            )
        )
        return _FakeAdapted(
            workchain_cls=dict,
            inputs={**self.software_params, "__structure__": structure},
        )


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
    """Wire up a fake adapter + fake submit so multi-structure behaviour
    can be inspected without touching the real AiiDA backend.

    The fake ``build_structure`` returns a unique marker per name so we
    can verify which structure was used for each submission.
    """
    _RecordingAdapter.reset()
    _FakeSubmit.reset()

    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.load_profile",
        lambda *_a, **_kw: None,
    )

    # Each structure name → a unique marker object.
    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.build_structure",
        lambda name, **_kw: f"atoms::{name}",
    )

    import aiida.orm as _aiida_orm

    monkeypatch.setattr(
        _aiida_orm,
        "StructureData",
        lambda **_kw: _kw.get("ase"),
    )

    submit = _FakeSubmit()
    submit.reset()
    import aiida.engine as _aiida_engine

    monkeypatch.setattr(_aiida_engine, "submit", submit)

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


def _make_bundle(structures, presets_per_backend):
    """Build a minimal :class:`ParamBundle` for the orchestrator."""
    from aiida_uranium_workflow.utils.common import ParamBundle

    flat: dict[str, list[dict]] = {}
    for backend, presets in presets_per_backend.items():
        flat[backend] = [
            {
                **preset,
                "__backend__": backend,
                "__idx__": idx,
                # __stru_idx__ is filled in by the orchestrator per-structure
            }
            for idx, preset in enumerate(presets)
        ]

    static = (
        {"structure": structures}
        if isinstance(structures, (list, tuple))
        else {"structure": structures}
    )

    return ParamBundle(
        input_params={
            "workflow": "smear",
            "profile": "aiida_profile",
            "code": {"abacus": "fake-abacus", "vasp": "fake-vasp"},
            "static": static,
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


def _write_input(tmp_path: Path, *, structure, **kwargs) -> Path:
    """Write an input.json whose ``static.structure`` can be a string or
    a list of strings."""
    body = {
        "workflow": "smear",
        "parameters": {
            "abacus": "test",
            "vasp": "test",
            "smear": "test",
        },
        "static": {
            "structure": structure,
            "metadata": "yeesuan",
        },
        "profile": "aiida_profile",
        "code": {
            "abacus": "abacus@yeesuan",
            "vasp": "vaspstd@yeesuan",
        },
    }
    body.update(kwargs)
    p = tmp_path / "input.json"
    p.write_text(json.dumps(body, indent=2))
    return p


# ---------------------------------------------------------------------------
# ConfigLoader validation
# ---------------------------------------------------------------------------


class TestStaticStructureValidation:
    """``ConfigLoader._validate`` rejects malformed ``static.structure``."""

    def test_string_structure_passes(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        path = _write_input(tmp_path, structure="bcc-uranium")
        # Should not raise.
        bundle = ConfigLoader(path).load_all()
        assert bundle.input_params["static"]["structure"] == "bcc-uranium"

    def test_list_of_strings_passes(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        path = _write_input(
            tmp_path, structure=["bcc-uranium", "bcc-uranium-0K"]
        )
        bundle = ConfigLoader(path).load_all()
        assert bundle.input_params["static"]["structure"] == [
            "bcc-uranium",
            "bcc-uranium-0K",
        ]

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        path = _write_input(tmp_path, structure=[])
        with pytest.raises(ValueError, match="non-empty"):
            ConfigLoader(path).load_all()

    def test_list_with_non_string_raises(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        path = _write_input(tmp_path, structure=["bcc-uranium", 123])
        with pytest.raises(TypeError, match="string or a list of strings"):
            ConfigLoader(path).load_all()

    def test_non_string_non_list_raises(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        path = _write_input(tmp_path, structure=42)
        with pytest.raises(TypeError, match="string or a list of strings"):
            ConfigLoader(path).load_all()

    def test_missing_static_structure_key_raises(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.config import ConfigLoader

        # Build a body with no ``static.structure`` at all.
        body = {
            "workflow": "smear",
            "parameters": {"abacus": "test", "smear": "test"},
            "static": {"metadata": "yeesuan"},
            "profile": "aiida_profile",
            "code": {"abacus": "abacus@yeesuan", "vasp": "vaspstd@yeesuan"},
        }
        path = tmp_path / "input.json"
        path.write_text(json.dumps(body))
        with pytest.raises(KeyError, match="static.structure"):
            ConfigLoader(path).load_all()


# ---------------------------------------------------------------------------
# Orchestrator behaviour with a list of structures
# ---------------------------------------------------------------------------


class TestSingleStructureLegacy:
    """Regression guard: a single-string structure produces one submission
    per (backend, preset), exactly as before."""

    def test_single_structure_one_submission_per_backend(self, orchestrator_env):
        _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

        bundle = _make_bundle(
            "bcc-uranium",
            {"abacus": [{"label": "a"}], "vasp": [{"label": "v"}]},
        )

        pks = SmearWorkflowOrchestrator(bundle).run()

        assert len(pks) == 2
        # Each adapter call carries the same structure marker (the only
        # structure) for both backends.
        assert _RecordingAdapter.instances == [
            ("abacus", 0, "atoms::bcc-uranium"),
            ("vasp", 0, "atoms::bcc-uranium"),
        ]
        # The single structure object is forwarded to the adapter.
        submitted = [inputs for _wc, inputs in _FakeSubmit.calls]
        assert submitted[0]["__structure__"] == "atoms::bcc-uranium"
        assert submitted[1]["__structure__"] == "atoms::bcc-uranium"


class TestMultipleStructures:
    """The list-of-structures feature: ``input_stru.json`` style."""

    def test_two_structures_two_backends_four_submissions(self, orchestrator_env):
        """2 structures × 2 backends = 4 WorkChain submissions."""
        _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

        bundle = _make_bundle(
            ["bcc-uranium", "bcc-uranium-0K"],
            {"abacus": [{"label": "a"}], "vasp": [{"label": "v"}]},
        )

        pks = SmearWorkflowOrchestrator(bundle).run()

        assert pks == [1, 2, 3, 4]
        assert len(_FakeSubmit.calls) == 4

        # Each call should have used the matching structure marker.
        submitted = [inputs for _wc, inputs in _FakeSubmit.calls]
        assert submitted[0]["__backend__"] == "abacus"
        assert submitted[0]["__structure__"] == "atoms::bcc-uranium"

        assert submitted[1]["__backend__"] == "abacus"
        assert submitted[1]["__structure__"] == "atoms::bcc-uranium-0K"

        assert submitted[2]["__backend__"] == "vasp"
        assert submitted[2]["__structure__"] == "atoms::bcc-uranium"

        assert submitted[3]["__backend__"] == "vasp"
        assert submitted[3]["__structure__"] == "atoms::bcc-uranium-0K"

        # Adapter instances: every (backend, preset_idx, structure)
        # combination must be realised exactly once.
        assert _RecordingAdapter.instances == [
            ("abacus", 0, "atoms::bcc-uranium"),
            ("abacus", 0, "atoms::bcc-uranium-0K"),
            ("vasp", 0, "atoms::bcc-uranium"),
            ("vasp", 0, "atoms::bcc-uranium-0K"),
        ]

    def test_structures_combine_with_multiple_presets(self, orchestrator_env):
        """2 structures × 2 presets × 1 backend = 4 WorkChain submissions."""
        _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

        bundle = _make_bundle(
            ["bcc-uranium", "bcc-uranium-0K"],
            {"abacus": [{"label": "a"}, {"label": "b"}]},
        )

        pks = SmearWorkflowOrchestrator(bundle).run()

        assert pks == [1, 2, 3, 4]
        assert len(_FakeSubmit.calls) == 4

        submitted = [inputs for _wc, inputs in _FakeSubmit.calls]
        # Order: backend iterations, then preset iterations, then structures.
        assert (
            submitted[0]["__backend__"],
            submitted[0]["__idx__"],
            submitted[0]["__structure__"],
        ) == ("abacus", 0, "atoms::bcc-uranium")
        assert (
            submitted[1]["__backend__"],
            submitted[1]["__idx__"],
            submitted[1]["__structure__"],
        ) == ("abacus", 0, "atoms::bcc-uranium-0K")
        assert (
            submitted[2]["__backend__"],
            submitted[2]["__idx__"],
            submitted[2]["__structure__"],
        ) == ("abacus", 1, "atoms::bcc-uranium")
        assert (
            submitted[3]["__backend__"],
            submitted[3]["__idx__"],
            submitted[3]["__structure__"],
        ) == ("abacus", 1, "atoms::bcc-uranium-0K")

    def test_only_flag_filters_backends_with_multi_structure(self, orchestrator_env):
        """``--only abacus`` with 2 structures still produces 2 submissions."""
        _RecordingAdapter, _FakeSubmit, SmearWorkflowOrchestrator = orchestrator_env

        bundle = _make_bundle(
            ["bcc-uranium", "bcc-uranium-0K"],
            {"abacus": [{"label": "a"}], "vasp": [{"label": "v"}]},
        )

        orchestrator = SmearWorkflowOrchestrator(bundle)
        orchestrator.backends = ("abacus",)
        pks = orchestrator.run()

        # vasp is filtered out; abacus still runs once per structure.
        assert len(pks) == 2
        assert _RecordingAdapter.instances == [
            ("abacus", 0, "atoms::bcc-uranium"),
            ("abacus", 0, "atoms::bcc-uranium-0K"),
        ]


class TestBuildStructureValidation:
    """``WorkflowOrchestrator._build_structure`` rejects bad inputs even
    if the ConfigLoader validation was bypassed."""

    def test_non_string_list_entry_raises(self, monkeypatch, aiida_profile_clean):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
        orchestrator.bundle = type(
            "B", (), {"input_params": {"static": {"structure": ["ok", 5]}}}
        )()
        with pytest.raises(TypeError, match="string or a list of strings"):
            orchestrator._build_structure()

    def test_empty_list_raises(self, monkeypatch, aiida_profile_clean):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
        orchestrator.bundle = type(
            "B", (), {"input_params": {"static": {"structure": []}}}
        )()
        with pytest.raises(ValueError, match="non-empty"):
            orchestrator._build_structure()


class TestLabelFormatting:
    """``WorkflowOrchestrator._label`` keeps the existing labels intact
    and adds a ``@stru<N>`` suffix only when multiple structures are run."""

    def test_label_with_only_backend(self):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        assert WorkflowOrchestrator._label("abacus", 0, 1, 0, 1) == "abacus"

    def test_label_with_multiple_presets(self):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        assert (
            WorkflowOrchestrator._label("abacus", 1, 3, 0, 1) == "abacus#1"
        )

    def test_label_with_multiple_structures(self):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        assert (
            WorkflowOrchestrator._label("vasp", 0, 1, 1, 2) == "vasp@stru1"
        )

    def test_label_with_presets_and_structures(self):
        from aiida_uranium_workflow.schedulers.base import WorkflowOrchestrator

        assert (
            WorkflowOrchestrator._label("abacus", 2, 3, 1, 2) == "abacus#2@stru1"
        )