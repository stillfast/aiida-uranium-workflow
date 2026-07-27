"""Focused tests for direct base WorkChain orchestration."""

from __future__ import annotations

from aiida_uranium_workflow.schedulers.base_workchain import BaseWorkChainOrchestrator
from aiida_uranium_workflow.utils.common import ParamBundle


def test_base_orchestrator_submits_direct_plugin_entry_points(monkeypatch):
    calls = []
    entry_points = []

    class FakeNode:
        def __init__(self, pk):
            self.pk = pk
            self.uuid = f"uuid-{pk}"

    def fake_factory(entry_point):
        entry_points.append(entry_point)
        return type(entry_point.replace(".", "_"), (), {})

    def fake_submit(workchain_cls, **inputs):
        calls.append((workchain_cls, inputs))
        return FakeNode(len(calls))

    monkeypatch.setattr(
        "aiida_uranium_workflow.schedulers.base.load_profile", lambda *_args: None
    )
    monkeypatch.setattr(
        BaseWorkChainOrchestrator,
        "_build_structure",
        lambda self: ["structure"],
    )
    monkeypatch.setattr(
        "aiida.plugins.WorkflowFactory", fake_factory
    )
    monkeypatch.setattr("aiida.engine.submit", fake_submit)
    monkeypatch.setattr("aiida.orm.load_code", lambda label: f"code:{label}")
    monkeypatch.setattr("aiida.orm.Dict", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr("aiida.orm.Str", lambda value: value)
    monkeypatch.setattr("aiida.orm.Float", lambda value: value)

    bundle = ParamBundle(
        input_params={
            "workflow": "base",
            "profile": "aiida_profile",
            "parameters": {
                "abacus": {"abacus": "test"},
                "vasp": {"vasp": "test"},
            },
            "static": {"structure": "bcc-uranium"},
            "code": {"abacus": "abacus@test", "vasp": "vasp@test"},
        },
        protocol={},
        workflow_data={},
        software_params={
            "abacus": [
                {
                    "parameters": {"input": {"calculation": "scf"}},
                    "kpoints_distance": 0.1,
                    "pseudo_family": "sg15_sz",
                }
            ],
            "vasp": [
                {
                    "parameters": {"incar": {"encut": 300}},
                    "potential_family": "PBE",
                    "potential_mapping": {"U": "U"},
                    "kpoints_spacing": 0.0159,
                }
            ],
        },
        metadata={"options": {"resources": {"num_machines": 1}}},
    )

    jobs = BaseWorkChainOrchestrator(bundle).run_with_jobs()

    assert entry_points == ["abacus.base", "vasp.v2.vasp"]
    assert len(calls) == 2
    assert [job.preset_name for job in jobs] == ["test", "test"]
    assert "smear" not in calls[0][1] and "sigma" not in calls[0][1]
    assert "smear" not in calls[1][1] and "sigma" not in calls[1][1]
    assert calls[0][1]["abacus"]["metadata"]["options"] == bundle.metadata["options"]
    assert calls[1][1]["calc"]["metadata"]["options"] == bundle.metadata["options"]
