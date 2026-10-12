"""Focused tests for direct base WorkChain orchestration."""

from __future__ import annotations

import pytest

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


def test_base_adapter_forwards_setting_block(monkeypatch):
    """The preset's ``setting`` block (e.g. additional_retrieve_list for
    the SPIN1_CHG.cube output) must reach ``abacus.settings`` — it is
    what makes abacus.base retrieve the charge-density file."""
    from aiida_uranium_workflow.input_builders.base_workchain import (
        AbacusBaseWorkChainAdapter,
    )

    dict_payloads = []
    monkeypatch.setattr("aiida.orm.Dict", lambda d: ("Dict", dict_payloads.append(d)))
    monkeypatch.setattr("aiida.orm.Str", lambda v: v)
    monkeypatch.setattr("aiida.orm.load_code", lambda label: f"code:{label}")

    adapter = AbacusBaseWorkChainAdapter(
        code_label="abacus@test",
        software_params={
            "parameters": {"input": {"calculation": "scf", "out_chg": "1 10"}},
            "setting": {"additional_retrieve_list": ["SPIN1_CHG.cube"]},
            "kpoints_distance": 0.1,
            "pseudo_family": "sg15_sz",
        },
        metadata={},
        workflow_data={},
    )
    inputs = adapter._build_workchain_inputs("structure")

    settings_payload = next(
        (p for p in dict_payloads if "additional_retrieve_list" in p), None
    )
    assert settings_payload == {"additional_retrieve_list": ["SPIN1_CHG.cube"]}
    assert "settings" in inputs["abacus"]
    # the INPUT parameters still carry the high-precision out_chg setting
    params_payload = next(p for p in dict_payloads if "input" in p)
    assert params_payload["input"]["out_chg"] == "1 10"


def test_base_adapter_forwards_extra_files(aiida_profile_clean, tmp_path):
    """The preset's ``extra_files`` (SinglefileData pk/uuid) must land in
    ``abacus.extra_files`` with the basepath forwarded, so abacus.base
    copies e.g. SPIN1_CHG.cube into ``./chg/`` for ``init_chg file``."""
    import aiida.orm
    from aiida import orm
    from aiida_uranium_workflow.input_builders.base_workchain import (
        AbacusBaseWorkChainAdapter,
    )

    cube_file = tmp_path / "SPIN1_CHG.cube"
    cube_file.write_text("fake cube content")
    chg = orm.SinglefileData(file=cube_file)

    nodes = {379124: chg}
    monkey_load = lambda ident: nodes[int(ident)]  # noqa: E731
    import aiida.orm as orm_mod

    adapter = AbacusBaseWorkChainAdapter(
        code_label="abacus@test",
        software_params={
            "parameters": {"input": {"calculation": "scf", "init_chg": "file",
                                     "read_file_dir": "./chg/"}},
            "extra_files": {"chg": 379124},
            "extra_files_basepath": "chg",
            "kpoints_distance": 0.1,
            "pseudo_family": "sg15_sz",
        },
        metadata={},
        workflow_data={},
    )
    orig_load_node = orm_mod.load_node
    orig_load_code = orm_mod.load_code
    orm_mod.load_node = monkey_load
    orm_mod.load_code = lambda label: f"code:{label}"
    try:
        inputs = adapter._build_workchain_inputs("structure")
    finally:
        orm_mod.load_node = orig_load_node
        orm_mod.load_code = orig_load_code

    assert inputs["abacus"]["extra_files"]["chg"] is chg
    assert inputs["abacus"]["extra_files_basepath"].value == "chg"
    params = inputs["abacus"]["parameters"].get_dict()["input"]
    assert params["init_chg"] == "file"
    assert params["read_file_dir"] == "./chg/"


def test_load_extra_files_rejects_wrong_node_type(aiida_profile_clean):
    """extra_files must reference SinglefileData/FolderData, not e.g. a Dict."""
    import aiida.orm as orm_mod
    from aiida import orm
    from aiida_uranium_workflow.input_builders.base import resolve_extra_files

    d = orm.Dict(dict={"a": 1})
    orig = orm_mod.load_node
    orm_mod.load_node = lambda ident: d
    try:
        with pytest.raises(TypeError, match="SinglefileData or FolderData"):
            resolve_extra_files({"chg": 123})
    finally:
        orm_mod.load_node = orig
