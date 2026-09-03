"""Tests for the DB-aware dry-run preview writer (``check --out``).

Covers the ORM-node → plain-YAML serializer and the one-file-per-WorkChain
writer. ``WorkflowOrchestrator.prepare`` itself is exercised against the
fake-adapter harness in ``test_orchestrator_multi_preset.py``; here the
orchestrator is stubbed so no live AiiDA profile / codes are required.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]


# ---------------------------------------------------------------------------
# serialize_inputs — ORM nodes → plain YAML values
# ---------------------------------------------------------------------------


class TestSerializeInputs:
    def test_plain_values_pass_through(self):
        from aiida_uranium_workflow.cli.preview import serialize_inputs

        inputs = {
            "a": 1,
            "b": "text",
            "c": [1.0, 2.0],
            "d": {"nested": True, "none": None},
        }
        assert serialize_inputs(inputs) == inputs

    def test_orm_nodes_convert(self, aiida_profile_clean):
        from aiida import orm
        from aiida_uranium_workflow.cli.preview import serialize_inputs

        out = serialize_inputs(
            {
                "params": orm.Dict(dict={"k": [1, 2], "s": "x"}),
                "items": orm.List(list=[3, 4]),
                "mesh": _kpoints_mesh(2),
                "num": orm.Float(1.5),
                "text": orm.Str("hi"),
                "flag": orm.Bool(True),
            }
        )
        assert out["params"] == {"k": [1, 2], "s": "x"}
        assert out["items"] == [3, 4]
        assert out["mesh"]["kpoints_mesh"] == [2, 2, 2]
        assert out["num"] == 1.5
        assert out["text"] == "hi"
        assert out["flag"] is True


def _kpoints_mesh(n: int):
    """An unstored KpointsData with an n×n×n mesh (a cell must be set
    before ``set_kpoints_mesh`` can derive the offset)."""
    from aiida import orm

    kpoints = orm.KpointsData()
    kpoints.set_cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kpoints.set_kpoints_mesh([n, n, n])
    return kpoints


# ---------------------------------------------------------------------------
# write_preview_files — one YAML per planned WorkChain
# ---------------------------------------------------------------------------


class FakeWorkChain:
    __name__ = "FakeWorkChain"


def test_write_preview_files_writes_one_yaml_per_job(tmp_path, monkeypatch):
    """Every PreparedJob becomes its own YAML under <out>/<workflow>/."""
    from aiida_uranium_workflow import schedulers
    from aiida_uranium_workflow.cli import preview as preview_module
    from aiida_uranium_workflow.schedulers.base import PreparedJob

    class FakeOrchestrator:
        def prepare(self, profile=None):
            return [
                PreparedJob(
                    backend="abacus",
                    preset_name="preset0",
                    structure_name="bcc-uranium",
                    workchain_cls=FakeWorkChain,
                    inputs={"params": {"ecutwfc": 65}},
                ),
                PreparedJob(
                    backend="vasp",
                    preset_name="preset0/test_soc",
                    structure_name="bcc-uranium",
                    workchain_cls=FakeWorkChain,
                    inputs={"encut": 400},
                ),
            ]

    # ``write_preview_files`` imports get_orchestrator lazily from the
    # schedulers package — patch it there.
    monkeypatch.setattr(
        schedulers, "get_orchestrator", lambda bundle: FakeOrchestrator()
    )

    bundle = _simple_bundle()
    files = preview_module.write_preview_files(bundle, tmp_path)

    assert len(files) == 2
    out_dir = tmp_path / "smear"
    assert (out_dir / "abacus_preset0_bcc-uranium.yml").is_file()
    assert (out_dir / "vasp_preset0_test_soc_bcc-uranium.yml").is_file()
    content = files[0].read_text()
    assert content.startswith("workflow: smear\n")
    assert "ecutwfc: 65" in content
    assert "workchain: FakeWorkChain" in content
    # The single-name protocol slot (smear: test) is recorded.
    assert "protocol_preset: test" in content


def _simple_bundle():
    from aiida_uranium_workflow.utils.common import ParamBundle

    return ParamBundle(
        input_params={
            "workflow": "smear",
            "profile": "aiida_profile",
            "parameters": {"smear": "test"},
            "static": {"structure": "bcc-uranium", "metadata": "yunhe"},
            "code": {},
        },
        protocol={},
        workflow_data={},
        software_params={},
        metadata={},
    )
