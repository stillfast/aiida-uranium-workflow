"""Tests for the child parser classes (``utils/parsers/child.py``).

Each backend parser turns a child node into a normalised
:class:`ChildResult` (pk / status / energy eV / time s / scf steps /
atoms + backend magnetism payload).  Tests use lightweight fake child
objects (duck-typed) so no AiiDA process is needed; the summary fields
(energy / time / steps) come from ``fetch_summary`` which is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _fake_node(pk: int, *, finished_ok=True, exit_status=0, outputs=None):
    """A minimal duck-typed child process node."""
    return SimpleNamespace(
        pk=pk,
        is_finished_ok=finished_ok,
        exit_status=exit_status,
        outputs=outputs or SimpleNamespace(),
    )


def _misc_dict(data: dict):
    """An object with ``get_dict()`` standing in for an orm.Dict output."""
    return SimpleNamespace(get_dict=lambda: data)


def _stub_summary(monkeypatch, summary: dict):
    """Stub ``fetch_summary`` so parsers don't touch real logs / inputs."""
    import aiida_uranium_workflow.utils.parsers as parsers

    monkeypatch.setattr(
        parsers, "fetch_summary", lambda node, backend: dict(summary)
    )


class TestParserBaseBehaviour:
    def test_unfinished_child_not_parsed(self):
        """A child that did not finish OK yields a bare result."""
        from aiida_uranium_workflow.utils.parsers.child import (
            AbacusChildParser,
        )

        node = _fake_node(pk=42, finished_ok=False, exit_status=None)
        result = AbacusChildParser().parse(node)

        assert result.pk == 42
        assert result.status is None
        assert result.finished_ok is False
        # No outputs were touched — magnetic stays empty.
        assert result.magnetic == {}
        assert result.energy_ev is None

    def test_exit_status_captured(self):
        from aiida_uranium_workflow.utils.parsers.child import (
            VaspChildParser,
        )

        node = _fake_node(pk=7, finished_ok=False, exit_status=300)
        result = VaspChildParser().parse(node)
        assert result.status == 300
        assert result.finished_ok is False

    def test_parse_never_raises_on_missing_outputs(self):
        """A child whose outputs are missing yields an empty result."""
        from aiida_uranium_workflow.utils.parsers.child import (
            QePwChildParser,
        )

        node = _fake_node(pk=1, outputs=SimpleNamespace())  # no output_parameters
        result = QePwChildParser().parse(node)
        assert result.pk == 1
        assert result.finished_ok is True
        assert result.energy_ev is None


class TestAbacusChildParser:
    def test_summary_and_magnetism(self, monkeypatch):
        from aiida_uranium_workflow.utils.parsers.child import (
            AbacusChildParser,
        )

        _stub_summary(
            monkeypatch,
            {"energy_ev": -100.5, "time_s": 12.3, "natoms": 2, "scf_steps": 18},
        )
        node = _fake_node(
            pk=5,
            outputs=SimpleNamespace(
                misc=_misc_dict(
                    {
                        "magnetism": [[0.0], [1.0]],
                        "final_magnetism": 1.0,
                    }
                )
            ),
        )
        node.inputs = SimpleNamespace(
            abacus=SimpleNamespace(
                parameters=_misc_dict({"input": {"nspin": 2}})
            )
        )

        result = AbacusChildParser().parse(node)

        assert result.pk == 5
        assert result.energy_ev == -100.5
        assert result.time_s == 12.3
        assert result.scf_steps == 18
        assert result.natoms == 2
        assert result.magnetic["magnetism"] == [[0.0], [1.0]]
        assert result.magnetic["final_magnetism"] == 1.0
        assert result.magnetic["nspin"] == 2


class TestVaspChildParser:
    def test_summary_and_magnetism(self, monkeypatch):
        from aiida_uranium_workflow.utils.parsers.child import (
            VaspChildParser,
        )

        _stub_summary(
            monkeypatch,
            {"energy_ev": -200.0, "time_s": 99.0, "natoms": 2, "scf_steps": None},
        )
        node = _fake_node(
            pk=9,
            outputs=SimpleNamespace(
                misc=_misc_dict(
                    {
                        "magnetization": 3.1,
                        "site_magnetization": {"sphere": {"x": {}}},
                    }
                )
            ),
        )

        result = VaspChildParser().parse(node)

        assert result.energy_ev == -200.0
        assert result.time_s == 99.0
        assert result.magnetic["magnetization"] == 3.1
        assert result.magnetic["site_magnetization"] == {"sphere": {"x": {}}}


class TestQePwChildParser:
    def test_output_parameters_read(self):
        from aiida_uranium_workflow.utils.parsers.child import (
            QePwChildParser,
        )

        node = _fake_node(
            pk=3,
            outputs=SimpleNamespace(
                output_parameters=_misc_dict(
                    {
                        "energy": -55.0,  # already eV (aiida-qe converts)
                        "wall_time_seconds": 42.5,
                        "total_magnetization": 2.0,
                        "absolute_magnetization": 4.0,
                        "atomic_magnetic_moments": [1.0, 1.0],
                    }
                )
            ),
        )
        node.inputs = SimpleNamespace(
            pw=SimpleNamespace(
                structure=SimpleNamespace(sites=[object(), object()])
            )
        )

        result = QePwChildParser().parse(node)

        assert result.energy_ev == -55.0
        assert result.time_s == 42.5
        assert result.natoms == 2
        # ``magnetization`` mirrors the legacy gather key (= total_magnetization).
        assert result.magnetic["magnetization"] == 2.0
        assert result.magnetic["absolute_magnetization"] == 4.0
        assert result.magnetic["atomic_magnetic_moments"] == [1.0, 1.0]


class TestFleurScfChildParser:
    def test_energy_converted_hartree_to_ev(self):
        from aiida_uranium_workflow.utils.parsers.child import (
            FleurScfChildParser,
        )

        parser = FleurScfChildParser()
        node = _fake_node(
            pk=11,
            outputs=SimpleNamespace(
                output_scf_wc_para=_misc_dict(
                    {"total_energy": -2.0, "total_wall_time": 60.0}
                ),
                last_calc=SimpleNamespace(
                    output_parameters=_misc_dict(
                        {"magnetic_vec_moments": [[0.0, 0.0, 4.0]]}
                    )
                ),
            ),
        )

        result = parser.parse(node)

        assert result.energy_ev == pytest.approx(-2.0 * parser.HA_TO_EV)
        assert result.time_s == 60.0
        # ``magnetization`` mirrors the legacy gather key
        # (= magnetic_vec_moments); Hartree energy kept for native units.
        assert result.magnetic["magnetization"] == [[0.0, 0.0, 4.0]]
        assert result.magnetic["total_energy_hartree"] == -2.0


class TestChildParserRegistry:
    def test_all_backends_registered(self):
        from aiida_uranium_workflow.utils.parsers.child import (
            CHILD_PARSERS,
        )

        assert set(CHILD_PARSERS) == {"abacus", "vasp", "qe", "fleur"}

    def test_get_child_parser_unknown_raises(self):
        from aiida_uranium_workflow.utils.parsers.child import (
            get_child_parser,
        )

        with pytest.raises(ValueError, match="No child parser"):
            get_child_parser("nope")
