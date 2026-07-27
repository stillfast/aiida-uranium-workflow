"""Tests for ``utils/copy_remote.py`` and the unified ``copy`` subcommand.

These tests deliberately avoid spinning up a real AiiDA profile —
``copy_remote`` is a thin orchestrator over ``load_node``,
``Computer.get_transport`` and ``Transport.gettree``. We replace each
of those seams with a fake so we can exercise the bookkeeping (path
layout, deduplication, dry-run, error handling) without touching the
real filesystem or SSH layers.

The test suite is organised as:

* :class:`TestSanitisePathComponent` — directory-name sanitiser.
* :class:`TestBuildLocalPath` — ``PATH/<backend>/<key>/<preset>/<calc>``.
* :class:`TestCopyRemoteFolderTransport` — the single transport helper.
* :class:`TestResolveCopyTargets` — dedup + destination planning.
* :class:`TestIterCopyTargets` — provenance walking with mocked AiiDA.
* :class:`TestExecuteCopyPlan` — end-to-end dry + wet runs.
* :class:`TestLoadCopyPlan` — top-level glue (JSON → plan).
* :class:`TestCopySubcommandCli` — argparse plumbing for ``copy``.
"""

from __future__ import annotations

from aiida_uranium_workflow.utils.copy_remote import (
    CopyError,
    CopyPlan,
    CopyPlanEntry,
    CopyTarget,
    build_local_path,
    collect_remote_folder_calcjobs,
    copy_remote_folder_to_local,
    execute_copy_plan,
    iter_copy_targets,
    load_copy_plan,
    resolve_copy_targets,
    sanitise_path_component,
)

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import json
import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRemoteData:
    """Minimal stand-in for ``aiida.orm.RemoteData``."""

    def __init__(self, remote_path: str, computer: Any = None):
        self._remote_path = remote_path
        self.computer = computer or SimpleNamespace(label="fake-comp")

    def get_remote_path(self) -> str:
        return self._remote_path


class _FakeOutputs:
    """Duck-typed stand-in for the ``outputs`` attribute mapping.

    The real AiiDA ``outputs`` proxy supports both attribute access
    (``node.outputs.remote_folder``) and membership tests
    (``"remote_folder" in node.outputs``). :class:`SimpleNamespace`
    supports the former but raises ``TypeError`` on the latter, so we
    emulate the proxy ourselves.
    """

    def __init__(self, remote_folder=None):
        self._data: dict[str, Any] = {}
        if remote_folder is not None:
            self._data["remote_folder"] = remote_folder

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name)

    def __contains__(self, name):
        return name in self._data

    def __iter__(self):
        return iter(self._data)


class _FakeCalcJobNode:
    """Stand-in for ``aiida.orm.CalcJobNode``.

    Only the attributes consumed by ``copy_remote`` are populated.
    """

    is_finished = True  # not used; we copy running ones too
    is_finished_ok = False  # explicitly False to prove we don't filter

    def __init__(self, pk: int, *, remote_path: str | None = None, label: str = ""):
        self.pk = pk
        remote = _FakeRemoteData(remote_path) if remote_path is not None else None
        self.outputs = _FakeOutputs(remote_folder=remote)
        self.process_label = label or f"calc_{pk}"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<FakeCalcJobNode pk={self.pk}>"


class _FakeWorkChainNode:
    """Stand-in for ``aiida.orm.WorkChainNode``.

    Tracks ``called`` (direct children) and ``process_class``.
    """

    def __init__(self, pk: int, *, class_name: str = "AbacusSmearWorkChain", called=None):
        self.pk = pk
        self.called = list(called or [])
        self.process_label = f"wc_{pk}"
        self.process_class = SimpleNamespace(__name__=class_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_target(
    *,
    wc_pk: int = 1,
    backend: str = "abacus",
    preset: str = "lcao",
    key: str = "smear",
    calcjob_pk: int = 10,
    calcjob_label: str = "abacus_gauss_0_06",
    remote_path: str = "/work/run_10",
) -> CopyTarget:
    return CopyTarget(
        wc_pk=wc_pk,
        wc_label=f"wc_{wc_pk}",
        backend=backend,
        preset=preset,
        key=key,
        calcjob_pk=calcjob_pk,
        calcjob_label=calcjob_label,
        remote_path=remote_path,
        computer=SimpleNamespace(label="fake-comp"),
    )


# ---------------------------------------------------------------------------
# Sanitise / build_local_path
# ---------------------------------------------------------------------------


class TestSanitisePathComponent:
    """``sanitise_path_component`` makes labels filesystem-safe."""

    def test_alphanumeric_unchanged(self):
        assert sanitise_path_component("lcao") == "lcao"
        assert sanitise_path_component("pw_r") == "pw_r"

    def test_spaces_and_slashes_replaced(self):
        # ``replace_all`` characters: only alnum + ``-_.`` survive
        # everything else becomes ``_``. The dot stays untouched so
        # ``ecutwfc_60_kp_0.2`` is the expected output (the legacy
        # smear scripts use ``replace(".", "_")`` themselves).
        assert sanitise_path_component("ecutwfc 60 kp 0.2") == "ecutwfc_60_kp_0.2"
        assert "/" not in sanitise_path_component("a/b")
        assert " " not in sanitise_path_component("a b")

    def test_strip_leading_trailing_dots_dashes(self):
        # Leading/trailing dots and dashes get stripped so we don't
        # accidentally create a hidden directory or escape the parent.
        assert sanitise_path_component("..lcao..") == "lcao"
        assert sanitise_path_component("---pw---") == "pw"

    def test_empty_string_becomes_underscore(self):
        assert sanitise_path_component("") == "_"


class TestBuildLocalPath:
    """``build_local_path`` composes the four-level destination layout."""

    def test_layout(self, tmp_path):
        result = build_local_path(
            tmp_path,
            backend="abacus",
            key="smear",
            preset="lcao",
            calcjob_label_or_pk="abacus_gauss_0_06",
        )
        expected = (
            tmp_path
            / "abacus"
            / "smear"
            / "lcao"
            / "abacus_gauss_0_06"
        )
        assert result == expected

    def test_sanitises_each_component(self, tmp_path):
        # ``key`` contains spaces; preset is fine. The result must
        # still be safe (no spaces in any directory component).
        result = build_local_path(
            tmp_path,
            backend="vasp",
            key="vasp smearing",
            preset="test preset",
            calcjob_label_or_pk="pk-99",
        )
        for part in result.parts:
            assert " " not in part, f"unexpected space in {part}"


# ---------------------------------------------------------------------------
# Transport helper
# ---------------------------------------------------------------------------


class TestCopyRemoteFolderTransport:
    """``copy_remote_folder_to_local`` is the only byte-moving piece."""

    def test_calls_gettree_with_remote_and_local_paths(self, tmp_path):
        remote = _FakeRemoteData("/work/run_42")
        seen = {}

        def fake_transport():
            class T:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return False

                def gettree(self_inner, remotepath, localpath):
                    seen["remotepath"] = remotepath
                    seen["localpath"] = localpath

            return T()

        copy_remote_folder_to_local(
            remote, tmp_path / "dst", transport_factory=lambda c: fake_transport()
        )
        assert seen["remotepath"] == "/work/run_42/"
        assert seen["localpath"] == str(tmp_path / "dst")

    def test_creates_destination_dir(self, tmp_path):
        remote = _FakeRemoteData("/work/run_42")
        dst = tmp_path / "new-dir" / "dst"

        def factory(c):
            t = MagicMock()
            t.__enter__ = lambda self: t
            t.__exit__ = lambda self, *a: False
            return t

        copy_remote_folder_to_local(remote, dst, transport_factory=factory)
        assert dst.is_dir()

    def test_raises_copy_error_on_transport_factory_failure(self):
        remote = _FakeRemoteData("/work/run_42")

        def bad_factory(c):
            raise RuntimeError("nope")

        with pytest.raises(CopyError, match="could not open transport"):
            copy_remote_folder_to_local(remote, "/tmp/x", transport_factory=bad_factory)

    def test_raises_copy_error_on_gettree_failure(self, tmp_path):
        remote = _FakeRemoteData("/work/run_42")

        def factory(c):
            t = MagicMock()
            t.__enter__ = lambda self: t
            t.__exit__ = lambda self, *a: False

            def boom(*a, **kw):
                raise RuntimeError("network down")

            t.gettree = boom
            return t

        with pytest.raises(CopyError, match="gettree.*failed"):
            copy_remote_folder_to_local(remote, tmp_path / "x", transport_factory=factory)


# ---------------------------------------------------------------------------
# resolve_copy_targets
# ---------------------------------------------------------------------------


class TestResolveCopyTargets:
    """``resolve_copy_targets`` deduplicates CalcJobs and plans paths."""

    def test_single_target(self, tmp_path):
        target = _build_target()
        entries, skipped = resolve_copy_targets([target], base_dir=tmp_path)
        assert len(entries) == 1
        assert skipped == []
        assert entries[0].local_path == build_local_path(
            tmp_path, backend="abacus", key="smear", preset="lcao",
            calcjob_label_or_pk="abacus_gauss_0_06",
        )

    def test_dedup_by_calcjob_pk(self, tmp_path):
        t1 = _build_target(calcjob_pk=10)
        t2 = _build_target(calcjob_pk=10)  # same pk, ignored
        entries, skipped = resolve_copy_targets([t1, t2], base_dir=tmp_path)
        assert len(entries) == 1
        assert any(reason == "duplicate calcjob pk" for _, reason in skipped)

    def test_empty_remote_path_is_skipped(self, tmp_path):
        t = _build_target(remote_path="")
        entries, skipped = resolve_copy_targets([t], base_dir=tmp_path)
        assert entries == []
        assert any(reason == "no remote_path" for _, reason in skipped)

    def test_empty_targets_returns_empty_plan(self, tmp_path):
        entries, skipped = resolve_copy_targets([], base_dir=tmp_path)
        assert entries == []
        assert skipped == []


# ---------------------------------------------------------------------------
# iter_copy_targets (provenance walking with mocked AiiDA)
# ---------------------------------------------------------------------------


class TestIterCopyTargets:
    """``iter_copy_targets`` reads ``load_node`` + ``process_class``."""

    def _patch_load_node(self, monkeypatch, registry):
        """Map ``pk -> WorkChainNode`` for ``copy_remote.iter_copy_targets``.

        Also replace the ``WorkChainNode`` / ``CalcJobNode`` references
        used for the ``isinstance`` checks with the test's stand-in
        classes so the fake WorkChains / CalcJobs are accepted.
        """
        from aiida_uranium_workflow.utils import copy_remote

        def fake_load_node(identifier):
            try:
                return registry[int(identifier)]
            except (KeyError, ValueError, TypeError):
                return registry[str(identifier)]

        monkeypatch.setattr(copy_remote, "load_node", fake_load_node)
        monkeypatch.setattr(copy_remote, "load_profile", lambda *_a, **_kw: None)
        # Replace the type sentinels used for isinstance checks so the
        # test's stand-in classes are recognised.
        monkeypatch.setattr(copy_remote, "WorkChainNode", _FakeWorkChainNode)
        monkeypatch.setattr(copy_remote, "CalcJobNode", _FakeCalcJobNode)
        monkeypatch.setattr(copy_remote, "RemoteData", _FakeRemoteData)

    def test_simple_walk_yields_one_target_per_calcjob(self, monkeypatch):
        calc1 = _FakeCalcJobNode(10, remote_path="/work/run_10", label="abacus_gauss")
        calc2 = _FakeCalcJobNode(11, remote_path="/work/run_11", label="abacus_mp")
        wc = _FakeWorkChainNode(1, class_name="AbacusSmearWorkChain", called=[calc1, calc2])
        self._patch_load_node(monkeypatch, {1: wc})

        pk_map = {"abacus": {"smear": {"lcao": 1, "pw": 2}}}
        # We only have one wc_pk in the registry; both leaves share it.
        pk_map = {"abacus": {"smear": {"lcao": 1}}}

        targets = list(
            iter_copy_targets(
                pk_map=pk_map,
                class_to_backend={"AbacusSmearWorkChain": "abacus"},
            )
        )

        assert len(targets) == 2
        # Order is implementation-defined; compare on pks.
        pks = sorted(t.calcjob_pk for t in targets)
        assert pks == [10, 11]
        for t in targets:
            assert t.backend == "abacus"
            assert t.preset == "lcao"
            assert t.key == "smear"
            assert t.remote_path.startswith("/work/run_")

    def test_skips_calcjobs_without_remote_folder(self, monkeypatch):
        calc_with = _FakeCalcJobNode(10, remote_path="/work/run_10")
        calc_without = _FakeCalcJobNode(11)  # no outputs.remote_folder
        wc = _FakeWorkChainNode(1, called=[calc_with, calc_without])
        self._patch_load_node(monkeypatch, {1: wc})

        targets = list(
            iter_copy_targets(
                pk_map={"abacus": {"smear": {"x": 1}}},
                class_to_backend={"AbacusSmearWorkChain": "abacus"},
            )
        )
        assert [t.calcjob_pk for t in targets] == [10]

    def test_method_mismatch_raises(self, monkeypatch):
        # JSON declares ``abacus`` but the WorkChain is Vasp — that
        # should raise so the CLI can surface a clean error.
        wc = _FakeWorkChainNode(1, class_name="VaspSmearWorkChain", called=[])
        self._patch_load_node(monkeypatch, {1: wc})

        with pytest.raises(ValueError, match="backend"):
            list(
                iter_copy_targets(
                    pk_map={"abacus": {"smear": {"x": 1}}},
                    class_to_backend={
                        "AbacusSmearWorkChain": "abacus",
                        "VaspSmearWorkChain": "vasp",
                    },
                )
            )

    def test_unknown_class_raises(self, monkeypatch):
        wc = _FakeWorkChainNode(1, class_name="TotallyMadeUpWorkChain", called=[])
        self._patch_load_node(monkeypatch, {1: wc})

        with pytest.raises(ValueError, match="not part of method"):
            list(
                iter_copy_targets(
                    pk_map={"abacus": {"smear": {"x": 1}}},
                    class_to_backend={"AbacusSmearWorkChain": "abacus"},
                )
            )

    def test_calcjob_label_uses_resolver_when_inputs_available(self, monkeypatch):
        """When the CalcJob exposes the relevant inputs, ``calcjob_label``
        should reflect the semantics-aware label (here, the ABACUS
        convergence format) — not the generic ``process_label``.

        This guards against future regressions of the integration
        between :mod:`utils.labels` and :func:`iter_copy_targets`.
        """
        from aiida_uranium_workflow.utils import copy_remote
        from tests.test_labels import (
            _FakeDict,
            _FakeFloat,
            _FakeInputs,
        )

        inputs = _FakeInputs(
            {
                "abacus": {
                    "parameters": _FakeDict({"input": {"ecutwfc": 60}}),
                    "structure": object(),
                },
                "kpoints_distance": _FakeFloat(0.2),
            }
        )
        calc = _FakeCalcJobNode(10, remote_path="/work/run_10")
        calc.inputs = inputs
        calc.metadata = SimpleNamespace(label="")
        calc.caller = None

        wc = _FakeWorkChainNode(
            1, class_name="AbacusConvergenceWorkChain", called=[calc]
        )
        self._patch_load_node(monkeypatch, {1: wc})

        targets = list(
            iter_copy_targets(
                pk_map={"abacus": {"convergence": {"lcao": 1}}},
                class_to_backend={"AbacusConvergenceWorkChain": "abacus"},
            )
        )
        assert len(targets) == 1
        assert (
            targets[0].calcjob_label == "ecutwfc_60_kpoints_distance_0_2"
        )

    def test_calcjob_label_falls_back_to_process_label(self, monkeypatch):
        """When no inputs are available, the resolver falls back to
        ``process_label`` so the directory name is at least readable
        (better than a bare pk)."""

        calc_with_empty = _FakeCalcJobNode(10, remote_path="/work/run_10")
        wc = _FakeWorkChainNode(
            1, class_name="AbacusConvergenceWorkChain", called=[calc_with_empty]
        )
        self._patch_load_node(monkeypatch, {1: wc})

        targets = list(
            iter_copy_targets(
                pk_map={"abacus": {"convergence": {"lcao": 1}}},
                class_to_backend={"AbacusConvergenceWorkChain": "abacus"},
            )
        )
        assert len(targets) == 1
        # The fallback goes to ``process_label`` (set by ``_FakeCalcJobNode``).
        assert targets[0].calcjob_label == "calc_10"


# ---------------------------------------------------------------------------
# execute_copy_plan
# ---------------------------------------------------------------------------


class TestExecuteCopyPlan:
    """``execute_copy_plan`` drives the transport and reports failures."""

    def test_no_transport_factory_fails(self, tmp_path, monkeypatch):
        """Without a transport_factory, ``computer.get_transport`` raises.

        We assert that ``execute_copy_plan`` reports the failure as a
        non-zero return rather than crashing the whole loop.
        """
        from aiida_uranium_workflow.utils import copy_remote

        fake_node = SimpleNamespace(
            outputs=SimpleNamespace(
                remote_folder=_FakeRemoteData("/work/run_10"),
            ),
        )
        monkeypatch.setattr(copy_remote, "load_node", lambda pk: fake_node)

        target = _build_target()
        entry = CopyPlanEntry(target=target, local_path=tmp_path / "x")

        # No transport_factory -> copy_remote_folder_to_local tries
        # to call ``computer.get_transport`` and fails. ``execute_copy_plan``
        # captures that as a per-entry failure.
        success, failures = execute_copy_plan([entry])
        assert success == 0
        assert len(failures) == 1
        # The failure message comes from the CopyError raised by the
        # helper (no transport factory -> attribute lookup fails).
        assert "could not open transport" in failures[0][1] or "gettree" in failures[0][1]

    def test_transport_factory_is_used(self, tmp_path):
        target = _build_target(calcjob_pk=10, remote_path="/work/run_10")
        entry = CopyPlanEntry(target=target, local_path=tmp_path / "out")

        # Stub ``load_node`` via monkeypatch to return a node whose
        # ``outputs.remote_folder`` is a ``_FakeRemoteData``.
        from aiida_uranium_workflow.utils import copy_remote

        fake_node = SimpleNamespace(
            outputs=SimpleNamespace(
                remote_folder=_FakeRemoteData("/work/run_10"),
            ),
        )
        original_load_node = copy_remote.load_node
        copy_remote.load_node = lambda pk: fake_node
        try:
            calls = []

            def factory(_computer):
                class T:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *exc):
                        return False

                    def gettree(self_inner, remotepath, localpath):
                        calls.append((remotepath, localpath))

                return T()

            success, failures = execute_copy_plan([entry], transport_factory=factory)
            assert success == 1
            assert failures == []
            assert calls == [("/work/run_10/", str(tmp_path / "out"))]
        finally:
            copy_remote.load_node = original_load_node

    def test_one_failure_does_not_abort_others(self, tmp_path):
        from aiida_uranium_workflow.utils import copy_remote

        good = CopyPlanEntry(
            target=_build_target(calcjob_pk=10, remote_path="/work/run_10"),
            local_path=tmp_path / "good",
        )
        # Craft a target whose calcjob_pk we cannot resolve: leave
        # ``load_node`` to raise.
        bad = CopyPlanEntry(
            target=_build_target(calcjob_pk=11, remote_path="/work/run_11"),
            local_path=tmp_path / "bad",
        )

        # Map pk -> fake node; missing pk raises on access.
        def fake_loader(pk):
            if pk == 10:
                return SimpleNamespace(
                    outputs=SimpleNamespace(
                        remote_folder=_FakeRemoteData("/work/run_10"),
                    )
                )
            raise RuntimeError("not in registry")

        original = copy_remote.load_node
        copy_remote.load_node = fake_loader
        try:
            calls = []

            def factory(_c):
                class T:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *exc):
                        return False

                    def gettree(self_inner, remotepath, localpath):
                        calls.append((remotepath, localpath))

                return T()

            success, failures = execute_copy_plan(
                [good, bad], transport_factory=factory
            )
            assert success == 1
            assert len(failures) == 1
            assert calls == [("/work/run_10/", str(tmp_path / "good"))]
        finally:
            copy_remote.load_node = original


# ---------------------------------------------------------------------------
# load_copy_plan (top-level glue)
# ---------------------------------------------------------------------------


class TestLoadCopyPlan:
    """``load_copy_plan`` is the JSON→plan wrapper the CLI uses."""

    def _patch_load_node(self, monkeypatch, registry):
        from aiida_uranium_workflow.utils import copy_remote

        def fake_load_node(identifier):
            return registry[int(identifier)]

        monkeypatch.setattr(copy_remote, "load_node", fake_load_node)
        monkeypatch.setattr(copy_remote, "load_profile", lambda *_a, **_kw: None)
        # Replace the type sentinels used for isinstance checks.
        monkeypatch.setattr(copy_remote, "WorkChainNode", _FakeWorkChainNode)
        monkeypatch.setattr(copy_remote, "CalcJobNode", _FakeCalcJobNode)
        monkeypatch.setattr(copy_remote, "RemoteData", _FakeRemoteData)

    def test_reads_output_json(self, tmp_path, monkeypatch):
        calc = _FakeCalcJobNode(10, remote_path="/work/run_10")
        wc = _FakeWorkChainNode(1, class_name="AbacusSmearWorkChain", called=[calc])
        self._patch_load_node(monkeypatch, {1: wc})

        out_json = tmp_path / "out.json"
        out_json.write_text(json.dumps({"abacus": {"smear": {"lcao": 1}}}))

        plan = load_copy_plan(
            input_json=out_json,
            method="smear",
            class_to_backend={"AbacusSmearWorkChain": "abacus"},
            base_dir=tmp_path / "dest",
        )
        assert isinstance(plan, CopyPlan)
        assert len(plan) == 1
        assert plan.entries[0].target.calcjob_pk == 10

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        self._patch_load_node(monkeypatch, {})
        from aiida_uranium_workflow.cli._common import collect_pk_map

        with pytest.raises(ValueError, match="not found"):
            load_copy_plan(
                input_json=tmp_path / "nope.json",
                method="smear",
                class_to_backend={"AbacusSmearWorkChain": "abacus"},
                base_dir=tmp_path / "dest",
            )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


class TestCopySubcommandCli:
    """``copy`` subcommand argparse wiring."""

    def test_help_lists_copy(self, capsys):
        from aiida_uranium_workflow.cli._common import build_unified_parser

        parser = build_unified_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["copy", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--method" in out
        assert "-i" in out and "--input" in out
        assert "-o" in out and "--output" in out
        assert "--dry-run" in out

    def test_required_arguments(self):
        from aiida_uranium_workflow.cli._common import build_unified_parser

        parser = build_unified_parser()
        # Missing -i / -o
        with pytest.raises(SystemExit):
            parser.parse_args(["copy", "--method", "smear"])
        # Missing --method
        with pytest.raises(SystemExit):
            parser.parse_args(["copy", "-i", "x.json", "-o", "x"])

    def test_full_command_parses(self, tmp_path):
        from aiida_uranium_workflow.cli._common import build_unified_parser

        parser = build_unified_parser()
        ns = parser.parse_args(
            [
                "copy",
                "--method",
                "smear",
                "-i",
                str(tmp_path / "i.json"),
                "-o",
                str(tmp_path / "o"),
                "-p",
                "alt-profile",
                "--dry-run",
            ]
        )
        assert ns.command == "copy"
        assert ns.method == "smear"
        assert ns.input_json == str(tmp_path / "i.json")
        assert ns.output == str(tmp_path / "o")
        assert ns.profile == "alt-profile"
        assert ns.dry_run is True

    def test_unknown_method_rejected(self):
        from aiida_uranium_workflow.cli._common import build_unified_parser

        parser = build_unified_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["copy", "--method", "bogus", "-i", "x.json", "-o", "x"]
            )


class TestCopyDispatchHandler:
    """The ``_copy`` handler delegates to ``load_copy_plan`` and exit codes."""

    def test_dry_run_returns_zero_no_transport_called(
        self, tmp_path, monkeypatch, capsys
    ):
        from aiida_uranium_workflow.cli import main as main_mod

        # Build a plan with one entry.
        target = _build_target(remote_path="/work/run_10")
        plan = CopyPlan(
            entries=[CopyPlanEntry(target=target, local_path=tmp_path / "p")],
            skipped=[],
        )

        # Stub the loader to return our prebuilt plan.
        monkeypatch.setattr(main_mod, "load_copy_plan", lambda **kw: plan)

        # Stub the executor to assert it is NOT called in --dry-run.
        called = {"executed": False}

        def fake_execute(entries, **kw):
            called["executed"] = True
            return 0, []

        monkeypatch.setattr(main_mod, "_execute_copy_plan", fake_execute)

        # ``load_profile`` must be a no-op for the test.
        monkeypatch.setattr(main_mod, "load_profile", lambda *_a, **_kw: None)

        rc = main_mod.main(
            [
                "copy",
                "--method",
                "smear",
                "-i",
                str(tmp_path / "in.json"),
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        assert rc == 0
        assert called["executed"] is False
        out = capsys.readouterr().out
        assert "/work/run_10" in out
        assert str(tmp_path / "p") in out

    def test_no_plan_entries_returns_one(self, tmp_path, monkeypatch, capsys):
        from aiida_uranium_workflow.cli import main as main_mod

        plan = CopyPlan(entries=[], skipped=[])
        monkeypatch.setattr(main_mod, "load_copy_plan", lambda **kw: plan)
        monkeypatch.setattr(main_mod, "load_profile", lambda *_a, **_kw: None)

        rc = main_mod.main(
            [
                "copy",
                "--method",
                "smear",
                "-i",
                str(tmp_path / "in.json"),
                "-o",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1
        assert "no copyable remote_folder" in capsys.readouterr().err

    def test_partial_failure_returns_one(self, tmp_path, monkeypatch, capsys):
        from aiida_uranium_workflow.cli import main as main_mod

        target = _build_target()
        plan = CopyPlan(
            entries=[CopyPlanEntry(target=target, local_path=tmp_path / "p")],
            skipped=[],
        )
        monkeypatch.setattr(main_mod, "load_copy_plan", lambda **kw: plan)
        monkeypatch.setattr(main_mod, "load_profile", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            main_mod,
            "_execute_copy_plan",
            lambda entries, **kw: (0, [(entries[0], "boom")]),
        )

        rc = main_mod.main(
            [
                "copy",
                "--method",
                "smear",
                "-i",
                str(tmp_path / "in.json"),
                "-o",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1
        out = capsys.readouterr()
        assert "boom" in out.err

    def test_full_success_returns_zero(self, tmp_path, monkeypatch, capsys):
        from aiida_uranium_workflow.cli import main as main_mod

        target = _build_target()
        plan = CopyPlan(
            entries=[CopyPlanEntry(target=target, local_path=tmp_path / "p")],
            skipped=[],
        )
        monkeypatch.setattr(main_mod, "load_copy_plan", lambda **kw: plan)
        monkeypatch.setattr(main_mod, "load_profile", lambda *_a, **_kw: None)
        monkeypatch.setattr(main_mod, "_execute_copy_plan", lambda entries, **kw: (1, []))

        rc = main_mod.main(
            [
                "copy",
                "--method",
                "smear",
                "-i",
                str(tmp_path / "in.json"),
                "-o",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0