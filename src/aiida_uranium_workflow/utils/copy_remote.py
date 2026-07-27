"""Helpers used by the unified ``aiida-uranium copy`` command.

This module deliberately stays side-effect free (except for the
filesystem copy itself) so each function can be unit-tested in
isolation. The legacy ``utils/copy_calc.copy_targets`` scp-based
implementation is left untouched for backwards compatibility; the
copy command uses the AiiDA :class:`~aiida.transports.Transport` API
through :func:`copy_remote_folder_to_local` instead.

The high-level workflow is:

1. :func:`iter_copy_targets` walks the provenance tree of each
   top-level WorkChain and emits one
   :class:`CopyTarget` per ``CalcJobNode`` that owns an
   ``outputs.remote_folder``.
2. :func:`resolve_copy_targets` applies the destination path layout
   (``PATH/<backend>/<key>/<preset>/<calcjob_label>``) and groups
   everything into a :class:`CopyPlan` ready for transport.
3. :func:`copy_remote_folder_to_local` is the only piece that actually
   moves bytes — it opens a transport on the ``RemoteData``'s computer
   and uses ``transport.gettree`` (the AiiDA-recommended API).

Single failures are surfaced as :class:`CopyError` instances; the
caller decides whether to abort or to keep going. By convention the
unified CLI keeps going and reports at the end (see
:func:`execute_copy_plan`).
"""

from __future__ import annotations

from aiida import load_profile
from aiida.orm import CalcJobNode, load_node, Node, RemoteData, WorkChainNode
from aiida_uranium_workflow.utils.copy_calc import collect_identifiers_from_json
from aiida_uranium_workflow.utils.labels import resolve_label

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import sys


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyTarget:
    """One ``CalcJobNode`` → ``RemoteData`` pair that should be copied.

    Attributes:
        wc_pk: pk of the top-level WorkChain this ``CalcJobNode`` belongs to.
        wc_label: process label of that WorkChain (used for logging).
        backend: ``"abacus"`` / ``"vasp"`` from the WorkChain class.
        preset: preset name as recorded in the JSON pk map.
        key: inner JSON key (``"smear"`` / ``"convergence"`` / ``"magmom"`` /
            ``"vasp"``).
        calcjob_pk: pk of the ``CalcJobNode``.
        calcjob_label: process label of the ``CalcJobNode`` (preferred
            destination name; falls back to the pk).
        remote_path: absolute remote path on the compute node.
        computer: AiiDA :class:`~aiida.orm.Computer` that owns the
            ``RemoteData``. Stored so the caller doesn't have to look it
            up again.
    """

    wc_pk: int
    wc_label: str
    backend: str
    preset: str
    key: str
    calcjob_pk: int
    calcjob_label: str
    remote_path: str
    computer: Any = None  # ``Computer`` instance; kept opaque for tests.


@dataclass(frozen=True)
class CopyPlanEntry:
    """A target plus its resolved local destination directory."""

    target: CopyTarget
    local_path: Path


@dataclass
class CopyPlan:
    """Resolved copy plan: a list of entries plus failure counters."""

    entries: list[CopyPlanEntry] = field(default_factory=list)
    skipped: list[tuple[CopyTarget, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


class CopyError(RuntimeError):
    """Raised by :func:`copy_remote_folder_to_local` when transport fails."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def sanitise_path_component(value: str) -> str:
    """Make ``value`` safe to use as a single directory name.

    The CLI uses user-supplied preset names, AiiDA process labels (which
    may contain spaces or ``/``) and the bare ``"convergence"`` style
    keys as directory names. This helper collapses everything outside
    ``[A-Za-z0-9._-]`` into a single ``_`` so the resulting path
    survives round-trips through the shell and most filesystems.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    cleaned = cleaned.strip("._-")
    return cleaned or "_"


def build_local_path(
    base_dir: Path | str,
    *,
    backend: str,
    key: str,
    preset: str,
    calcjob_label_or_pk: str,
) -> Path:
    """Return ``<base>/<backend>/<key>/<preset>/<calcjob_label>``.

    Each component is sanitised through :func:`sanitise_path_component`
    so callers can pass raw labels (with spaces, slashes, …) without
    fearing path traversal.
    """
    base = Path(base_dir)
    return base / sanitise_path_component(backend) / sanitise_path_component(
        key
    ) / sanitise_path_component(preset) / sanitise_path_component(calcjob_label_or_pk)


# ---------------------------------------------------------------------------
# Provenance walking
# ---------------------------------------------------------------------------


def _walk_called_descendants(root: WorkChainNode) -> Iterator[Node]:
    """Yield every descendant process node of ``root`` (depth-first)."""
    queue: list[Node] = list(root.called)
    while queue:
        current = queue.pop(0)
        yield current
        # ``called`` is the list of *direct* children. Recurse so nested
        # WorkChains (e.g. the smear WorkChain spawning a base WorkChain
        # spawning the actual CalcJob) are still covered.
        if hasattr(current, "called"):
            try:
                queue.extend(current.called)
            except Exception:
                # CalcJobs have ``called`` set to an empty list but we
                # still try to extend; if for any reason it raises,
                # swallow it — descendants are best-effort.
                pass


def _has_remote_folder(calcjob: CalcJobNode) -> bool:
    """Return ``True`` when ``calcjob`` has produced a ``remote_folder``.

    CalcJobs are kept around even when they crash before submitting, so
    the mere presence of the node is not enough. We deliberately do
    NOT require ``is_finished_ok`` here: the user asked us to also
    copy ``running`` remote folders, so as long as the
    ``RemoteData`` was created, we honour that request.
    """
    if not isinstance(calcjob, CalcJobNode):
        return False
    try:
        return "remote_folder" in calcjob.outputs
    except Exception:
        return False


def collect_remote_folder_calcjobs(
    workchain: WorkChainNode,
    *,
    predicate: Callable[[CalcJobNode], bool] | None = None,
) -> list[CalcJobNode]:
    """Walk ``workchain``'s descendants and return the eligible CalcJobs.

    A CalcJob is eligible when it has an ``outputs.remote_folder`` and
    the optional ``predicate`` (e.g. a deduplicator) accepts it.

    The walk uses :func:`_walk_called_descendants` so nested WorkChain
    trees are handled — only the **leaf-most** CalcJobs that actually
    own a ``remote_folder`` are returned, which matches what the user
    wants to copy.
    """
    seen_pks: set[int] = set()
    results: list[CalcJobNode] = []
    for node in _walk_called_descendants(workchain):
        if not isinstance(node, CalcJobNode):
            continue
        if not _has_remote_folder(node):
            continue
        if node.pk in seen_pks:
            continue
        if predicate is not None and not predicate(node):
            continue
        seen_pks.add(node.pk)
        results.append(node)
    return results


# ---------------------------------------------------------------------------
# JSON → (backend, key, preset) → CopyTarget
# ---------------------------------------------------------------------------


def _resolve_workchain_class(workchain: WorkChainNode, class_to_backend: Mapping[str, str]) -> tuple[str, str]:
    """Return ``(class_name, backend)`` for ``workchain``; raises on mismatch."""
    class_name = workchain.process_class.__name__
    if class_name not in class_to_backend:
        raise ValueError(
            f"WorkChain id={workchain.pk} has class {class_name!r}, "
            f"which is not part of method's supported set "
            f"{sorted(class_to_backend)}"
        )
    return class_name, class_to_backend[class_name]


def _resolve_remote_folder(calcjob: CalcJobNode) -> RemoteData:
    """Return the ``RemoteData`` produced by ``calcjob``."""
    remote = calcjob.outputs.remote_folder
    if not isinstance(remote, RemoteData):
        # AiiDA's ``outputs.remote_folder`` should always be a
        # ``RemoteData``; defend against schema drift anyway.
        raise ValueError(
            f"calcjob pk={calcjob.pk}: outputs.remote_folder is "
            f"{type(remote).__name__}, expected RemoteData"
        )
    return remote


def _calcjob_label_or_pk(
    calcjob: CalcJobNode,
    *,
    backend: str,
    method: str,
    root_workchain: WorkChainNode | None = None,
) -> str:
    """Return a stable, semantics-aware label for ``calcjob``.

    Defers to :func:`aiida_uranium_workflow.utils.labels.resolve_label`
    so the leaf directory matches what each workflow's
    ``submit_children`` outlined. ``backend`` / ``method`` come from the
    output.json pk-map layout so the resolver can build
    ``ecutwfc_<v>_kpoints_*`` etc. The resolver itself guarantees a
    non-empty string (falls back to ``calcjob_<pk>``), so the caller
    never has to defend against empty labels.
    """
    try:
        return resolve_label(
            calcjob,
            backend=backend,
            method=method,
            root_workchain=root_workchain,
        )
    except Exception:
        pass
    label = getattr(calcjob, "process_label", None)
    if label:
        return str(label)
    return f"calcjob_{getattr(calcjob, 'pk', 'unknown')}"


def iter_copy_targets(
    *,
    pk_map: Mapping[str, Any],
    class_to_backend: Mapping[str, str],
) -> Iterator[CopyTarget]:
    """Yield :class:`CopyTarget` records for every leaf of ``pk_map``.

    ``pk_map`` matches the shape returned by
    :func:`aiida_uranium_workflow.utils.cal_json.build_cal_json`::

        {<backend>: {<key>: {<preset>: <identifier>}}}

    ``identifier`` is either an integer pk (legacy ``output.json``) or
    a UUID string (modern). ``load_node`` accepts both. The backend
    declared in the JSON is cross-checked against the WorkChain class;
    mismatches are raised so the CLI can surface them to the user.
    """
    load_profile()
    for backend, by_key in pk_map.items():
        if not isinstance(by_key, dict):
            continue
        for key, presets in by_key.items():
            if not isinstance(presets, dict):
                continue
            for preset, node_id in presets.items():
                if not isinstance(node_id, (int, str)) or isinstance(node_id, bool):
                    continue
                wc = load_node(node_id)
                if not isinstance(wc, WorkChainNode):
                    raise ValueError(
                        f"identifier {node_id!r} resolved to "
                        f"{type(wc).__name__}, expected WorkChainNode"
                    )
                _class_name, wc_backend = _resolve_workchain_class(
                    wc, class_to_backend
                )
                if wc_backend != backend:
                    raise ValueError(
                        f"identifier {node_id!r} has class "
                        f"{_class_name!r} -> backend {wc_backend!r}, "
                        f"but JSON declares backend {backend!r}"
                    )
                for calcjob in collect_remote_folder_calcjobs(wc):
                    remote = _resolve_remote_folder(calcjob)
                    yield CopyTarget(
                        wc_pk=wc.pk,
                        wc_label=str(getattr(wc, "process_label", "") or f"wc_{wc.pk}"),
                        backend=backend,
                        preset=str(preset),
                        key=key,
                        calcjob_pk=calcjob.pk,
                        calcjob_label=_calcjob_label_or_pk(
                            calcjob,
                            backend=backend,
                            method=key,
                            root_workchain=wc,
                        ),
                        remote_path=remote.get_remote_path(),
                        computer=remote.computer,
                    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _dedup_key(target: CopyTarget) -> int:
    """One CalcJob → one destination folder: deduplicate by ``calcjob_pk``."""
    return target.calcjob_pk


def resolve_copy_targets(
    targets: Iterable[CopyTarget],
    *,
    base_dir: Path | str,
) -> tuple[list[CopyPlanEntry], list[tuple[CopyTarget, str]]]:
    """Group ``targets`` into a list of :class:`CopyPlanEntry`.

    Duplicates (same CalcJob pk) are removed; the first occurrence
    wins. Targets whose ``outputs.remote_folder`` is missing are
    recorded in the ``skipped`` list instead of being silently dropped
    — :func:`collect_remote_folder_calcjobs` already filters them out,
    but the separation here keeps this function independently
    testable.
    """
    seen: set[int] = set()
    entries: list[CopyPlanEntry] = []
    skipped: list[tuple[CopyTarget, str]] = []
    for target in targets:
        if not target.remote_path:
            skipped.append((target, "no remote_path"))
            continue
        key = _dedup_key(target)
        if key in seen:
            skipped.append((target, "duplicate calcjob pk"))
            continue
        seen.add(key)
        local = build_local_path(
            base_dir,
            backend=target.backend,
            key=target.key,
            preset=target.preset,
            calcjob_label_or_pk=target.calcjob_label,
        )
        entries.append(CopyPlanEntry(target=target, local_path=local))
    return entries, skipped


# ---------------------------------------------------------------------------
# Transport driver
# ---------------------------------------------------------------------------


def _find_uuid_subdir(transport: Any, base_path: str) -> str | None:
    """Find the UUID subdirectory under the remote base path.

    AiiDA CalcJobs create a UUID-named subdirectory under the remote
    working directory. This function returns the full path to that UUID
    subdirectory.

    Returns:
        The full path to the UUID subdirectory, or None if not found.
    """
    try:
        entries = transport.listdir(base_path)
    except Exception:
        return None

    for entry in entries:
        parts = entry.split("-")
        if len(parts) == 5:
            try:
                uuid_path = f"{base_path}/{entry}"
                stat = transport.stat(uuid_path)
                if stat.st_mode & 0o40000:
                    return uuid_path
            except Exception:
                continue
    return None


def copy_remote_folder_to_local(
    remote_folder: RemoteData,
    local_path: Path | str,
    *,
    transport_factory: Callable[[Any], Any] | None = None,
) -> None:
    """Copy ``remote_folder``'s contents into ``local_path`` via AiiDA transport.

    This is the single function that actually moves bytes. It uses
    :meth:`Computer.get_transport` (the canonical AiiDA entry point)
    combined with ``transport.gettree`` (the canonical recursive-fetch
    primitive). The transport itself is used as a context manager so
    the connection is closed even on failure.

    Args:
        remote_folder: The :class:`~aiida.orm.RemoteData` whose contents
            should be pulled.
        local_path: Destination directory on the **current** host. The
            directory is created if it doesn't exist; existing files
            are not deleted — AiiDA's ``gettree`` will overwrite files
            of the same name (which is the behaviour the legacy
            script used, modulo the scp tool).
        transport_factory: Optional override for tests. When ``None``
            the real ``remote_folder.computer.get_transport`` is used;
            tests can pass a fake factory that yields a mock transport
            to avoid opening real connections.

    Raises:
        CopyError: when the transport factory, ``gettree``, or any
            related AiiDA call fails.
    """
    try:
        computer = remote_folder.computer
    except Exception as exc:  # pragma: no cover - defensive
        raise CopyError(f"could not resolve computer for RemoteData: {exc}") from exc

    factory = transport_factory or (lambda c: c.get_transport())
    try:
        transport = factory(computer)
    except Exception as exc:
        raise CopyError(f"could not open transport on {computer.label}: {exc}") from exc

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        with transport:
            remote_path = remote_folder.get_remote_path()
            source_path = _find_uuid_subdir(transport, remote_path) or remote_path
            source_with_trailing_slash = f"{source_path}/"
            transport.gettree(source_with_trailing_slash, str(local_path))
    except Exception as exc:
        raise CopyError(
            f"gettree({remote_folder.get_remote_path()} -> {local_path}) "
            f"failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def execute_copy_plan(
    entries: Sequence[CopyPlanEntry],
    *,
    transport_factory: Callable[[Any], Any] | None = None,
) -> tuple[int, list[tuple[CopyPlanEntry, str]]]:
    """Copy every entry and return ``(success_count, failures)``.

    The function never raises on a per-entry failure — it captures the
    :class:`CopyError` and reports it in the returned list. The CLI
    uses the failure count to decide on the process exit code.
    """
    failures: list[tuple[CopyPlanEntry, str]] = []
    success = 0
    for entry in entries:
        try:
            remote = load_node(entry.target.calcjob_pk).outputs.remote_folder
        except Exception as exc:
            failures.append((entry, f"could not reload remote_folder: {exc}"))
            continue
        try:
            copy_remote_folder_to_local(
                remote,
                entry.local_path,
                transport_factory=transport_factory,
            )
            success += 1
        except CopyError as exc:
            failures.append((entry, str(exc)))
    return success, failures


# ---------------------------------------------------------------------------
# Top-level glue used by the CLI handler
# ---------------------------------------------------------------------------


def load_copy_plan(
    *,
    input_json: Path | str,
    method: str,
    class_to_backend: Mapping[str, str],
    base_dir: Path | str,
) -> CopyPlan:
    """Load copy plan from input JSON file.

    Convenience wrapper that combines
    :func:`collect_pk_map` (parse) → :func:`iter_copy_targets`
    (resolve provenance) → :func:`resolve_copy_targets` (plan paths).

    Errors raised by :func:`iter_copy_targets` (e.g. method mismatch)
    propagate to the caller — the CLI handler turns them into a
    non-zero exit code.
    """
    from aiida_uranium_workflow.cli._common import collect_pk_map

    pk_map = collect_pk_map(input_json)
    targets = list(
        iter_copy_targets(
            pk_map=pk_map,
            class_to_backend=class_to_backend,
        )
    )
    entries, skipped = resolve_copy_targets(targets, base_dir=base_dir)
    return CopyPlan(entries=entries, skipped=skipped)


__all__ = [
    "CopyTarget",
    "CopyPlanEntry",
    "CopyPlan",
    "CopyError",
    "sanitise_path_component",
    "build_local_path",
    "collect_remote_folder_calcjobs",
    "iter_copy_targets",
    "resolve_copy_targets",
    "copy_remote_folder_to_local",
    "execute_copy_plan",
    "load_copy_plan",
    "resolve_label",
    # Re-exported for callers that already imported from copy_calc
    "collect_identifiers_from_json",
]