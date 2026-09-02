"""Generic orchestrator + workflow registry.

A workflow (smear, relax, band, ...) registers itself through
``register_workflow()`` here. The registry maps the ``workflow`` name
that appears in ``input.json`` to:

* the protocol YAML file (``parameters/protocol/<file>.yml``)
* a parser hook that extracts workflow-specific data
* a concrete :class:`WorkflowOrchestrator` subclass

The generic :meth:`WorkflowOrchestrator.run` handles the boring parts:

1. ``load_profile()``
2. build :class:`StructureData`
3. iterate over backends and ask the matching input_builder to assemble
   the AiiDA ``inputs`` dict
4. submit

Concrete orchestrators only need to set ``ADAPTERS`` / ``BACKENDS``.
"""

from __future__ import annotations

from aiida import load_profile
from aiida_uranium_workflow.utils.common import ParamBundle
from aiida_uranium_workflow.utils.structure import build_structure
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass
class WorkflowEntry:
    """Metadata for one registered workflow."""

    protocol_file: Optional[str] = None
    # Key under input.json["parameters"] that names the protocol to load.
    # Defaults to ``name`` at registration time. Protocol-free workflows
    # leave this unset.
    workflow_key: Optional[str] = None
    parser_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    orchestrator_cls: Optional[Type["WorkflowOrchestrator"]] = None


@dataclass
class SubmittedJob:
    """One submitted WorkChain — returned in submission order by
    :meth:`WorkflowOrchestrator.run`.

    The dataclass keeps ``pk`` for backwards compatibility (older code
    paths / tests pre-date UUID support) while adding ``uuid`` (the
    top-level WorkChain UUID string) as the canonical identifier.  New
    consumers should prefer ``uuid``; older callers can keep reading
    ``pk`` unchanged.
    """

    backend: str
    preset_name: str
    pk: int
    structure_name: str
    #: Top-level WorkChain UUID string. Optional / empty for legacy
    #: test stubs that fabricate ``SubmittedJob`` instances without a
    #: real submit.
    uuid: str = ""


_WORKFLOW_REGISTRY: Dict[str, WorkflowEntry] = {}


def register_workflow(
    name: str,
    *,
    protocol_file: Optional[str] = None,
    workflow_key: Optional[str] = None,
    parser_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    orchestrator_cls: Optional[Type["WorkflowOrchestrator"]] = None,
) -> None:
    """Register a workflow under ``name`` (e.g. ``"smear"``).

    ``workflow_key`` is the key inside ``input.json["parameters"]`` whose
    value names the protocol preset to load. It defaults to ``name`` when
    ``protocol_file`` is set; protocol-free workflows leave it unset.
    """
    _WORKFLOW_REGISTRY[name] = WorkflowEntry(
        protocol_file=protocol_file,
        workflow_key=workflow_key or (name if protocol_file else None),
        parser_hook=parser_hook,
        orchestrator_cls=orchestrator_cls,
    )


def get_workflow_entry(name: str) -> WorkflowEntry:
    if name not in _WORKFLOW_REGISTRY:
        raise ValueError(
            f"Unknown workflow '{name}'. Registered: {list(_WORKFLOW_REGISTRY.keys())}"
        )
    return _WORKFLOW_REGISTRY[name]


def get_orchestrator(
    bundle: ParamBundle,
    backends: Optional[Tuple[str, ...]] = None,
) -> "WorkflowOrchestrator":
    """Factory: return the orchestrator that owns ``bundle.input_params['workflow']``."""
    entry = get_workflow_entry(bundle.input_params["workflow"])
    if entry.orchestrator_cls is None:
        raise RuntimeError(
            f"Workflow '{bundle.input_params['workflow']}' has no orchestrator registered"
        )
    return entry.orchestrator_cls(bundle, backends=backends)


# ---------------------------------------------------------------------------
# Base orchestrator
# ---------------------------------------------------------------------------


class WorkflowOrchestrator:
    """Generic loop: for each backend, build inputs and submit."""

    #: Backend name -> input_builder class (set by subclass)
    ADAPTERS: Dict[str, Type] = {}

    #: Default backend names to run (set by subclass)
    BACKENDS: Tuple[str, ...] = ()

    #: Key under ``input.json["parameters"][<backend>]`` that holds the
    #: list of preset names for this workflow. When unset, the orchestrator
    #: looks at ``parameters[<backend>]`` directly (str/list form).
    #: Smear / magmom / convergence override :attr:`PRESET_SUBKEYS` so
    #: the same logic can be applied to backends with different
    #: sub-key names (``"smear"`` / ``"magmom"`` / ``"convergence"``).
    PRESET_SUBKEY: Optional[str] = None

    #: Per-backend preset-subkey mapping. When a backend has its own
    #: sub-key in ``parameters[<backend>]`` (e.g. ``"abacus": {"smear":
    #: [...]}`` or ``"abacus": {"convergence": [...]}``), subclasses set
    #: this so :meth:`_preset_names_for` can look up the right key per
    #: backend. Falls back to :attr:`PRESET_SUBKEY` (single-backend key)
    #: when no backend-specific entry is present.
    PRESET_SUBKEYS: Dict[str, str] = {}

    def __init__(
        self,
        bundle: ParamBundle,
        backends: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.bundle = bundle
        self.backends = tuple(backends) if backends else self.BACKENDS
        # Populated by ``run_with_jobs``; cleared at the start of every run.
        # ``run()`` delegates to ``run_with_jobs`` so this list mirrors the
        # pk list returned to legacy callers while also carrying UUIDs.
        self._submitted_jobs: list[SubmittedJob] = []

    @property
    def last_submitted(self) -> List[SubmittedJob]:
        """Return :class:`SubmittedJob` records from the most recent run.

        Empty before :meth:`run` (or :meth:`run_with_jobs`) is called.
        Callers should not mutate the returned list.
        """
        return list(self._submitted_jobs)

    # ---- public API ------------------------------------------------------

    def run(self) -> List[int]:
        """Run the orchestrator and return the list of submitted pks.

        For minimal compatibility with existing callers / tests, this
        method returns the integer pk of each submission (same as the
        legacy interface). Callers that need the full
        :class:`SubmittedJob` — including the canonical UUID — can use
        :meth:`run_with_jobs` or read :attr:`last_submitted` after this
        method returns.
        """
        jobs = self.run_with_jobs()
        return [job.pk for job in jobs]

    def run_with_jobs(self) -> List[SubmittedJob]:
        """Run the orchestrator and return :class:`SubmittedJob` records.

        Each entry carries both the integer pk (for log readability)
        and the WorkChain UUID string (the canonical identifier written
        to ``output.json``). When the submit returned a node without a
        ``uuid`` attribute (e.g. test stubs), ``uuid`` is left empty —
        callers fall back to the pk in :func:`_node_identifier`.
        """
        self._submitted_jobs = []
        load_profile(self.bundle.input_params["profile"])
        structures = self._build_structure()
        structure_names = self._structure_names()
        n_structures = len(structures)
        submitted: List[SubmittedJob] = []
        # ``workflow_presets`` lets ``parameters[<workflow_key>]`` be a
        # list of protocol preset names — one WorkChain per preset, each
        # with its own ``workflow_data``. Empty ⇒ single-preset behaviour.
        wf_presets = self.bundle.workflow_presets or [None]
        for backend, builder_cls, preset_idx, preset_name in self._select_backends():
            n_presets = len(self.bundle.software_params.get(backend, []))
            for wf_preset in wf_presets:
                for structure_idx, (structure, structure_name) in enumerate(
                    zip(structures, structure_names)
                ):
                    node = self._submit_one(
                        backend, builder_cls, structure, preset_idx,
                        workflow_preset=wf_preset,
                    )
                    pk = node.pk
                    # ``uuid`` is the canonical identifier of the WorkChain
                    # for downstream consumers (``output.json`` writes it
                    # instead of the local pk); ``load_node(uuid)`` round-trips.
                    uuid = str(getattr(node, "uuid", "")) if node is not None else ""
                    # Combine the SCF preset and the protocol preset into a
                    # unique output.json key when both vary (e.g.
                    # "lcao/vacancy_scf").
                    final_name = (
                        f"{preset_name}/{wf_preset}" if wf_preset else preset_name
                    )
                    job = SubmittedJob(
                        backend=backend,
                        preset_name=final_name,
                        pk=pk,
                        structure_name=structure_name,
                        uuid=uuid,
                    )
                    submitted.append(job)
                    self._submitted_jobs.append(job)
                    label = self._label(
                        backend, preset_idx, n_presets,
                        structure_idx, n_structures,
                        preset_name=final_name,
                    )
                    print(
                        f"[{self.bundle.input_params['workflow']}] {label} submitted, "
                        f"pk={pk} uuid={uuid}"
                    )
        return submitted

    # ---- internals (shared by all workflows) ----------------------------

    def _structure_names(self) -> List[str]:
        """Return one human-readable name per requested structure.

        Mirrors :meth:`_build_structure`'s str-or-list parsing. Used as
        the value of :attr:`SubmittedJob.structure_name` so the CLI can
        record per-preset structure information in the output JSON.
        """
        raw = self.bundle.input_params["static"]["structure"]
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ValueError("'static.structure' list must be non-empty")
            if not all(isinstance(s, str) for s in raw):
                raise TypeError(
                    "'static.structure' must be a string or a list of strings, "
                    f"got {raw!r}"
                )
            return list(raw)
        raise TypeError(
            "'static.structure' must be a string or a list of strings, "
            f"got type {type(raw).__name__}"
        )

    def _build_structure(self) -> List[Any]:
        """Build every structure requested by ``input.json['static']['structure']``.

        The user may pass either a single name (``"bcc-uranium"``) or a
        list of names (``["bcc-uranium", "bcc-uranium-0K"]``). In both
        cases a list of AiiDA ``StructureData`` objects is returned — one
        per requested structure — so the orchestrator can submit one
        WorkChain per (backend, preset, structure) combination.
        """
        from aiida import orm

        return [
            orm.StructureData(ase=build_structure(name))
            for name in self._structure_names()
        ]

    @staticmethod
    def _label(
        backend: str,
        preset_idx: int,
        n_presets: int,
        structure_idx: int,
        n_structures: int,
        preset_name: str = "",
    ) -> str:
        """Compose a human-readable label for one submission.

        ``preset_name`` is optional and reserved for future use; the
        legacy positional signature ``(backend, preset_idx, n_presets,
        structure_idx, n_structures)`` is kept so existing callers /
        tests don't have to be updated.
        """
        parts = [backend]
        if n_presets > 1:
            if preset_name:
                parts.append(f"/{preset_name}")
            else:
                parts.append(f"#{preset_idx}")
        if n_structures > 1:
            parts.append(f"@stru{structure_idx}")
        return "".join(parts)

    def _preset_names_for(self, backend: str) -> List[str]:
        """Resolve the user-supplied preset name list for ``backend``.

        Supports the three layouts seen across workflows:

        1. ``"abacus": "test"``                     → ``["test"]``
        2. ``"abacus": ["test", "test_soc"]``       → ``["test", "test_soc"]``
        3. ``"abacus": {"smear": ["lcao", ...]}``   → uses ``PRESET_SUBKEYS``
           / ``PRESET_SUBKEY`` (smear sets it to ``"smear"``; magmom /
           convergence use ``"magmom"`` / ``"convergence"``).

        When nothing matches we fall back to synthetic ``f"{backend}#{i}"``
        names so callers still get a stable per-preset identifier.
        """
        params = self.bundle.input_params.get("parameters", {}).get(backend)
        n_presets = len(self.bundle.software_params.get(backend, []))
        if isinstance(params, str):
            return [params]
        if isinstance(params, (list, tuple)) and all(
            isinstance(p, str) for p in params
        ):
            return list(params)
        subkey = self.PRESET_SUBKEYS.get(backend, self.PRESET_SUBKEY)
        if isinstance(params, dict) and subkey:
            sub = params.get(subkey)
            if isinstance(sub, str):
                return [sub]
            if isinstance(sub, (list, tuple)) and all(
                isinstance(s, str) for s in sub
            ):
                return list(sub)
        return [f"{backend}#{i}" for i in range(n_presets)]

    def _select_backends(
        self,
    ) -> List[Tuple[str, Type, int, str]]:
        """Return ``(backend, builder_cls, preset_idx, preset_name)`` for every run.

        If a backend was requested multiple times in ``input.json`` (e.g.
        ``"abacus": ["test", "test_soc"]``), one entry per preset is yielded
        in the order given.
        """
        chosen: List[Tuple[str, Type, int, str]] = []
        requested = self.bundle.input_params.get("parameters", {})
        for backend in self.backends:
            if not requested.get(backend):
                continue
            if backend not in self.ADAPTERS:
                raise KeyError(f"No input_builder for backend '{backend}'")
            presets = self.bundle.software_params.get(backend)
            if not presets:
                continue
            preset_names = self._preset_names_for(backend)
            for idx, _preset in enumerate(presets):
                name = (
                    preset_names[idx]
                    if idx < len(preset_names)
                    else f"{backend}#{idx}"
                )
                chosen.append((backend, self.ADAPTERS[backend], idx, name))
        if not chosen:
            raise RuntimeError("No backend requested in input.json")
        return chosen

    def _submit_one(
        self,
        backend: str,
        cls: Type,
        structure,
        preset_idx: int = 0,
        workflow_preset: str | None = None,
    ):
        """Submit one WorkChain and return its process node.

        Returning the full node (rather than just ``.pk``) lets the
        caller capture both the integer pk (for log readability) and
        the top-level uuid (the canonical identifier written into
        ``output.json``).

        ``workflow_preset`` selects the per-preset ``workflow_data`` when
        ``parameters[<workflow_key>]`` was given as a list (one WorkChain
        per protocol preset); ``None`` uses the default single preset.
        """
        from aiida.engine import submit

        preset = self.bundle.software_params[backend][preset_idx]
        workflow_data = (
            self.bundle.workflow_data_map.get(workflow_preset, self.bundle.workflow_data)
            if workflow_preset
            else self.bundle.workflow_data
        )
        builder = cls(
            code_label=self.bundle.input_params["code"][backend],
            software_params=preset,
            metadata=self.bundle.metadata,
            workflow_data=workflow_data,
            # Pass the full ``input.json["code"]`` mapping so adapters
            # that need sibling codes (e.g. FLEUR's ``inpgen`` inside
            # the SCF namespace) can pull them from
            # ``self.extra_codes`` without us hard-coding backend-specific
            # knowledge here.
            extra_codes=dict(self.bundle.input_params.get("code", {})),
        )
        adapted = builder.adapt(structure)
        return submit(adapted.workchain_cls, **adapted.inputs)
