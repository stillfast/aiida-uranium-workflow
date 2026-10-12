"""Abstract base for input builders (one per DFT backend).

An input builder takes the canonical config (a ParamBundle) and turns
it into the AiiDA ``inputs`` dict for the matching WorkChain.

Subclasses only need to implement:
* :meth:`_workchain_entry_point` — AiiDA entry-point string
* :meth:`_build_workchain_inputs` — backend-specific assembly
* (optional) :meth:`_prepare_workflow_inputs` — workflow-specific fields
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

#: Cache for :func:`scheduler_option_keys`, keyed by
#: ``(workchain entry point, backend)``.
_SCHEDULER_OPTION_KEYS: Dict[tuple, Optional[FrozenSet[str]]] = {}

#: The CalcJob that ultimately receives the scheduler options, per backend.
#: Its ``metadata.options`` namespace is the authority when the submitted
#: WorkChain hides that namespace inside a dynamic port (the supercell /
#: defects / eos / phonopy wrappers expose ``base`` only).
_BACKEND_CALCJOB: Dict[str, str] = {
    "abacus": "abacus.abacus",
    "vasp": "vasp.v2.vasp",
    "fleur": "fleur.fleur",
    "qe": "quantumespresso.pw",
}


@dataclass
class AdaptedInputs:
    """A WorkChain class plus the kwargs it needs."""

    workchain_cls: type
    inputs: dict[str, Any]


def _collect_metadata_options(namespace, found: set) -> None:
    """Collect the keys of every ``…metadata.options`` namespace in a spec."""
    from aiida.engine.processes.ports import PortNamespace

    for name, port in namespace.items():
        if not isinstance(port, PortNamespace):
            continue
        if name == "metadata":
            options = port.get("options")
            if isinstance(options, PortNamespace):
                found.update(options.keys())
        _collect_metadata_options(port, found)


def scheduler_option_keys(
    entry_point: Optional[str] = None, backend: Optional[str] = None
) -> Optional[FrozenSet[str]]:
    """Keys the ``metadata.options`` namespace of the target CalcJob accepts.

    Resolved **per backend**, because the set differs between plugins and
    the namespace is not dynamic:

    * ABACUS / VASP / QE expose the 25 (QE: 26) standard scheduler options;
      ``optimize_resources`` — a FLEUR *workchain* option, not a CalcJob
      one — is not among them;
    * FLEUR's ``fleur.fleur`` CalcJob has 26 keys, also without it.  FLEUR
      workchains receive the option through their plugin ``options``
      ``Dict`` input instead (``aiida_fleur`` pops it in
      ``base_fleur.py``), which the FLEUR adapters pass through untouched.

    Lookup order: the submitted WorkChain's own ``…metadata.options``
    namespace(s); then, when the WorkChain hides it inside a dynamic port
    (supercell / defects / eos / phonopy expose only ``base``), the CalcJob
    of ``backend``; finally the generic aiida-core ``CalcJob`` namespace.
    """
    cache_key = (entry_point, backend)
    if cache_key in _SCHEDULER_OPTION_KEYS:
        return _SCHEDULER_OPTION_KEYS[cache_key]

    keys: Optional[FrozenSet[str]] = None
    if entry_point:
        try:
            from aiida.plugins import WorkflowFactory

            found: set = set()
            # The entry point is a WorkChain in most cases, but the backend
            # fallback below hands over a CalcJob — accept either.
            from aiida.plugins import CalculationFactory, WorkflowFactory

            for factory in (WorkflowFactory, CalculationFactory):
                try:
                    node_class = factory(entry_point)
                except Exception:  # noqa: BLE001 — wrong factory for this name
                    continue
                try:
                    _collect_metadata_options(node_class.spec().inputs, found)
                except Exception:  # noqa: BLE001 — no metadata namespace at all
                    continue
                if found:
                    break
            keys = frozenset(found) if found else None
        except Exception:  # noqa: BLE001 — unknown entry point: keep looking
            keys = None

    if keys is None and backend:
        keys = scheduler_option_keys(_BACKEND_CALCJOB.get(backend), None)

    if keys is None and not entry_point:
        from aiida.engine.processes.calcjobs.calcjob import CalcJob

        keys = frozenset(CalcJob.spec().inputs["metadata"]["options"].keys())

    _SCHEDULER_OPTION_KEYS[cache_key] = keys
    return keys


def filter_scheduler_options(
    options: Optional[dict], *, entry_point: Optional[str] = None, backend: Optional[str] = None
) -> dict:
    """Keep only the options the target backend's CalcJob accepts.

    ``static/metadata.yml`` presets are shared across backends, so the
    ``yunhe`` preset carries ``optimize_resources`` (a FLEUR workchain
    option).  Forwarding it into a CalcJob ``metadata.options`` namespace
    fails port validation::

        ValueError: Error occurred validating port
        'inputs.base.abacus.metadata.options': Unexpected ports
        {'optimize_resources': False}, for a non dynamic namespace

    so keys the target does not understand are dropped — deliberately
    silently and per backend: a shared preset is *expected* to hold keys
    for several plugins, each backend keeps what it understands, and the
    FLEUR adapters (which hand the whole dict to FLEUR's own ``options``
    port, where ``optimize_resources`` is meaningful) never come here.
    """
    if not options:
        return {}
    allowed = scheduler_option_keys(entry_point, backend)
    if allowed is None:  # target takes the raw dict — keep everything
        return dict(options)
    return {key: value for key, value in options.items() if key in allowed}


def resolve_extra_files(spec) -> dict[str, Any]:
    """Resolve a preset ``extra_files`` mapping to Data nodes.

    ``spec`` is ``{label: pk_or_uuid}`` or ``{label: {pk: ...}}`` /
    ``{label: {uuid: ...}}``. Each referenced node must be a
    ``SinglefileData`` or ``FolderData`` (what aiida-abacus's
    ``abacus.extra_files`` accepts); the file is copied into the
    calculation folder at ``<extra_files_basepath>/<node.filename>``.
    """
    from aiida.orm import FolderData, SinglefileData, load_node

    if not spec:
        return {}
    out: dict[str, Any] = {}
    for label, ref in dict(spec).items():
        if isinstance(ref, dict):
            ident = ref.get("uuid") or ref.get("pk")
            if ident is None:
                raise ValueError(
                    f"extra_files['{label}'] dict must contain 'pk' or "
                    f"'uuid', got {ref!r}"
                )
        else:
            ident = ref
        node = load_node(ident)
        if not isinstance(node, (SinglefileData, FolderData)):
            raise TypeError(
                f"extra_files['{label}'] must reference a SinglefileData "
                f"or FolderData node, got {type(node).__name__} "
                f"(pk={node.pk}). Store the file with "
                "`SinglefileData(file=...).store()` first."
            )
        out[label] = node
    return out


class SoftwareAdapter(ABC):
    """Strategy pattern: one concrete subclass per DFT backend."""

    name: str = ""

    def __init__(
        self,
        code_label: str,
        software_params: dict[str, Any],
        metadata: dict[str, Any],
        workflow_data: Dict[str, Any],
        *,
        extra_codes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.code_label = code_label
        self.software_params = software_params
        self.metadata = metadata
        self.workflow_data = workflow_data
        # ``extra_codes`` carries any sibling AiiDA code labels the
        # adapter may need to forward into nested namespaces (e.g.
        # FLEUR's ``inpgen`` inside the SCF namespace). Adapters that
        # don't need extras simply ignore this dict. The orchestrator
        # passes the entire ``input.json["code"]`` dict so the adapter
        # can ``extra_codes.get("inpgen")`` etc. without the caller
        # having to know which codes are relevant.
        self.extra_codes: Dict[str, str] = dict(extra_codes or {})

    # ---- subclasses must implement ---------------------------------------

    @abstractmethod
    def _workchain_entry_point(self) -> str:
        """AiiDA entry-point string of the WorkChain to submit."""

    @abstractmethod
    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        """Backend-specific input assembly."""

    # ---- optional overrides ----------------------------------------------

    def _prepare_workflow_inputs(self) -> Tuple[List, List[float]]:
        """Workflow-specific (smear_kws, sigma_vals) extraction. Default: empty."""
        return [], []

    # ---- public surface used by the orchestrator -------------------------

    def adapt(self, structure) -> AdaptedInputs:
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        extra_kws, extra_vals = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        for key, vals in zip(("smear", "sigma"), (extra_kws, extra_vals)):
            if vals:
                inputs.setdefault(key, orm.List(list=vals))

        self._inject_options(inputs, options)

        return AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    # ---- internal helpers -------------------------------------------------

    def _inject_options(self, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        """Place scheduler options under the calculation namespace.

        Mirrors the shape used by ``test_abacus_base.py``: scheduler
        options live at ``inputs.<backend>.metadata.options``.  The
        backend namespace (``abacus`` / ``calc``) is exposed by the
        child base workchain via ``expose_inputs(...)`` and accepts a
        nested ``metadata.options`` because that namespace is *not*
        marked non-exposable the same way the workchain-level
        ``metadata`` port is.

        Foreign keys (e.g. ``optimize_resources`` from the ``yunhe``
        preset) are dropped here — the namespace is not dynamic and the
        submission would otherwise fail port validation.
        """
        options = filter_scheduler_options(
            options,
            entry_point=self._workchain_entry_point(),
            backend=self.name,
        )
        if not options:
            return
        if "abacus" in inputs and isinstance(inputs["abacus"], dict):
            inputs["abacus"].setdefault("metadata", {})["options"] = options
        elif "calc" in inputs and isinstance(inputs["calc"], dict):
            inputs["calc"].setdefault("metadata", {})["options"] = options
