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
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AdaptedInputs:
    """A WorkChain class plus the kwargs it needs."""

    workchain_cls: type
    inputs: dict[str, Any]


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
        """
        if not options:
            return
        if "abacus" in inputs and isinstance(inputs["abacus"], dict):
            inputs["abacus"].setdefault("metadata", {})["options"] = options
        elif "calc" in inputs and isinstance(inputs["calc"], dict):
            inputs["calc"].setdefault("metadata", {})["options"] = options
