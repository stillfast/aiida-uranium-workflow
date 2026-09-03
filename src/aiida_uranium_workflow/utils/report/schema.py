"""Report-side data schema: what a WorkChain's ``output_parameters``
must look like for the report layer to consume it.

Every WorkChain currently hand-writes its own ``parse_and_gather_*``
calcfunction whose output dict layout is only documented in comments;
the report modules then re-discover that layout with ``dict.get``
chains and per-backend branches.  This module turns the *contract* into
code:

* :class:`ChildRecord` — one normalised per-child row (pk / label /
  exit status / energy eV / time s / SCF steps / atom count).
* :class:`GatherResult` — the schema a parent WorkChain writes as its
  ``output_parameters``: a schema version, the backend, and one
  :class:`ChildRecord` per child, in submission order.

The workflow side (``utils/parsers/child.py``) produces
:class:`ChildRecord` values; the workflow's gather calcfunction
collects them into a :class:`GatherResult` and stores
``result.to_dict()``.  The report side parses it back with
:func:`GatherResult.from_output_params` and renders — no magic keys, no
per-backend branches.

Backward compatibility
----------------------
:func:`GatherResult.from_output_params` returns ``None`` when the dict
does not carry the schema marker (legacy layout produced by older
gather functions).  Callers then fall back to the legacy render path
and emit a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Marker key + current version written into every new gather result.
SCHEMA_KEY = "gather_schema"
SCHEMA_VERSION = 1


@dataclass
class ChildRecord:
    """One child's normalised result (backend-agnostic core fields)."""

    pk: int
    status: Optional[int] = None  # AiiDA exit status; None = unfinished
    finished_ok: bool = False
    energy_ev: Optional[float] = None
    time_s: Optional[float] = None
    scf_steps: Optional[int] = None
    natoms: Optional[int] = None
    #: Backend-specific magnetism etc. (shape defined by the child
    #: parser; the report module interprets it per backend).
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pk": self.pk,
            "status": self.status,
            "finished_ok": self.finished_ok,
            "energy_ev": self.energy_ev,
            "time_s": self.time_s,
            "scf_steps": self.scf_steps,
            "natoms": self.natoms,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ChildRecord":
        return cls(
            pk=int(raw["pk"]),
            status=raw.get("status"),
            finished_ok=bool(raw.get("finished_ok", False)),
            energy_ev=raw.get("energy_ev"),
            time_s=raw.get("time_s"),
            scf_steps=raw.get("scf_steps"),
            natoms=raw.get("natoms"),
            data=dict(raw.get("data") or {}),
        )


@dataclass
class GatherResult:
    """Schema of a workflow's ``output_parameters`` (new layout).

    ``children`` are in submission order (index ``i`` corresponds to the
    ``i``-th child the parent submitted).  ``meta`` carries workflow /
    run level context the report needs (e.g. the original
    ``magmom_list`` for the matrix table's "initial magnetisation"
    columns) — free-form, workflow-specific.
    """

    backend: str
    children: List[ChildRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            SCHEMA_KEY: SCHEMA_VERSION,
            "backend": self.backend,
            "children": [child.to_dict() for child in self.children],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_output_params(
        cls, output_params: Dict[str, Any]
    ) -> Optional["GatherResult"]:
        """Parse ``output_params``; ``None`` when it is not new-schema.

        Legacy dicts (produced by the pre-schema gather functions) have
        no ``gather_schema`` key and return ``None`` — the caller falls
        back to the legacy rendering path.
        """
        if not isinstance(output_params, dict):
            return None
        if output_params.get(SCHEMA_KEY) != SCHEMA_VERSION:
            return None
        return cls(
            backend=str(output_params.get("backend", "")),
            children=[
                ChildRecord.from_dict(item)
                for item in output_params.get("children", [])
            ],
            meta=dict(output_params.get("meta") or {}),
        )


def schema_children_map(
    result: GatherResult,
) -> Dict[int, ChildRecord]:
    """Index a :class:`GatherResult`'s children by pk.

    Rendering helpers that historically consumed ``{pk: value}`` dicts
    (energy / time / steps / status tables) can keep that shape by
    looking values up through this map — one child per pk.
    """
    return {child.pk: child for child in result.children}
