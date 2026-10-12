"""Helpers for writing the ``final_cal.json``-style output layout.

Layout written (current)::

    {
        "workflow": "<method_name>",
        "<backend>": {
            "<workflow_key>": {
                "<preset_name>": "<uuid>",
                ...
            },
            ...
        },
        ...
    }

The top-level ``"workflow"`` key carries the canonical method name
(``"smear"`` / ``"convergence"`` / ``"magmom"`` / ``"base"``) so the
unified CLI can later re-derive the method without asking the user
again. Legacy files written before this field existed remain readable:
the parser falls back to scanning the second-level keys.

Example for the smear workflow::

    {
        "workflow": "smear",
        "abacus": {"smear": {"lcao": "8c0f...-uuid", "pw": "33e1...-uuid"}},
        "vasp":   {"vasp":  {"test": "5f4a...-uuid"}}
    }

The leaf identifiers are top-level WorkChain UUID strings (the canonical
AiiDA identifier that survives profile re-imports). Legacy files written
with integer pk leaves are still readable; see
:func:`collect_pks_from_json` / :func:`collect_identifiers_from_json` in
``utils/json_collect.py`` and ``cli/_common.py`` for the parser side.

The mapping ``backend -> workflow_key`` defaults to the smear layout
(abacus→smear, vasp→vasp) but can be customised per workflow.
"""

from __future__ import annotations

from aiida_uranium_workflow.schedulers import SubmittedJob
from collections import OrderedDict
from typing import Iterable, Mapping


# Default per-workflow key per backend. Magmom / convergence may want
# different mappings; they can override ``build_cal_json``.
DEFAULT_BACKEND_TO_KEY: dict[str, dict[str, str]] = {
    "base": {"abacus": "abacus", "vasp": "vasp"},
    "smear": {"abacus": "smear", "vasp": "vasp"},
    "magmom": {"abacus": "magmom", "vasp": "magmom"},
    "convergence": {"abacus": "convergence", "vasp": "convergence"},
    "banddos": {"abacus": "scf", "fleur": "scf"},
    "relax": {"abacus": "scf", "fleur": "scf"},
}


def _default_key(workflow: str, backend: str) -> str:
    """Resolve the inner JSON key for ``(workflow, backend)``.

    Falls back to ``workflow`` itself if the workflow is unknown — this
    keeps the file readable even when a new workflow hasn't registered
    its mapping yet.
    """
    return DEFAULT_BACKEND_TO_KEY.get(workflow, {}).get(backend, workflow)


def _node_identifier(job: SubmittedJob) -> str | int:
    """Return the canonical identifier to record for ``job``.

    Prefers ``job.uuid`` when the submit captured one (the modern path);
    falls back to ``job.pk`` for legacy callers / test stubs that didn't
    record a UUID. We always emit strings (UUIDs are strings; pks are
    coerced via ``str(...)``) so the resulting JSON shape stays
    uniform regardless of which path was taken.
    """
    if getattr(job, "uuid", None):
        return job.uuid
    return job.pk


def build_cal_json(
    submitted: Iterable[SubmittedJob],
    *,
    workflow: str,
    backend_to_key: Mapping[str, str] | None = None,
) -> "OrderedDict[str, Any]":
    """Group ``submitted`` jobs into the nested ``final_cal.json`` layout.

    The structure is::

        {
          "workflow": "<name>",
          <backend>: {<key>: {<preset_name>: <identifier>, ...}, ...},
          ...
        }

    where ``key`` is the second-level dict name (``"smear"`` for abacus
    smear, ``"vasp"`` for vasp smear, …) and ``identifier`` is the
    WorkChain UUID string (or, for backwards-compat, the integer pk).

    When one preset is submitted for **several structures** (a
    ``static.structure`` list, e.g. a twist-angle / supercell-size scan)
    the leaf key is ``"<preset>@<structure>"`` so that each structure
    keeps its own identifier. Without the suffix every later structure
    would overwrite the previous one — a 12-structure scan would record
    a single UUID. The plain ``<preset>`` key is unchanged whenever a
    preset/structure pair is unique, so single-structure output.json
    files are byte-for-byte the same as before.
    """
    from collections import Counter

    jobs = list(submitted)
    # ``(backend, key, preset_name)`` appearing more than once ⇒ the
    # preset was run for multiple structures → disambiguate by structure.
    seen = Counter(
        (
            job.backend,
            (backend_to_key or {}).get(job.backend)
            or _default_key(workflow, job.backend),
            job.preset_name,
        )
        for job in jobs
    )

    out: "OrderedDict[str, Any]" = OrderedDict()
    out["workflow"] = workflow
    mapping = backend_to_key or {}
    for job in jobs:
        backend = job.backend
        key = mapping.get(backend, _default_key(workflow, backend))
        backend_dict = out.setdefault(backend, OrderedDict())
        key_dict = backend_dict.setdefault(key, OrderedDict())
        name = job.preset_name
        structure = getattr(job, "structure_name", "") or ""
        if seen[(backend, key, job.preset_name)] > 1 and structure:
            name = f"{name}@{structure}"
        key_dict[name] = _node_identifier(job)
    return out


def write_cal_json(
    submitted: Iterable[SubmittedJob],
    *,
    output_path,
    workflow: str,
    backend_to_key: Mapping[str, str] | None = None,
) -> None:
    """Convenience: build the dict and write it as pretty JSON."""
    import json

    data = build_cal_json(
        submitted, workflow=workflow, backend_to_key=backend_to_key
    )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")