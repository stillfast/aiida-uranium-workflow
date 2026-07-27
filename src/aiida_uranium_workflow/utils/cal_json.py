"""Helpers for writing the ``final_cal.json``-style output layout.

Layout written (current)::

    {
        "<backend>": {
            "<workflow_key>": {
                "<preset_name>": "<uuid>",
                ...
            },
            ...
        },
        ...
    }

Example for the smear workflow::

    {
        "abacus": {"smear": {"lcao": "8c0f...-uuid", "pw": "33e1...-uuid"}},
        "vasp":   {"vasp":  {"test": "5f4a...-uuid"}}
    }

The leaf identifiers are top-level WorkChain UUID strings (the canonical
AiiDA identifier that survives profile re-imports). Legacy files written
with integer pk leaves are still readable; see
:func:`collect_pks_from_json` / :func:`read_unique_pks` in
``utils/copy_calc.py`` and ``cli/_common.py`` for the parser side.

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
) -> "OrderedDict[str, OrderedDict[str, OrderedDict[str, str | int]]]":
    """Group ``submitted`` jobs into the nested ``final_cal.json`` layout.

    The structure is::

        {backend: {key: {preset_name: identifier, ...}, ...}, ...}

    where ``key`` is the second-level dict name (``"smear"`` for abacus
    smear, ``"vasp"`` for vasp smear, …) and ``identifier`` is the
    WorkChain UUID string (or, for backwards-compat, the integer pk).
    Jobs with the same ``(backend, key, preset_name)`` overwrite one
    another — this is intentional, callers usually only have one
    WorkChain per preset.
    """
    out: "OrderedDict[str, OrderedDict[str, OrderedDict[str, str | int]]]" = (
        OrderedDict()
    )
    mapping = backend_to_key or {}
    for job in submitted:
        backend = job.backend
        key = mapping.get(backend, _default_key(workflow, backend))
        backend_dict = out.setdefault(backend, OrderedDict())
        key_dict = backend_dict.setdefault(key, OrderedDict())
        key_dict[job.preset_name] = _node_identifier(job)
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