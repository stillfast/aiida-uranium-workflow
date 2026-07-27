"""Walk an ``output.json``-shaped nested mapping and collect node identifiers.

These helpers centralise the small amount of JSON-trawling logic that
``cli/_common.py`` and ``utils/copy_remote.py`` need to interpret
``output.json`` files (which may contain integer pks, UUID strings, or
both, depending on when the file was written).

Migrated verbatim from the legacy ``utils/copy_calc.py`` so the new
``json_collect`` module owns what is actually used at runtime; the
remaining copy-script helpers in ``copy_calc`` are scheduled for
removal.
"""

from __future__ import annotations

from typing import Any


def collect_pks_from_json(data: Any) -> list[int]:
    """Walk the nested JSON structure and collect all integer pk values.

    Retained for backwards compatibility — older ``output.json`` files
    store integer pks in the leaves. New files (since the UUID switch)
    store UUID strings; use :func:`collect_node_ids_from_json` to walk
    those, or :func:`collect_identifiers_from_json` to handle both
    shapes transparently.
    """
    pks: list[int] = []
    if isinstance(data, dict):
        for value in data.values():
            pks.extend(collect_pks_from_json(value))
    elif isinstance(data, list):
        for item in data:
            pks.extend(collect_pks_from_json(item))
    elif isinstance(data, int) and not isinstance(data, bool):
        pks.append(data)
    return pks


def _looks_like_uuid(value: str) -> bool:
    """Heuristic UUID check — accept only the canonical 8-4-4-4-12 form.

    AiiDA UUIDs are always 36-char hyphenated hex strings, so a cheap
    structural test is enough. Anything that doesn't match (e.g. an
    arbitrary preset name) is left untouched by the identifier
    collectors.
    """
    if not isinstance(value, str) or len(value) != 36:
        return False
    if value[8] != "-" or value[13] != "-" or value[18] != "-" or value[23] != "-":
        return False
    body = value.replace("-", "")
    return len(body) == 32 and all(c in "0123456789abcdefABCDEF" for c in body)


def collect_node_ids_from_json(data: Any) -> list[str]:
    """Walk the JSON and collect UUID-string leaves.

    Counterpart to :func:`collect_pks_from_json` for the modern
    ``output.json`` layout (UUID strings in the leaves). Non-UUID
    string leaves (e.g. preset names sitting one level deeper than
    expected) are silently skipped.
    """
    ids: list[str] = []
    if isinstance(data, dict):
        for value in data.values():
            ids.extend(collect_node_ids_from_json(value))
    elif isinstance(data, list):
        for item in data:
            ids.extend(collect_node_ids_from_json(item))
    elif isinstance(data, str) and _looks_like_uuid(data):
        ids.append(data)
    return ids


def collect_identifiers_from_json(data: Any) -> list[str]:
    """Walk the JSON and collect **both** integer pk and UUID-string leaves.

    Returns every leaf as a string (``str(pk)`` or the UUID as-is). Use
    this when you want a single helper that handles both old pk-based
    and new UUID-based ``output.json`` files without loss.

    ``load_node(...)`` in AiiDA accepts both integer pks and UUID
    strings, so the returned identifiers can be passed straight in.
    """
    out: list[str] = []
    if isinstance(data, dict):
        for value in data.values():
            out.extend(collect_identifiers_from_json(value))
    elif isinstance(data, list):
        for item in data:
            out.extend(collect_identifiers_from_json(item))
    elif isinstance(data, bool):
        # ``bool`` is a subclass of ``int`` in Python — guard explicitly
        # so ``True`` / ``False`` aren't mistakenly treated as pks.
        return out
    elif isinstance(data, int):
        out.append(str(data))
    elif isinstance(data, str) and _looks_like_uuid(data):
        out.append(data)
    return out
