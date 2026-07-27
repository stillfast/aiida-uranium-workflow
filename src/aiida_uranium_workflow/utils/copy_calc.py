"""Helpers for selecting child calc pks from a smear WorkChain and
copying their remote_folder to a local target.

Used by the legacy ``aiida-uranium-scripts/smear_test/copy_calculations.py``
script, which is kept as a thin wrapper around this module.

The selection logic intentionally mirrors what the legacy script does so
behaviour does not change when migrating to the CLI.
"""

from __future__ import annotations

from aiida.orm import load_node
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _resolve_smear_pk(smear_pk) -> list[int]:
    """Coerce ``smear_pk`` (int / list / tuple) into a flat list of pks."""
    if isinstance(smear_pk, int):
        return [smear_pk]
    if isinstance(smear_pk, (list, tuple)):
        return list(smear_pk)
    raise TypeError(
        f"smear_pk must be an int or list of ints, got {type(smear_pk).__name__}"
    )


def _snapshot_children(
    pk_list: list[int], software: str
) -> list[tuple[Any, Any, int]]:
    """Return one ``(smear, sigma, child_pk)`` tuple per finished child calc.

    Centralises the lazy-load of ``base.inputs.abacus.parameters`` /
    ``base.inputs.parameters`` so callers can compare tuples without
    re-triggering SQLAlchemy.
    """
    bases = []
    for pk in pk_list:
        bases.extend(list(load_node(pk).called))

    snapshots: list[tuple[Any, Any, int]] = []
    for base in bases:
        try:
            if software == "abacus":
                if "abacus" not in base.inputs:
                    continue
                parameters = base.inputs.abacus.parameters.get_dict()
                input_block = parameters.get("input", {})
                smear = input_block.get("smearing_method")
                sigma = input_block.get("smearing_sigma")
            else:
                # VASP: skip calcfunctions, read INCAR.
                if "parameters" not in base.inputs:
                    continue
                parameters = base.inputs.parameters.get_dict()
                incar = parameters.get("incar", {})
                smear = incar.get("ismear")
                sigma = incar.get("sigma")
            if smear is None or sigma is None:
                continue
            snapshots.append((smear, sigma, base.pk))
        except Exception:
            # Skip calcfunctions / un-stored / partial inputs without
            # aborting the whole pass.
            continue
    return snapshots


def discover_actual_calcs(
    smear_pk,
    software: str,
    *,
    wc_labels: dict[int, str] | None = None,
) -> list[dict]:
    """Return the (label, method, sigma, pk) actually computed by ``smear_pk``.

    Walks the children of every WorkChain in ``smear_pk`` and emits one
    entry per child calcjob (no deduplication — each WorkChain's
    children are physically distinct even when the (method, sigma)
    combination is identical, e.g. ``lcao`` vs ``pw`` runs both produce
    ``mp × 0.06`` but with different pseudo/basis).

    Args:
        smear_pk: PK (or list of PKs) of the smear workflow(s).
        software: ``"abacus"`` or ``"vasp"``.
        wc_labels: optional ``{wc_pk: preset_name}`` mapping (e.g.
            ``{328349: "lcao", 328355: "lcao_soc", ...}``). Each entry
            returned by this function carries a ``label`` key matching
            the WorkChain it came from. When a wc_pk is missing from
            the mapping, ``"wc_<pk>"`` is used as fallback.

    Returns:
        List of dicts, each with keys ``label``, ``smearing_method`` /
        ``ismear``, ``smearing_sigma`` / ``sigma``, ``pk`` (the child
        calcjob pk), and ``wc_pk`` (the parent WorkChain pk).
    """
    pk_list = _resolve_smear_pk(smear_pk)
    wc_labels = wc_labels or {}
    results: list[dict] = []

    # Iterate per WorkChain so we can attach the right label.
    for wc_pk in pk_list:
        snapshots = _snapshot_children([wc_pk], software)
        label = wc_labels.get(wc_pk, f"wc_{wc_pk}")
        for smear, sigma, base_pk in snapshots:
            try:
                if software == "abacus":
                    results.append(
                        {
                            "label": label,
                            "wc_pk": wc_pk,
                            "smearing_method": str(smear),
                            "smearing_sigma": float(sigma),
                            "pk": base_pk,
                        }
                    )
                else:
                    results.append(
                        {
                            "label": label,
                            "wc_pk": wc_pk,
                            "ismear": int(smear),
                            "sigma": float(sigma),
                            "pk": base_pk,
                        }
                    )
            except (TypeError, ValueError):
                continue
    return results


def find_matching_pks(smear_pk, software: str, targets: list) -> list[dict]:
    """Find child pk values matching the requested (smear_method, sigma) combinations.

    Args:
        smear_pk: PK (or list of PKs) of the smear workflow(s). Pass a list
            when the same backend has been submitted via multiple parent
            WorkChains (e.g. ``abacus_soc`` and ``abacus_no_soc``); their
            children are pooled together before matching.
        software: 'vasp' or 'abacus'
        targets: List of dicts. Key naming follows the workflow's own input schema:
            - ``abacus``: each entry is ``{"smearing_method": str, "smearing_sigma": float (Ry)}``
            - ``vasp``:   each entry is ``{"ismear": int, "sigma": float (eV)}``

    Returns:
        List of dicts with the original keys plus a ``pk`` key (``None`` if not found).
    """
    pk_list = _resolve_smear_pk(smear_pk)
    snapshots = _snapshot_children(pk_list, software)

    results = []

    for target in targets:
        if software == "abacus":
            target_method = target["smearing_method"]
            target_sigma = float(target["smearing_sigma"])
            target_method_str = str(target_method).lower()
        else:
            target_method = target["ismear"]
            target_sigma = float(target["sigma"])
            target_method_int = int(target_method)

        matched_pk = None
        for smear, sigma, base_pk in snapshots:
            try:
                if software == "abacus":
                    if (
                        str(smear).lower() == target_method_str
                        and abs(float(sigma) - target_sigma) < 1e-9
                    ):
                        matched_pk = base_pk
                        break
                else:
                    if (
                        int(smear) == target_method_int
                        and abs(float(sigma) - target_sigma) < 1e-6
                    ):
                        matched_pk = base_pk
                        break
            except Exception:
                continue

        result = dict(target)
        result["pk"] = matched_pk
        results.append(result)

    return results


def _label_for_abacus(method: str, sigma_ry: float) -> str:
    """Build the ABACUS smear WorkChain ``copy`` leaf name.

    Thin wrapper around :func:`aiida_uranium_workflow.utils.labels.format_abacus_smear_label`
    so the legacy scp-based copy script and the new ``aiida-uranium copy``
    command stay on the same naming rule. The acceptance test
    :func:`tests.test_copy_remote.<...>test_format_abacus_smear_label`
    covers the centralised helper.
    """
    from aiida_uranium_workflow.utils.labels import format_abacus_smear_label

    return format_abacus_smear_label(method, sigma_ry)


def _label_for_vasp(ismear: int, sigma_ev: float) -> str:
    """Build the VASP smear WorkChain ``copy`` leaf name (see above)."""
    from aiida_uranium_workflow.utils.labels import format_vasp_smear_label

    return format_vasp_smear_label(ismear, sigma_ev)


def detect_software(smear_pk: int) -> str:
    """Return ``"abacus"`` or ``"vasp"`` based on the workchain process class."""
    process_class = load_node(smear_pk).process_class.__name__
    if process_class == "VaspSmearWorkChain":
        return "vasp"
    if process_class == "AbacusSmearWorkChain":
        return "abacus"
    raise ValueError(
        f"Unsupported WorkChain type: {process_class}. "
        f"Supported types: VaspSmearWorkChain, AbacusSmearWorkChain"
    )


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


# ---------------------------------------------------------------------------
# Defaults (mirror the canonical ``parameters/smear.yml`` 'test' preset)
# ---------------------------------------------------------------------------

#: Smearing methods scanned by default (ABACUS / VASP both support these).
DEFAULT_SMEAR_METHODS: tuple[str, ...] = ("gauss", "mp", "mp2")

#: Sigma grid in eV — mirrors ``parameters/smear.yml`` 'test' protocol.
#:
#: Note: ABACUS ``smearing_sigma`` is persisted in **eV** in the calcjob
#: parameters dict (the input_builder multiplies by ``_EV_TO_RY`` only
#: when forwarding the value into ABACUS' input file, not when storing
#: it back into ``base.inputs.abacus.parameters``). So we use eV for both
#: backends here.
DEFAULT_SIGMA_EV: tuple[float, ...] = (0.8, 0.5, 0.3, 0.2, 0.1, 0.08, 0.06)


def _build_default_targets() -> dict[str, list[dict[str, Any]]]:
    """Cross product ``DEFAULT_SMEAR_METHODS`` × ``DEFAULT_SIGMA_EV``.

    * VASP:  ``ismear`` maps from method name:
      - ``gauss`` → ``ismear=0``
      - ``mp``    → ``ismear=2`` (Methfessel-Paxton, order 1)
      - ``mp2``   → ``ismear=2`` (the order-2 variant is controlled by
        ``parameters/abacus/smear.yml``, not VASP's ismear alone)

    * ABACUS: ``smearing_method`` keeps the string name; sigma is in eV
      (see comment on ``DEFAULT_SIGMA_EV``).
    """
    vasp: list[dict[str, Any]] = []
    vasp_ismear_for_method = {"gauss": 0, "mp": 2, "mp2": 2}
    for method in DEFAULT_SMEAR_METHODS:
        ismear = vasp_ismear_for_method[method]
        for sigma_ev in DEFAULT_SIGMA_EV:
            vasp.append({"ismear": ismear, "sigma": sigma_ev})

    abacus = [
        {"smearing_method": method, "smearing_sigma": sigma_ev}
        for method in DEFAULT_SMEAR_METHODS
        for sigma_ev in DEFAULT_SIGMA_EV
    ]

    return {"vasp": vasp, "abacus": abacus}


DEFAULT_COPY_TARGETS: dict[str, list[dict[str, Any]]] = _build_default_targets()


# ---------------------------------------------------------------------------
# scp driver
# ---------------------------------------------------------------------------

def copy_targets(
    targets_with_pk: list[dict],
    software: str,
    subdir: str,
    *,
    base_dir: str,
    target_computer: str | None = None,
    ssh_opts: list[str] | None = None,
    source_computer: str | None = None,
) -> None:
    """Copy each matched child's remote_folder to ``{base_dir}/{subdir}/{label}``.

    Two modes:

    * **local** (``target_computer=None``): pull each remote_folder from
      the **compute node** (``source_computer``, e.g. ``"yeesuan"``) via
      ``scp -r`` into ``{base_dir}/{subdir}/{label}`` on the **current
      host**. ``source_computer`` must be set.
    * **remote** (``target_computer=<alias>``): ``remote_folder`` is
      already on ``target_computer``'s FS (because AiiDA submitted the
      job to that host), so we just ``ssh <alias> mkdir -p`` + ``ssh
      <alias> cp -r`` into ``{base_dir}/{subdir}/{label}`` on that host.

    When an entry carries a ``label`` key (as returned by
    :func:`discover_actual_calcs`), the destination becomes
    ``{base_dir}/{subdir}/{label}/{method_sigma_label}`` — useful when
    multiple WorkChains share the same (method, sigma) but should still
    land in different folders (e.g. ``lcao`` vs ``pw``).

    Args:
        targets_with_pk: list of dicts as returned by :func:`find_matching_pks`.
        software: ``"abacus"`` or ``"vasp"`` — controls label format.
        subdir: sub-path under ``base_dir`` (e.g. ``"smear/abacus"``).
        base_dir: destination root directory.
        target_computer: ssh host alias for remote mode, or ``None`` for
            local mode.
        ssh_opts: extra ``-o`` options forwarded to both ssh invocations
            (remote mode only).
        source_computer: ssh host alias for the compute node where the
            ``remote_folder`` lives. Used in local mode (scp source).
    """
    if target_computer is None and not source_computer:
        raise ValueError(
            "copy_targets(local mode) requires source_computer — the "
            "AiiDA remote_folder lives on a compute node, not the "
            "current host."
        )

    base_root = Path(base_dir)
    ssh_prefix: list[str] = []
    if ssh_opts and target_computer is not None:
        for opt in ssh_opts:
            ssh_prefix.extend(["-o", opt])

    for entry in targets_with_pk:
        pk = entry.get("pk")
        if pk is None:
            print(f"[{software}] skip (no pk for {entry})")
            continue

        if software == "abacus":
            leaf = _label_for_abacus(
                entry["smearing_method"], float(entry["smearing_sigma"])
            )
        else:
            leaf = _label_for_vasp(int(entry["ismear"]), float(entry["sigma"]))

        # When ``discover_actual_calcs`` was used, each entry has a
        # ``label`` (preset_name) — nest one more level so different
        # WorkChains don't overwrite each other.
        if "label" in entry:
            target_folder = base_root / subdir / entry["label"] / leaf
        else:
            target_folder = base_root / subdir / leaf

        remote_folder = load_node(pk).outputs.remote_folder.get_remote_path()

        if target_computer is None:
            # Local copy — scp from the compute node into the current host.
            target_folder.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "scp",
                    "-r",
                    f"{source_computer}:{remote_folder}/.",
                    str(target_folder),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ssh",
                    *ssh_prefix,
                    target_computer,
                    "mkdir",
                    "-p",
                    str(target_folder),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ssh",
                    *ssh_prefix,
                    target_computer,
                    "cp",
                    "-r",
                    f"{remote_folder}/.",
                    str(target_folder),
                ],
                check=True,
            )
        print(f"[{software}] pk={pk} -> {target_folder}")