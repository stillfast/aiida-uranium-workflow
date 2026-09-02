"""Stable, method/backend-aware directory labels for the unified ``copy`` command.

Background
----------
Before this module existed, ``utils/copy_remote.iter_copy_targets`` used
``CalcJobNode.process_label`` as the leaf directory name. Workflows that
submit bare CalcJobs without setting ``metadata.label`` (or that simply
inherit the generic class-level label) end up with names like
``AbacusCalculation`` — useless when scanning 40 child folders in a
shell. The WorkChain ``submit_children`` outline does set per-child
``metadata.label`` derived from the (parameter, kpoint) sweep
(``ecutwfc_60_kpoints_distance_0_2`` etc.), but a ProcessNode's
``process_label`` is the *class* name, not the submit-time label.

This module closes the gap by rebuilding the same label each workflow's
``submit_children`` outlined, working from the CalcJob itself and (if
needed) walking up its caller chain to the enclosing WorkChain that
exposed the original abacus.parameters / parameters.incar input. When
reconciliation is impossible we fall back through:

1. ``calcjob.metadata.label`` if the workflow explicitly set one.
2. ``calcjob.process_label`` (generic class name).
3. ``f"calcjob_{pk}"`` — guaranteed non-empty.

The label formats intentionally mirror each WorkChain's
``submit_children`` builders:
* smear labels follow the canonical format
  (``smearing_<method>_sigma_<v>`` / ``ismear_<n>_sigma_<v>``) — this
  is the single source of truth for the new ``aiida-uranium copy``
  command and any future re-exports.
* convergence labels mirror the workflow's exact
  ``f"ecutwfc_{ecutwfc}_kpoints_distance_{kpoints_val}".replace(".", "_")``
  / ``f"kpoints_spacing_{kpoints_val}_encut_{encut}".replace(".", "_")``
  pattern so ``meta-morphic`` and ``mesh`` submissions land in the same
  directory regardless of which command produced them.
* magmom labels mirror ``workflows/magmom/*.py``'s
  ``_mag_to_label`` / ``_magmom_to_label`` format
  (``f"{v:g}"`` then ``replace(".", "_").replace("-", "m")``).

Public surface
--------------
* :func:`resolve_label` — top-level entry point used by
  ``utils/copy_remote.iter_copy_targets``.
* :func:`format_*_label` — pure helpers; the single source of truth for
  label formatting, individually unit-testable.
"""

from __future__ import annotations

from typing import Any, Mapping


#: Generic WorkChain / CalcJob class names that are never useful as
#: directory names for sweep outputs. When the resolver cannot read the
#: calculation parameters and falls back to ``process_label`` or
#: ``metadata.label``, it must skip any of these and return
#: ``calcjob_<pk>`` instead — otherwise the user ends up with 40 folders
#: all named ``VaspWorkChain`` and a final UUID layer added on top by
#: AiiDA's remote folder layout.
_REJECTED_GENERIC_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "AbacusCalculation",
        "VaspCalculation",
        "AbacusWorkChain",
        "VaspWorkChain",
        "AbacusBaseWorkChain",
        "VaspBaseWorkChain",
        "AbacusConvergenceWorkChain",
        "VaspConvergenceWorkChain",
        "AbacusSmearWorkChain",
        "VaspSmearWorkChain",
        "AbacusMagmomWorkChain",
        "VaspMagmomWorkChain",
    }
)


# ---------------------------------------------------------------------------
# Pure formatters (no AiiDA imports; easy to unit-test)
# ---------------------------------------------------------------------------


def _sanitise_decimal(value: float | int) -> str:
    """Format a float as a smear-style filesystem-safe token.

    ``"%.6f"`` → ``replace(".", "_")`` → ``replace("-", "m")``.
    Used by ``smear`` only; convergence labels go through
    :func:`_decimal_token_short` which matches the workflow's submit
    children format.
    """
    return f"{float(value):.6f}".replace(".", "_").replace("-", "m")


def _decimal_token_short(value: float | int) -> str:
    """Format a number the same way the WorkChain's ``submit_children`` did.

    ``f"{value}"`` then ``replace(".", "_")`` then
    ``replace("-", "m")`` — matches ``workflows/*/.../submit_children``.
    """
    text = f"{value}"
    return text.replace(".", "_").replace("-", "m")


def format_smear_label(method: str, sigma: float | int) -> str:
    """Canonical *workflow* smear label — ``smear_<method>_sigma_<v>``.

    Single source of truth for the labels the smear WorkChains assign
    to their children (and the keys of their ``output_parameters``):
    ABACUS passes ``smearing_method`` (e.g. ``"mp"``), VASP passes the
    integer ``ismear`` — both share this format. The sigma token keeps
    the full ``f"{sigma}"`` precision (matching the historical
    workflow labels), with ``.`` → ``_``.

    Note: this differs from the ``copy``-command labels
    (:func:`format_abacus_smear_label` / :func:`format_vasp_smear_label`,
    ``smearing_*`` / ``ismear_*`` with ``%.6f``-rounded sigma) — those
    are used for remote-folder directory names.
    """
    return f"smear_{method}_sigma_{_decimal_token_short(sigma)}"


def format_abacus_smear_label(method: str, sigma: float | int) -> str:
    """``smearing_<method>_sigma_<value>``."""
    return f"smearing_{method}_sigma_{_sanitise_decimal(sigma)}"


def format_vasp_smear_label(ismear: int, sigma: float | int) -> str:
    """``ismear_<n>_sigma_<value>``."""
    return f"ismear_{int(ismear)}_sigma_{_sanitise_decimal(sigma)}"


def format_abacus_convergence_label(
    ecutwfc: float | int,
    kpoints: Any,
) -> str:
    """``ecutwfc_<v>_kpoints_distance_<v>`` *or* ``ecutwfc_<v>_kpoints_<NxNxN>``.

    Matches :class:`AbacusConvergenceWorkChain.submit_children` /
    :func:`parse_and_gather_convergence_results` exactly. ``kpoints``
    is treated as a ``kpoints_distance`` scalar by default; a 3-int
    iterable (or tuple of ints) is rendered as ``NxNxN``.
    """
    ecutwfc_str = _decimal_token_short(ecutwfc)
    if _looks_like_mesh(kpoints):
        mesh = "x".join(str(int(k)) for k in kpoints)
        return f"ecutwfc_{ecutwfc_str}_kpoints_{mesh}"
    return f"ecutwfc_{ecutwfc_str}_kpoints_distance_{_decimal_token_short(kpoints)}"


def format_vasp_convergence_label(
    kpoints: Any,
    encut: float | int,
) -> str:
    """``kpoints_spacing_<v>_encut_<v>`` *or* ``kpoints_<NxNxN>_encut_<v>``.

    Matches :class:`VaspConvergenceWorkChain.submit_children` exactly.
    """
    encut_str = _decimal_token_short(encut)
    if _looks_like_mesh(kpoints):
        mesh = "x".join(str(int(k)) for k in kpoints)
        return f"kpoints_{mesh}_encut_{encut_str}"
    return f"kpoints_spacing_{_decimal_token_short(kpoints)}_encut_{encut_str}"


def format_magmom_label(
    mag_value: Any,
    *,
    index: int | None = None,
) -> str:
    """Stable, sortable label for a magmom configuration.

    Mirrors ``workflows/magmom/abacus._mag_to_label`` and
    ``workflows/magmom/vasp._magmom_to_label``: ``f"{v:g}"`` per scalar
    → ``replace(".", "_").replace("-", "m")`` once at the end.

    ``mag_value`` accepts:
    * a flat list (ABACUS ``stru.mag`` rows): ``[1.0, -1.0]``
    * a nested list of per-atom vectors (ABACUS nested form):
      ``[[1.0], [-1.0]]``
    * a dict (VASP ``magmom_mapping`` ``{"U": 1.0}`` /
      ``{"U": [1.0, -1.0]}``)
    * a scalar

    ``index`` prepends ``magmom_<idx:03d>_`` when the workflow knows
    the child order. The :func:`resolve_label` glue doesn't have the
    index, so callers that want the same form the workflow uses
    should pass it.
    """
    parts: list[str] = []
    if isinstance(mag_value, Mapping):
        for element, value in mag_value.items():
            if isinstance(value, (list, tuple)):
                inner = "_".join(_scalar_token(float(v)) for v in value)
            else:
                inner = _scalar_token(float(value))
            parts.append(f"{element}_{inner}")
        body = "__".join(parts)
    elif isinstance(mag_value, (list, tuple)):
        for row in mag_value:
            if isinstance(row, (list, tuple)):
                parts.extend(_scalar_token(float(v)) for v in row)
            else:
                parts.append(_scalar_token(float(row)))
        body = "_".join(parts)
    else:
        body = _scalar_token(float(mag_value))
    body = body.replace(".", "_").replace("-", "m")
    if index is not None:
        return f"magmom_{index:03d}_{body}"
    return f"magmom_{body}"


def _scalar_token(value: float) -> str:
    """``f"{value:g}"`` — short Python repr (matches workflow ``_mag_*``)."""
    return f"{float(value):g}"


# ---------------------------------------------------------------------------
# AiiDA-side introspection (lazy, defensive)
# ---------------------------------------------------------------------------


def _looks_like_mesh(value: Any) -> bool:
    """``True`` when ``value`` is a 3-int iterable — i.e. a kpoints mesh."""
    if isinstance(value, str):
        return False
    try:
        seq = list(value)
    except TypeError:
        return False
    return len(seq) == 3 and all(
        isinstance(x, (int,)) and not isinstance(x, bool) for x in seq
    )


def _safe_get_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    getter = getattr(node, "get_dict", None)
    if not callable(getter):
        return {}
    try:
        result = getter()
    except Exception:
        return {}
    return result if isinstance(result, dict) else {}


def _safe_get_input(inputs: Any, key: str) -> Any:
    """Safely get an input from inputs, handling both dict-like and NodeLinksManager."""
    if inputs is None:
        return None
    try:
        return getattr(inputs, key)
    except Exception:
        pass
    try:
        if hasattr(inputs, "__contains__") and key in inputs:
            return inputs[key]
    except Exception:
        pass
    return None


def _safe_value(node: Any) -> Any:
    if node is None:
        return None
    return getattr(node, "value", None)


def _safe_get_kpoints_mesh(node: Any) -> tuple[int, int, int] | None:
    if node is None:
        return None
    getter = getattr(node, "get_kpoints_mesh", None)
    if not callable(getter):
        return None
    try:
        mesh, _offsets = getter()
    except Exception:
        try:
            mesh = getter()
        except Exception:
            return None
    if mesh is None or len(mesh) < 3:
        return None
    try:
        return int(mesh[0]), int(mesh[1]), int(mesh[2])
    except (TypeError, ValueError):
        return None


def _walk_caller_ancestors(node: Any) -> list[Any]:
    """Return ``[calcjob, calcjob.caller, ..., top-level caller]``.

    AiiDA exposes the direct caller chain via the ``caller`` link. We
    follow it upward until we run out of nodes, breaking on cycles.
    """
    seen: set[int] = set()
    chain: list[Any] = []
    current = node
    while current is not None:
        node_id = id(current)
        if node_id in seen:
            break
        seen.add(node_id)
        chain.append(current)
        try:
            pk = getattr(current, "pk", None)
        except Exception:
            pk = None
        if pk is not None:
            try:
                seen.add(int(pk))
            except (TypeError, ValueError):
                pass
        current = getattr(current, "caller", None)
    return chain


def _read_abacus_kpoints(calcjob: Any, parent: Any) -> tuple[Any, str]:
    """Return ``(value, mode)`` for kpoints resolution on ABACUS side.

    ``mode`` is ``"distance"`` or ``"mesh"``.
    """
    for node in (calcjob, parent):
        if node is None:
            continue
        try:
            inputs = node.inputs
        except (AttributeError, KeyError):
            inputs = None
        if inputs is None:
            continue

        value = _safe_value(_safe_get_input(inputs, "kpoints_distance"))
        if value is not None:
            try:
                return float(value), "distance"
            except (TypeError, ValueError):
                pass

        kp = _safe_get_input(inputs, "kpoints")
        mesh = _safe_get_kpoints_mesh(kp)
        if mesh is not None:
            return mesh, "mesh"

        raw = getattr(kp, "value", None)
        if isinstance(raw, str):
            tokens = raw.split()
            if len(tokens) == 3 and all(t.lstrip("-").isdigit() for t in tokens):
                return (int(tokens[0]), int(tokens[1]), int(tokens[2])), "mesh"
    return None, "distance"


def _read_vasp_kpoints(calcjob: Any, parent: Any) -> tuple[Any, str]:
    """Same as :func:`_read_abacus_kpoints` but for VASP namespace."""
    for node in (calcjob, parent):
        if node is None:
            continue
        try:
            inputs = node.inputs
        except (AttributeError, KeyError):
            inputs = None
        if inputs is None:
            continue

        value = _safe_value(_safe_get_input(inputs, "kpoints_spacing"))
        if value is not None:
            try:
                return float(value), "spacing"
            except (TypeError, ValueError):
                pass

        kp = _safe_get_input(inputs, "kpoints")
        mesh = _safe_get_kpoints_mesh(kp)
        if mesh is not None:
            return mesh, "mesh"
    return None, "spacing"


def _extract_abacus_param(inputs: Any, key: str = "ecutwfc") -> Any:
    """Pull ``inputs.abacus.parameters.input.<key>`` if available."""
    if inputs is None:
        return None
    abacus_block = _safe_get_input(inputs, "abacus")
    if abacus_block is None:
        return None
    parameters = _safe_get_input(abacus_block, "parameters")
    data = _safe_get_dict(parameters)
    input_block = data.get("input", {}) if isinstance(data.get("input"), dict) else {}
    return input_block.get(key)


def _extract_abacus_inputs(inputs: Any) -> dict[str, Any]:
    """Return the ``abacus.parameters.input`` dict (or empty)."""
    if inputs is None:
        return {}
    abacus_block = _safe_get_input(inputs, "abacus")
    if abacus_block is None:
        # A submitted ABACUS CalcJob exposes ``parameters`` directly;
        # the ``abacus`` namespace is used by the parent WorkChain.
        parameters = _safe_get_input(inputs, "parameters")
        data = _safe_get_dict(parameters)
        return data.get("input", {}) if isinstance(data.get("input"), dict) else {}
    parameters = _safe_get_input(abacus_block, "parameters")
    data = _safe_get_dict(parameters)
    return data.get("input", {}) if isinstance(data.get("input"), dict) else {}


def _extract_incar(inputs: Any) -> dict[str, Any]:
    """Return ``inputs.parameters.get_dict().incar`` (or empty)."""
    if inputs is None:
        return {}
    parameters = _safe_get_input(inputs, "parameters")
    data = _safe_get_dict(parameters)
    return data.get("incar", {}) if isinstance(data.get("incar"), dict) else {}


def _extract_stru_mag(inputs: Any) -> Any:
    """Return ``inputs.abacus.parameters.get_dict().stru.mag`` if any."""
    if inputs is None:
        return None
    abacus_block = _safe_get_input(inputs, "abacus")
    if abacus_block is None:
        return None
    parameters = _safe_get_input(abacus_block, "parameters")
    data = _safe_get_dict(parameters)
    stru = data.get("stru", {}) if isinstance(data.get("stru"), dict) else {}
    return stru.get("mag")


def _extract_vasp_magmom(inputs: Any) -> Any:
    """Return the VASP initial magnetic-moment input if available.

    Two port styles are supported:

    * ``magmom_mapping`` — per-species dict (``{"U": 4.0}`` /
      ``{"U": [1.0, -1.0]}``), the older ``magmom_list`` form.
    * ``magmom_per_atom`` — per-site list (``[4.0, -4.0]`` /
      ``[[0.0, 0.0, 4.0], [0.0, 0.0, -4.0]]``), the per-atom form used
      by the newer ``magmom_per_atom_list`` sweep.
    """
    if inputs is None:
        return None
    candidate = _safe_get_input(inputs, "magmom_mapping")
    if candidate is not None:
        data = _safe_get_dict(candidate)
        return data if data else candidate
    candidate = _safe_get_input(inputs, "magmom_per_atom")
    if candidate is None:
        return None
    try:
        return list(candidate.get_list())
    except (AttributeError, TypeError):
        return None


def _get_inputs_safely(node: Any) -> Any:
    """``node.inputs``; ``None`` when unavailable (un-stored, dry-run, …)."""
    if node is None:
        return None
    try:
        return node.inputs
    except (AttributeError, KeyError):
        return None
    except Exception:
        return None


def _root_direct_candidates(root_workchain: Any, calcjob: Any) -> list[Any]:
    """Return root children most likely associated with ``calcjob``.

    Prefer the direct child that is on the CalcJob's caller chain, then
    inspect the remaining direct children.  Sweep parameters are persisted
    on these base nodes even when the final CalcJob caller chain does not
    expose them.
    """
    if root_workchain is None:
        return []
    try:
        direct = list(root_workchain.called)
    except Exception:
        return []

    ancestors = _walk_caller_ancestors(calcjob)[1:]
    ancestor_ids = {id(node) for node in ancestors}
    ancestor_pks = {getattr(node, "pk", None) for node in ancestors}
    matched = [
        node
        for node in direct
        if id(node) in ancestor_ids or getattr(node, "pk", None) in ancestor_pks
    ]
    return [*matched, *(node for node in direct if node not in matched)]


def _candidate_nodes(calcjob: Any, root_workchain: Any = None) -> list[Any]:
    """Return parameter sources in preferred resolution order."""
    nodes = _root_direct_candidates(root_workchain, calcjob)
    nodes.extend(_walk_caller_ancestors(calcjob))
    unique: list[Any] = []
    seen: set[int] = set()
    for node in nodes:
        marker = id(node)
        if marker not in seen:
            seen.add(marker)
            unique.append(node)
    return unique


# ---------------------------------------------------------------------------
# Top-level resolver
# ---------------------------------------------------------------------------


def resolve_label(
    calcjob: Any,
    *,
    backend: str,
    method: str,
    root_workchain: Any = None,
) -> str:
    """Compute a stable, semantics-aware label for ``calcjob``.

    ``root_workchain`` optionally supplies the top-level workflow whose
    direct ``called`` base nodes hold the submitted sweep parameters.
    Always returns a non-empty string. The hierarchy is documented at
    the top of the module; briefly:

    1. ``method``-specific formatters built from matching direct children
       of ``root_workchain`` when supplied.
    2. Same formatters using the CalcJob and its caller WorkChains.
    3. ``metadata.label`` if the workflow set it explicitly.
    4. ``process_label`` (generic class name).
    5. ``f"calcjob_{pk}"`` as ultimate fallback.
    """
    backend = (backend or "").lower()
    method = (method or "").lower()

    candidates = _candidate_nodes(calcjob, root_workchain)
    parent_chain = _walk_caller_ancestors(calcjob)[1:]

    computed: str | None = None
    if backend == "abacus" and method == "smear":
        computed = _resolve_abacus_smear(candidates)
    elif backend == "vasp" and method == "smear":
        computed = _resolve_vasp_smear(candidates)
    elif backend == "abacus" and method == "convergence":
        computed = _resolve_abacus_convergence(candidates)
    elif backend == "vasp" and method == "convergence":
        computed = _resolve_vasp_convergence(candidates)
    elif method == "magmom":
        computed = _resolve_magmom(candidates, backend=backend)

    if computed:
        return computed

    # Generic fallbacks — only consulted when the method-specific chain
    # could not derive a label. AiiDA may store the submit-time label on a
    # caller WorkChain rather than on the CalcJob itself. Prefer a
    # meaningful caller label before falling back to the generic CalcJob
    # class name. The class names ``*WorkChain`` / ``*Calculation`` are
    # never meaningful directory names — they would be useless in a
    # sweep with 40 children — so we reject them at every fallback step
    # and use ``calcjob_<pk>`` instead.
    rejected = _REJECTED_GENERIC_CLASS_NAMES
    for node in parent_chain:
        meta_label = getattr(getattr(node, "metadata", None), "label", None)
        if meta_label and str(meta_label) not in rejected:
            return str(meta_label)
        process_label = getattr(node, "process_label", None)
        if process_label and str(process_label) not in rejected:
            return str(process_label)

    meta_label = getattr(getattr(calcjob, "metadata", None), "label", None)
    if meta_label and str(meta_label) not in rejected:
        return str(meta_label)

    process_label = getattr(calcjob, "process_label", None)
    if process_label and str(process_label) not in rejected:
        return str(process_label)

    return f"calcjob_{getattr(calcjob, 'pk', 'unknown')}"


def _resolve_abacus_smear(candidates: list[Any]) -> str | None:
    for node in candidates:
        inputs = _get_inputs_safely(node)
        abacus_input = _extract_abacus_inputs(inputs)
        if "smearing_method" in abacus_input and "smearing_sigma" in abacus_input:
            try:
                return format_abacus_smear_label(
                    str(abacus_input["smearing_method"]),
                    float(abacus_input["smearing_sigma"]),
                )
            except (TypeError, ValueError):
                continue
    return None


def _resolve_vasp_smear(candidates: list[Any]) -> str | None:
    for node in candidates:
        inputs = _get_inputs_safely(node)
        incar = _extract_incar(inputs)
        if "ismear" in incar and "sigma" in incar:
            try:
                return format_vasp_smear_label(
                    int(incar["ismear"]),
                    float(incar["sigma"]),
                )
            except (TypeError, ValueError):
                continue
    return None


def _resolve_abacus_convergence(candidates: list[Any]) -> str | None:
    for node in candidates:
        inputs = _get_inputs_safely(node)
        ecutwfc = _extract_abacus_param(inputs, "ecutwfc")
        if ecutwfc is None:
            continue
        kpoints, _mode = _read_abacus_kpoints(node, None)
        if kpoints is None:
            continue
        try:
            return format_abacus_convergence_label(ecutwfc, kpoints)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_vasp_convergence(candidates: list[Any]) -> str | None:
    for node in candidates:
        inputs = _get_inputs_safely(node)
        incar = _extract_incar(inputs)
        if "encut" not in incar:
            continue
        encut = incar["encut"]
        kpoints, _mode = _read_vasp_kpoints(node, None)
        if kpoints is None:
            continue
        try:
            return format_vasp_convergence_label(kpoints, encut)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_magmom(candidates: list[Any], *, backend: str) -> str | None:
    for node in candidates:
        inputs = _get_inputs_safely(node)
        if backend == "abacus":
            mag_value = _extract_stru_mag(inputs)
        else:
            mag_value = _extract_vasp_magmom(inputs)
        if mag_value is None:
            continue
        try:
            return format_magmom_label(mag_value)
        except (TypeError, ValueError, AttributeError):
            continue
    return None


__all__ = [
    "format_abacus_smear_label",
    "format_vasp_smear_label",
    "format_abacus_convergence_label",
    "format_vasp_convergence_label",
    "format_magmom_label",
    "resolve_label",
]
