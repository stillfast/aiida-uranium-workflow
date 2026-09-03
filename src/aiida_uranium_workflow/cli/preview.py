"""DB-aware dry-run preview: one YAML file per planned WorkChain.

``check --out DIR`` builds the inputs every planned submission *would*
receive — running each backend's input builder against the real AiiDA
profile (codes, pseudo families, structures are resolved) — and writes
one YAML per (backend, preset, protocol-preset, structure) so the user
can review the parameters that actually reach each WorkChain before
anything is submitted.

Nothing here submits: :meth:`WorkflowOrchestrator.prepare` stops after
the adapter ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml


def serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Turn an adapter-produced inputs dict into YAML-safe plain values.

    AiiDA ORM nodes are converted to reviewable forms: ``Code`` → code
    label, ``Dict`` → dict, ``List`` → list, ``KpointsData`` → mesh /
    distance, ``StructureData`` → formula summary, pseudo nodes → element
    name. Plain values pass through untouched, so the YAML mirrors what
    ``submit(workchain_cls, **inputs)`` would receive.
    """
    from aiida import orm

    def plain(value: Any) -> Any:
        if isinstance(value, orm.Dict):
            return value.get_dict()
        if isinstance(value, orm.List):
            return value.get_list()
        if isinstance(value, orm.Float):
            return value.value
        if isinstance(value, orm.Int):
            return value.value
        if isinstance(value, orm.Str):
            return value.value
        if isinstance(value, orm.Bool):
            return value.value
        if isinstance(value, orm.Code):
            return value.full_label
        if isinstance(value, orm.KpointsData):
            mesh = value.get_kpoints_mesh()
            if mesh is not None:
                return {
                    "kpoints_mesh": [int(n) for n in mesh[0]],
                    "offset": [float(o) for o in mesh[1]],
                }
            distance = value.get_kpoints_distance()
            if distance is not None:
                return {"kpoints_distance": float(distance)}
            return value.get_kpoints().tolist()
        if isinstance(value, orm.StructureData):
            return {
                "formula": value.get_formula(),
                "n_atoms": len(value.sites),
            }
        if isinstance(value, orm.Node):
            # Pseudo nodes (UpfData etc.) — show the element when the
            # node carries one, otherwise fall back to a type marker.
            element = getattr(value, "element", None)
            if element is not None:
                return f"{element} ({value.__class__.__name__})"
            return f"<{value.__class__.__name__}>"
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    return plain(inputs)


def _slugify(name: str) -> str:
    """Sanitize a plan label into a filename-safe fragment."""
    out = []
    for char in name:
        if char.isalnum() or char in "-_.":
            out.append(char)
        else:
            out.append("_")
    return "".join(out)


def write_preview_files(
    bundle,
    out_dir: str | Path,
    *,
    profile: str | None = None,
) -> List[Path]:
    """Dry-run every planned WorkChain and write one YAML per submission.

    ``bundle`` is a loaded :class:`ParamBundle` (from
    ``ConfigLoader(...).load_all()``). Requires a live AiiDA profile with
    the referenced codes / pseudo families / structures installed, since
    each input builder runs for real (minus the submit).

    Returns the list of written file paths (under
    ``<out_dir>/<workflow>/``).
    """
    from aiida_uranium_workflow.schedulers import get_orchestrator

    workflow = bundle.input_params["workflow"]
    orchestrator = get_orchestrator(bundle)
    jobs = orchestrator.prepare(profile=profile)

    # Single-protocol-preset inputs name the preset under the workflow
    # key (e.g. ``"magmom": "test_u_afm_qe"``); multi-preset lists are
    # covered by the per-job ``preset_name`` ("preset/protocol").
    protocol_preset = None
    try:
        from aiida_uranium_workflow.schedulers import get_workflow_entry

        wf_key = get_workflow_entry(workflow).workflow_key
        raw = bundle.input_params.get("parameters", {}).get(wf_key) if wf_key else None
        if isinstance(raw, str):
            protocol_preset = raw
    except Exception:  # noqa: BLE001 — protocol name is cosmetic only
        pass

    out_root = Path(out_dir)
    out_workflow = out_root / _slugify(workflow)
    out_workflow.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for job in jobs:
        preset = job.preset_name  # may be "preset/protocol" (list form)
        stem = _slugify(f"{job.backend}_{preset}_{job.structure_name}")
        path = out_workflow / f"{stem}.yml"
        payload = {
            "workflow": workflow,
            "backend": job.backend,
            "preset": job.preset_name,
            "structure": job.structure_name,
            "workchain": job.workchain_cls.__name__,
            "inputs": serialize_inputs(job.inputs),
        }
        if protocol_preset:
            payload["protocol_preset"] = protocol_preset
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        written.append(path)
    return written
