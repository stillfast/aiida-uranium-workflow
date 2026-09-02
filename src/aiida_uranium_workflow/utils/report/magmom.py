"""Magmom WorkChain report generation module.

Extracts magnetism data from ``AbacusMagmomWorkChain`` /
``VaspMagmomWorkChain`` / ``FleurMagmomWorkChain`` ``output_parameters``
and renders a Markdown report.

The per-child single-column tables (``status`` / ``energy`` /
``wall_time``) and the report header / footer are shared with the
smear / convergence reports via :mod:`utils.report._common`. The
``generate_*`` entry points keep their original signatures so existing
callers and tests are unaffected.

Per-backend key layout produced by the magmom workflows:

* ABACUS (``AbacusMagmomWorkChain``)
    - ``magnetism``        — ``{pk: [<per-atom mag>, ...] | None}``
    - ``final_magnetism``  — ``{pk: <total magnetization> | None}``
    - ``nspin``            — ``{pk: <nspin value> | None}``
* VASP (``VaspMagmomWorkChain``)
    - ``magnetization``        — ``{pk: <total magnetization> | None}``
    - ``site_magnetization``   — ``{pk: <per-site magnetization> | None}``
* FLEUR (``FleurMagmomWorkChain``)
    - ``magnetization``        — ``{pk: [[mx, my, mz], ...] | None}``
      (per-atom 3-vectors from ``<globalMagMoment vec="…"/>``)
    - ``total_energy_hartree`` — ``{pk: <E in Hartree> | None}``
    - ``final_energy``         — ``{pk: <E in eV> | None}``
    - ``config_labels``        — ``{pk: <config tag, e.g. "FM"> | None}``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._common import (
    format_scalar,
    render_per_child_table,
    render_report_footer,
    render_report_header,
)

#: eV per Hartree (used to convert the FLEUR ΔE from Hartree to meV).
_HA_TO_EV = 27.211386245988


def _fmt_vec(value: Any, prec: int = 2) -> str:
    """Render a scalar or 3-vector like the reference table (``+0.00`` /
    ``[+0.00, +0.00, -0.00]``)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):+.{prec}f}"
    if isinstance(value, (list, tuple)):
        try:
            return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in value) + "]"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _entry_is_nm(entry: Any) -> bool:
    """True when a ``magmom_list`` entry has zero initial magnetization."""
    if entry is None:
        return True
    if isinstance(entry, dict):
        bmu = entry.get("bmu")
        if bmu is None:
            return True
        values = bmu if isinstance(bmu, (list, tuple)) else [bmu]
        try:
            return all(abs(float(x)) < 1e-8 for x in values)
        except (TypeError, ValueError):
            return False

    flat: List[float] = []

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
        else:
            flat.append(x)

    _walk(entry)
    if not flat:
        return True
    try:
        return all(abs(float(x)) < 1e-8 for x in flat)
    except (TypeError, ValueError):
        return False


def _cell_total_vector(final_value: Any) -> Any:
    """Extract the cell total magnetization vector from a child's final value."""
    if isinstance(final_value, dict):
        total = final_value.get("total_magnetism")
        if total is not None:
            return total
        return final_value.get("absolute_magnetism")
    return final_value


def _initial_atom_mag(entry: Any, atom_idx: int) -> str:
    """Per-atom initial magnetization from a ``magmom_list`` entry.

    * FLEUR — config dict with ``bmu`` (scalar or vector) seeding every
      atom of the species.
    * VASP — per-species mapping ``{"U": 4.0}``; when a single species
      is present the same seed is repeated for every atom column.
    * ABACUS — nested per-atom list ``[[mx], [my]]``.
    """
    if entry is None:
        return "—"
    if isinstance(entry, dict):
        if "bmu" in entry:
            bmu = entry["bmu"]
            return "—" if bmu is None else _fmt_vec(bmu, 2)
        # VASP species mapping: repeat the single seed for every atom.
        values = [v for v in entry.values() if v is not None]
        if len(values) == 1:
            return _fmt_vec(values[0], 2)
        return "—"
    if isinstance(entry, (list, tuple)):
        if atom_idx < len(entry):
            return _fmt_vec(entry[atom_idx], 2)
        return "—"
    return "—"


def _site_to_atom_moments(site_magnetization: Any) -> Optional[List[Any]]:
    """Best-effort extraction of per-atom moments from VASP ``site_magnetization``.

    VASP stores ``{"sphere": {"x": {"site_moment": {...}}, ...},
    "full_cell": [...]}``. Only when the per-site moments were parsed do
    we return them; otherwise ``None`` (columns render as "—").
    """
    if not isinstance(site_magnetization, dict):
        return None
    sphere = site_magnetization.get("sphere")
    if not isinstance(sphere, dict):
        return None
    for axis_dict in sphere.values():
        if not isinstance(axis_dict, dict):
            continue
        site_moment = axis_dict.get("site_moment")
        if isinstance(site_moment, dict) and site_moment:
            # {site_index: value} → ordered by integer index.
            try:
                return [
                    site_moment[k] for k in sorted(site_moment, key=lambda x: int(x))
                ]
            except (TypeError, ValueError):
                return list(site_moment.values())
    return None


def generate_magmom_matrix_table(
    output_params: Dict[str, Any],
    workflow_type: str = "abacus",
) -> str:
    """Render the compact per-configuration magnetism matrix table.

    Mirrors the ``magmom_test`` postprocess reference, with the child
    **pk** as the row key::

        pk | E_tot (Htr|eV) | ΔE vs ref (meV) | M_cell (µ_B) |
        M_U1 .. M_Un (µ_B) | M_start_U1 .. M_start_Un (µ_B)

    The ΔE reference is the non-magnetic (NM) child — the one whose
    ``magmom_list`` entry has zero initial magnetization (falls back to
    the first child). Per-atom *final* moments are only available for the
    FLEUR backend (ABACUS's ``misc`` reports the cell total per
    electronic step only, not per-atom moments).
    """
    # Energies — native unit per backend (Htr for FLEUR, Ry for QE, eV
    # otherwise).
    if workflow_type == "fleur":
        energy = output_params.get("total_energy_hartree") or {}
        e_unit = "Htr"
    elif workflow_type == "qe":
        energy = output_params.get("final_energy") or {}
        e_unit = "Ry"
    else:
        energy = output_params.get("final_energy") or {}
        e_unit = "eV"

    # Per-atom final moments + cell total.
    if workflow_type == "fleur":
        raw = output_params.get("magnetization") or {}
        per_atom_final: Dict[Any, Any] = dict(raw)
        cell_total: Dict[Any, Any] = {}
        for pk, vecs in raw.items():
            if (
                isinstance(vecs, list)
                and vecs
                and all(isinstance(v, (list, tuple)) for v in vecs)
            ):
                cell_total[pk] = [sum(a[i] for a in vecs) for i in range(3)]
            else:
                cell_total[pk] = vecs
    elif workflow_type == "vasp":
        raw = output_params.get("magnetization") or {}
        per_atom_final = {
            pk: _site_to_atom_moments(v)
            for pk, v in (output_params.get("site_magnetization") or {}).items()
        }
        cell_total = {pk: _cell_total_vector(v) for pk, v in raw.items()}
    elif workflow_type == "qe":
        # QE reports scalar total / absolute magnetization (μB per cell).
        raw = output_params.get("magnetization") or {}
        per_atom_final = {
            pk: (output_params.get("atomic_magnetic_moments") or {}).get(pk)
            for pk in raw
        }
        cell_total = {
            pk: (v if isinstance(v, (list, tuple)) else [float(v)])
            for pk, v in raw.items()
        }
    else:
        per_atom_final = {}
        raw = output_params.get("final_magnetism") or {}
        cell_total = {pk: _cell_total_vector(v) for pk, v in raw.items()}

    magmom_list = output_params.get("magmom_list") or []

    # Row order: the pks of the energy dict (submission order).
    pks = list(energy.keys())
    if not pks:
        pks = list(cell_total.keys())
    if not pks:
        return "No magnetism data available."

    # ΔE reference: the first NM (zero initial magmom) child, else the first.
    ref_pk = None
    for idx, pk in enumerate(pks):
        entry = magmom_list[idx] if idx < len(magmom_list) else None
        if _entry_is_nm(entry):
            ref_pk = pk
            break
    if ref_pk is None:
        ref_pk = pks[0]
    ref_e = energy.get(ref_pk)

    # Number of atoms: from the per-atom final moments, else magmom_list.
    natoms: Optional[int] = None
    for pk in pks:
        vecs = per_atom_final.get(pk)
        if isinstance(vecs, list) and vecs:
            natoms = len(vecs)
            break
    if natoms is None:
        for entry in magmom_list:
            if isinstance(entry, (list, tuple)):
                natoms = len(entry)
                break
    natoms = natoms or 0

    # Per-atom final moments are only available for FLEUR (per-atom
    # vectors) and VASP (per-site moments when parsed). ABACUS never
    # prints per-atom converged magnetization (only the cell total per
    # SCF iteration), so hide the M_U columns entirely instead of
    # showing a whole column of "—".
    has_per_atom_final = any(isinstance(v, list) and v for v in per_atom_final.values())

    header = ["pk", f"E_tot ({e_unit})", "ΔE vs ref (meV)", "M_cell (µ_B)"]
    if has_per_atom_final:
        for i in range(natoms):
            header.append(f"M_U{i + 1} (µ_B)")
    for i in range(natoms):
        header.append(f"M_start_U{i + 1} (µ_B)")
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | " + " | ".join("---" for _ in header[1:]) + " |",
    ]

    for idx, pk in enumerate(pks):
        e = energy.get(pk)
        e_str = "—" if e is None else f"{float(e):.8f}"
        de = None
        if e is not None and ref_e is not None:
            de = (float(e) - float(ref_e)) * (
                1000.0 if e_unit == "eV" else 1000.0 * _HA_TO_EV
            )
        de_str = "—" if de is None else f"{de:+.3f}"
        cell_str = _fmt_vec(cell_total.get(pk), 2)

        entry = magmom_list[idx] if idx < len(magmom_list) else None

        fin = per_atom_final.get(pk)
        if has_per_atom_final and isinstance(fin, list) and fin:
            fin_cols = [
                _fmt_vec(fin[i], 2) if i < len(fin) else "—" for i in range(natoms)
            ]
        else:
            fin_cols = []

        start_cols = [_initial_atom_mag(entry, i) for i in range(natoms)]

        row = [str(pk), e_str, de_str, cell_str] + fin_cols + start_cols
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-child single-column tables
# ---------------------------------------------------------------------------


def generate_status_table(status: Dict[str, int]) -> str:
    """Render the per-child exit-code status as a Markdown table."""
    if not status:
        return "No status data available."

    lines = ["| child pk | exit_status |", "| --- | --- |"]
    for label, exit_code in status.items():
        lines.append(f"| {label} | {exit_code} |")
    return "\n".join(lines)


def generate_energy_table(final_energy: Dict[Any, Any]) -> str:
    """Render a Markdown table of per-child final energy."""
    return render_per_child_table(
        final_energy,
        column_header="final_energy",
        column_name="final_energy",
        cell_format=lambda v: format_scalar(v, fmt="%.6f"),
    )


def generate_wall_time_table(wall_time_seconds: Dict[Any, Any]) -> str:
    """Render a Markdown table of per-child wall-clock time (seconds)."""
    return render_per_child_table(
        wall_time_seconds,
        column_header="wall_time [s]",
        column_name="wall_time",
        cell_format=lambda v: format_scalar(v, fmt="%.3f"),
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def generate_summary_table(
    output_params: Dict[str, Any],
    workflow_type: Optional[str] = None,
) -> str:
    """Generate a summary table with basic info.

    ``workflow_type`` ("abacus" / "vasp" / "fleur") pins the backend
    block in the table — needed because VASP and FLEUR both expose
    ``magnetization`` and would otherwise be ambiguous.
    """
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    if "magmom_list" in output_params:
        lines.append(f"| magmom_list entries | {len(output_params['magmom_list'])} |")

    backend_keys = {
        "abacus": ("magnetism", "final_magnetism"),
        "vasp": ("magnetization", "site_magnetization"),
        "fleur": ("magnetization", "total_energy_hartree"),
    }
    if workflow_type is not None and workflow_type in backend_keys:
        primary, secondary = backend_keys[workflow_type]
        primary_data = output_params.get(primary, {}) or {}
        secondary_data = output_params.get(secondary, {}) or {}
        lines.append(f"| backend | {workflow_type} |")
        lines.append(f"| {primary} entries | {len(primary_data)} |")
        if secondary_data:
            lines.append(f"| {secondary} entries | {len(secondary_data)} |")
    else:
        # Fall back to first backend whose primary key has data.
        for backend, (primary, secondary) in backend_keys.items():
            primary_data = output_params.get(primary, {}) or {}
            secondary_data = output_params.get(secondary, {}) or {}
            if primary_data:
                lines.append(f"| backend | {backend} |")
                lines.append(f"| {primary} entries | {len(primary_data)} |")
                if secondary_data:
                    lines.append(f"| {secondary} entries | {len(secondary_data)} |")
                break

    return "\n".join(lines)


# ---------------------------------------------------------------------------
def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report for a magmom WorkChain."""
    lines: List[str] = [
        render_report_header(
            title="Magmom WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params, workflow_type),
        "",
        "## Magnetism Matrix",
        "",
        generate_magmom_matrix_table(output_params, workflow_type),
        "",
    ]

    if "status" in output_params:
        lines += [
            "## Calculation Status",
            "",
            generate_status_table(output_params["status"]),
            "",
        ]

    if "final_energy" in output_params and output_params["final_energy"]:
        lines += [
            "## Final Energy",
            "",
            generate_energy_table(output_params["final_energy"]),
            "",
        ]

    if "wall_time_seconds" in output_params and output_params["wall_time_seconds"]:
        lines += [
            "## Wall Time [s]",
            "",
            generate_wall_time_table(output_params["wall_time_seconds"]),
            "",
        ]

    lines += [render_report_footer()]

    return "\n".join(lines)
