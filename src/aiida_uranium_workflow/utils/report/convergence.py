"""Convergence WorkChain report generation module.

Extracts total energy data from VaspConvergenceWorkChain or AbacusConvergenceWorkChain
output_parameters and generates a Markdown report with tables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Tuple


def _get_kpoints_mode(output_params: Dict[str, Any]) -> str | None:
    """Determine kpoints mode from output_params.

    Returns one of: 'mesh', 'spacing', 'distance', or None if undetermined.
    Priority:
        1. Explicit 'kpoints_mode' field in output_params
        2. Presence of 'kpoints_mesh_list' / 'kpoints_list' field
    """
    mode = output_params.get("kpoints_mode")
    if mode:
        return str(mode)

    if "kpoints_mesh_list" in output_params or "kpoints_list" in output_params:
        return "mesh"

    return None


def _resolve_kpoints_label(workflow_type: str, mode: str | None) -> str:
    """Return the column header label for kpoints based on mode and workflow type."""
    if mode == "mesh":
        return "kpoints_mesh"
    if workflow_type == "vasp":
        return "kpoints_spacing (A^-1 * 2pi)"
    return "kpoints_distance (A^-1)"


def _sort_kpoints_values(kpoints_values: list) -> list:
    """Sort kpoints values with natural ordering for mesh strings.

    Mesh values are stored as strings like '11x11x11'. Plain string sort yields
    '11x11x11, 13x13x13, 5x5x5, ...' because '1' < '5'. We want the natural
    numeric ordering '5x5x5, 7x7x7, 9x9x9, 11x11x11, 13x13x13'.
    """
    if not kpoints_values:
        return kpoints_values

    def mesh_key(value):
        if isinstance(value, str):
            try:
                return tuple(int(part) for part in value.split("x"))
            except ValueError:
                return (value,)
        return value

    try:
        return sorted(kpoints_values, key=mesh_key)
    except TypeError:
        return sorted(kpoints_values)


def _parse_label(label: str) -> tuple[float | None, float | str | None]:
    """Parse a label into (ecut_value, kpoints_value).

    Supports both ABACUS and VASP label formats:
    - ABACUS (distance): 'ecutwfc_80_kpoints_distance_0_1'
    - ABACUS (mesh): 'ecutwfc_80_kpoints_11x11x11'
    - VASP (spacing): 'encut_300_kpoints_spacing_0_0159154943'
    - VASP (mesh): 'encut_300_kpoints_11x11x11'

    Returns:
        (ecut_value, kpoints_value) where kpoints_value is either a float
        (for spacing/distance mode) or a string (for mesh mode, e.g. '11x11x11').
    """
    parts = label.split("_")
    if len(parts) >= 4:
        try:
            ecut_keywords = ["ecutwfc", "encut"]

            ecut_idx = None
            for keyword in ecut_keywords:
                if keyword in parts:
                    ecut_idx = parts.index(keyword)
                    break

            kpoints_idx = None
            if "kpoints" in parts:
                kpoints_idx = parts.index("kpoints")

            if (
                ecut_idx is not None
                and kpoints_idx is not None
                and ecut_idx < kpoints_idx
            ):
                ecut_part = "_".join(parts[ecut_idx + 1 : kpoints_idx])
                ecut_value = float(ecut_part.replace("_", "."))

                if kpoints_idx + 1 < len(parts):
                    next_part = parts[kpoints_idx + 1]
                    if next_part in ["spacing", "distance"]:
                        kpoints_part = "_".join(parts[kpoints_idx + 2 :])
                        kpoints_value = float(kpoints_part.replace("_", "."))
                    else:
                        kpoints_part = "_".join(parts[kpoints_idx + 1 :])
                        kpoints_value = kpoints_part.replace("_", ".")
                else:
                    kpoints_value = None

                return ecut_value, kpoints_value
        except (ValueError, IndexError):
            pass
    return None, None


def generate_total_energy_table(
    total_energy: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate a Markdown table for total energy.

    Rows are ecut values, columns are kpoints values.
    """
    if not total_energy:
        return "No data available."

    ecut_values = sorted(
        set(
            _parse_label(k)[0]
            for k in total_energy.keys()
            if _parse_label(k)[0] is not None
        )
    )
    kpoints_values = _sort_kpoints_values(
        sorted(
            set(
                _parse_label(k)[1]
                for k in total_energy.keys()
                if _parse_label(k)[1] is not None
            )
        )
    )

    ecut_label = "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"
    mode = _get_kpoints_mode(output_params or {})
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label} \\ {kpoints_label} | "
        + " | ".join(f"{s}" for s in kpoints_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in kpoints_values) + " |"

    rows = []
    for ecut in ecut_values:
        row = [f"| {ecut}"]
        for kpoints in kpoints_values:
            found = False
            for label, value in total_energy.items():
                l_ecut, l_kpoints = _parse_label(label)
                if l_ecut == ecut and l_kpoints == kpoints:
                    row.append(f"| {value:.6f}")
                    found = True
                    break
            if not found:
                row.append("| ")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def generate_total_energy_per_atom_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate a Markdown table for total energy per atom.

    Rows are ecut values, columns are kpoints values.
    """
    if not total_energy_per_atom:
        return "No data available."

    ecut_values = sorted(
        set(
            _parse_label(k)[0]
            for k in total_energy_per_atom.keys()
            if _parse_label(k)[0] is not None
        )
    )
    kpoints_values = _sort_kpoints_values(
        sorted(
            set(
                _parse_label(k)[1]
                for k in total_energy_per_atom.keys()
                if _parse_label(k)[1] is not None
            )
        )
    )

    ecut_label = "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"
    mode = _get_kpoints_mode(output_params or {})
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label} \\ {kpoints_label} | "
        + " | ".join(f"{s}" for s in kpoints_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in kpoints_values) + " |"

    rows = []
    for ecut in ecut_values:
        row = [f"| {ecut}"]
        for kpoints in kpoints_values:
            found = False
            for label, value in total_energy_per_atom.items():
                l_ecut, l_kpoints = _parse_label(label)
                if l_ecut == ecut and l_kpoints == kpoints:
                    row.append(f"| {value:.8f}")
                    found = True
                    break
            if not found:
                row.append("| ")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def generate_encut_convergence_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate a Markdown table showing energy difference when increasing encut.

    For each kpoints_spacing, shows the energy difference (meV/atom) between
    consecutive encut values (current - previous). The first row shows '-'
    since there's no previous encut to compare.
    """
    if not total_energy_per_atom:
        return "No data available."

    ecut_values = sorted(
        set(
            _parse_label(k)[0]
            for k in total_energy_per_atom.keys()
            if _parse_label(k)[0] is not None
        )
    )
    kpoints_values = _sort_kpoints_values(
        sorted(
            set(
                _parse_label(k)[1]
                for k in total_energy_per_atom.keys()
                if _parse_label(k)[1] is not None
            )
        )
    )

    ecut_label = "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"
    mode = _get_kpoints_mode(output_params or {})
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label} \\ {kpoints_label} | "
        + " | ".join(f"{s}" for s in kpoints_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in kpoints_values) + " |"

    rows = []
    for idx, ecut in enumerate(ecut_values):
        row = [f"| {ecut}"]
        for kpoints in kpoints_values:
            if idx == 0:
                row.append("| -")
            else:
                prev_ecut = ecut_values[idx - 1]
                curr_value = None
                prev_value = None
                for label, value in total_energy_per_atom.items():
                    l_ecut, l_kpoints = _parse_label(label)
                    if l_ecut == ecut and l_kpoints == kpoints:
                        curr_value = value
                    if l_ecut == prev_ecut and l_kpoints == kpoints:
                        prev_value = value
                if curr_value is not None and prev_value is not None:
                    diff = (curr_value - prev_value) * 1000
                    row.append(f"| {diff:+.4f}")
                else:
                    row.append("| N/A")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def generate_kpoints_convergence_table(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate a Markdown table showing energy difference when decreasing kpoints density.

    For each encut, shows the energy difference (meV/atom) between consecutive
    kpoints values (current - next, i.e., higher density - lower density).
    The last column shows '-' since there's no lower density kpoints to compare.
    """
    if not total_energy_per_atom:
        return "No data available."

    ecut_values = sorted(
        set(
            _parse_label(k)[0]
            for k in total_energy_per_atom.keys()
            if _parse_label(k)[0] is not None
        )
    )
    kpoints_values = _sort_kpoints_values(
        sorted(
            set(
                _parse_label(k)[1]
                for k in total_energy_per_atom.keys()
                if _parse_label(k)[1] is not None
            )
        )
    )
    mode = _get_kpoints_mode(output_params or {})
    # Display columns from highest density (most converged) to lowest so that
    # ``diff = current - next`` reads as ``E(higher density) - E(lower density)``.
    if mode == "mesh":
        kpoints_values = list(reversed(kpoints_values))

    ecut_label = "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label} \\ {kpoints_label} | "
        + " | ".join(f"{s}" for s in kpoints_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in kpoints_values) + " |"

    rows = []
    for ecut in ecut_values:
        row = [f"| {ecut}"]
        for idx, kpoints in enumerate(kpoints_values):
            if idx == len(kpoints_values) - 1:
                row.append("| -")
            else:
                next_kpoints = kpoints_values[idx + 1]
                curr_value = None
                next_value = None
                for label, value in total_energy_per_atom.items():
                    l_ecut, l_kpoints = _parse_label(label)
                    if l_ecut == ecut and l_kpoints == kpoints:
                        curr_value = value
                    if l_ecut == ecut and l_kpoints == next_kpoints:
                        next_value = value
                if curr_value is not None and next_value is not None:
                    diff = (curr_value - next_value) * 1000
                    row.append(f"| {diff:+.4f}")
                else:
                    row.append("| N/A")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Generate a summary table with basic info."""
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    mode = _get_kpoints_mode(output_params)
    if mode:
        lines.append(f"| kpoints_mode | {mode} |")

    if "total_energy" in output_params:
        lines.append(f"| total_energy entries | {len(output_params['total_energy'])} |")

    if "num_atoms" in output_params and output_params["num_atoms"]:
        first_key = next(iter(output_params["num_atoms"].keys()))
        lines.append(f"| num_atoms | {output_params['num_atoms'][first_key]} |")

    return "\n".join(lines)


def generate_status_table(
    status: Dict[str, int],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate a status table showing each child's exit code.

    Each entry maps convergence parameters to the exit code
    reported by ``verdi process status``: ``0`` means the calculation
    finished successfully, anything else represents the corresponding
    AiiDA exit status (e.g. ``300`` for ``ERROR_CHILD``).
    """
    if not status:
        return "No status data available."

    ecut_values = sorted(
        set(_parse_label(k)[0] for k in status.keys() if _parse_label(k)[0] is not None)
    )
    kpoints_values = _sort_kpoints_values(
        sorted(
            set(_parse_label(k)[1] for k in status.keys() if _parse_label(k)[1] is not None)
        )
    )

    ecut_label = "encut (eV)" if workflow_type == "vasp" else "ecutwfc (Ry)"
    mode = _get_kpoints_mode(output_params or {})
    kpoints_label = _resolve_kpoints_label(workflow_type, mode)

    header = (
        f"| {ecut_label} \\ {kpoints_label} | "
        + " | ".join(f"{s}" for s in kpoints_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in kpoints_values) + " |"

    rows = []
    for ecut in ecut_values:
        row = [f"| {ecut}"]
        for kpoints in kpoints_values:
            found = False
            for label, exit_code in status.items():
                l_ecut, l_kpoints = _parse_label(label)
                if l_ecut == ecut and l_kpoints == kpoints:
                    row.append(f"| {exit_code}")
                    found = True
                    break
            if not found:
                row.append("| -")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def find_converged_parameters(
    total_energy_per_atom: Dict[str, float],
    energy_threshold: float = 1e-5,
) -> Dict[str, tuple[float, float | str] | None]:
    """Find converged ecutwfc and kpoints parameters.

    Recommendation algorithm
    ------------------------
    Iterating over ascending ecutwfc and kpoints values, we identify
    the smallest parameter set where the energy difference between consecutive
    values is below the threshold.

    Args:
        total_energy_per_atom: Dict of label -> total energy per atom values
        energy_threshold: Maximum allowed energy difference (default: 1e-5 Ry/atom)

    Returns:
        Dict with 'ecutwfc' and 'kpoints' keys mapping to the recommended
        values, or None if convergence is not achieved.
        kpoints value can be a float (spacing/distance) or string (mesh).
    """
    ecutwfc_data: Dict[float, list[tuple[float | str, float]]] = {}

    for label, value in total_energy_per_atom.items():
        if value is None or isinstance(value, str):
            continue
        ecutwfc, kpoints_val = _parse_label(label)
        if ecutwfc is not None and kpoints_val is not None:
            ecutwfc_data.setdefault(ecutwfc, []).append((kpoints_val, value))

    converged_ecutwfc: float | None = None
    converged_kpoints: float | str | None = None

    if ecutwfc_data:
        sorted_ecutwfcs = sorted(ecutwfc_data.keys())

        for i in range(1, len(sorted_ecutwfcs)):
            prev_ecutwfc = sorted_ecutwfcs[i - 1]
            curr_ecutwfc = sorted_ecutwfcs[i]

            prev_data = sorted(ecutwfc_data[prev_ecutwfc], key=lambda x: x[0])
            curr_data = sorted(ecutwfc_data[curr_ecutwfc], key=lambda x: x[0])

            if prev_data and curr_data:
                for prev_k, prev_energy in prev_data:
                    for curr_k, curr_energy in curr_data:
                        if prev_k == curr_k:
                            diff = abs(curr_energy - prev_energy)
                            if diff < energy_threshold:
                                converged_ecutwfc = curr_ecutwfc
                                converged_kpoints = curr_k
                                break
                    if converged_ecutwfc is not None:
                        break
            if converged_ecutwfc is not None:
                break

    return {
        "ecutwfc": converged_ecutwfc,
        "kpoints": converged_kpoints,
    }


def generate_converged_section(
    total_energy_per_atom: Dict[str, float],
    workflow_type: str = "abacus",
    output_params: Dict[str, Any] | None = None,
) -> str:
    """Generate the converged parameters recommendation section."""
    lines = []
    lines.append("## Recommended Converged Parameters")
    lines.append("")

    ecut_label = "encut" if workflow_type == "vasp" else "ecutwfc"
    ecut_unit = "eV" if workflow_type == "vasp" else "Ry"

    converged = find_converged_parameters(total_energy_per_atom)
    kpoints_val = converged.get("kpoints")

    mode = _get_kpoints_mode(output_params or {})
    if mode == "mesh" or isinstance(kpoints_val, str):
        kpoints_label = "kpoints_mesh"
        kpoints_unit = ""
    elif workflow_type == "vasp":
        kpoints_label = "kpoints_spacing"
        kpoints_unit = " A^-1 * 2pi"
    else:
        kpoints_label = "kpoints_distance"
        kpoints_unit = " A^-1"

    lines.append(
        f"This section recommends the smallest {ecut_label} and {kpoints_label} "
        f"where the total energy difference between consecutive values is < 1e-5 {ecut_unit}/atom."
    )
    lines.append("")

    if not converged["ecutwfc"] or not kpoints_val:
        lines.append("No convergence achieved within the tested parameter range.")
        return "\n".join(lines)

    lines.append("| Parameter | Recommended Value |")
    lines.append("| --- | --- |")
    lines.append(f"| {ecut_label} | {converged['ecutwfc']} {ecut_unit} |")
    lines.append(f"| {kpoints_label} | {kpoints_val}{kpoints_unit} |")

    return "\n".join(lines)


def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report.

    Args:
        output_params: The output_parameters dict from the WorkChain
        pk: The WorkChain pk
        workflow_type: 'vasp' or 'abacus'

    Returns:
        Markdown report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = []

    report_lines.append(f"# Convergence WorkChain Report (PK: {pk})")
    report_lines.append("")
    report_lines.append(f"**Workflow Type**: {workflow_type.upper()}")
    report_lines.append(f"**Generated**: {timestamp}")
    report_lines.append("")

    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(generate_summary_table(output_params))
    report_lines.append("")

    if "status" in output_params:
        report_lines.append("## Calculation Status")
        report_lines.append("")
        report_lines.append(
            generate_status_table(
                output_params["status"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

    energy_unit = "eV" if workflow_type == "vasp" else "Ry"

    if "total_energy" in output_params:
        report_lines.append(f"## Total Energy ({energy_unit})")
        report_lines.append("")
        report_lines.append(
            generate_total_energy_table(
                output_params["total_energy"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

    if "total_energy_per_atom" in output_params:
        report_lines.append(f"## Total Energy per Atom ({energy_unit}/atom)")
        report_lines.append("")
        report_lines.append(
            generate_total_energy_per_atom_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

    if (
        "total_energy_per_atom" in output_params
        and output_params["total_energy_per_atom"]
    ):
        report_lines.append("## Encut Convergence (meV/atom)")
        report_lines.append("")
        report_lines.append(
            generate_encut_convergence_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

        report_lines.append("## Kpoints Convergence (meV/atom)")
        report_lines.append("")
        report_lines.append(
            generate_kpoints_convergence_table(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

        report_lines.append(
            generate_converged_section(
                output_params["total_energy_per_atom"],
                workflow_type=workflow_type,
                output_params=output_params,
            )
        )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("*Generated by aiida-uranium-workflow*")

    return "\n".join(report_lines)
