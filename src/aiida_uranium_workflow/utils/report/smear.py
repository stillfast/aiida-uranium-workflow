"""Smear WorkChain report generation module.

Extracts electronic entropy data from VaspSmearWorkChain or AbacusSmearWorkChain
output_parameters and generates a Markdown report with tables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Tuple


def _parse_label(label: str) -> tuple[str, float | None]:
    """Parse a label like 'smear_1_sigma_0_05' into (smear, sigma)."""
    parts = label.split("_")
    if len(parts) >= 4:
        smear_part = "_".join(parts[1 : parts.index("sigma")])
        sigma_part = "_".join(parts[parts.index("sigma") + 1 :])
        try:
            sigma = float(sigma_part.replace("_", "."))
        except ValueError:
            sigma = None
        return smear_part, sigma
    return "", None


def generate_eentropy_table(
    eentropy_per_atom: Dict[str, float], sigma_unit: str = "ev"
) -> str:
    """Generate a Markdown table for eentropy per atom.

    Rows are smear methods, columns are sigma values.  ``sigma_unit``
    controls the unit annotation in the table header: ``"ry"`` for
    ABACUS (sigma in Rydberg), ``"ev"`` for VASP.
    """
    if not eentropy_per_atom:
        return "No data available."

    smear_values = sorted(set(_parse_label(k)[0] for k in eentropy_per_atom.keys()))
    sigma_values = sorted(set(_parse_label(k)[1] for k in eentropy_per_atom.keys()))

    unit_label = "Ry" if sigma_unit == "ry" else "eV"
    header = (
        f"| smearing_method\\smearing_sigma [{unit_label}] | "
        + " | ".join(f"{s}" for s in sigma_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in sigma_values) + " |"

    rows = []
    for smear in smear_values:
        row = [f"| {smear}"]
        for sigma in sigma_values:
            found = False
            for label, value in eentropy_per_atom.items():
                l_smear, l_sigma = _parse_label(label)
                if l_smear == smear and l_sigma == sigma:
                    row.append(f"| {value:.6f}")
                    found = True
                    break
            if not found:
                row.append("| ")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


def generate_summary_table(output_params: Dict[str, Any]) -> str:
    """Generate a summary table with basic info."""
    lines = ["| Property | Value |", "| --- | --- |"]

    if "dry_run" in output_params:
        lines.append(f"| dry_run | {output_params['dry_run']} |")

    if "smear_pairs" in output_params:
        lines.append(f"| smear_pairs | {len(output_params['smear_pairs'])} |")

    if "eentropy" in output_params:
        lines.append(f"| eentropy entries | {len(output_params['eentropy'])} |")

    if "num_atoms" in output_params and output_params["num_atoms"]:
        first_key = next(iter(output_params["num_atoms"].keys()))
        lines.append(f"| num_atoms | {output_params['num_atoms'][first_key]} |")

    return "\n".join(lines)


def generate_status_table(status: Dict[str, int], sigma_unit: str = "ev") -> str:
    """Generate a status table showing each child's exit code.

    Each entry maps ``Smear_{method}&Sigma_{sigma}`` to the exit code
    reported by ``verdi process status``: ``0`` means the calculation
    finished successfully, anything else represents the corresponding
    AiiDA exit status (e.g. ``300`` for ``ERROR_CHILD``).

    ``sigma_unit`` is used to annotate the table header with the
    matching unit (``"ry"`` for ABACUS, ``"ev"`` for VASP).
    """
    if not status:
        return "No status data available."

    smear_values = sorted(set(_parse_label(k)[0] for k in status.keys()))
    sigma_values = sorted(set(_parse_label(k)[1] for k in status.keys()))

    unit_label = "Ry" if sigma_unit == "ry" else "eV"
    header = (
        f"| smearing_method\\smearing_sigma [{unit_label}] | "
        + " | ".join(f"{s}" for s in sigma_values)
        + " |"
    )
    separator = "| --- | " + " | ".join("---" for _ in sigma_values) + " |"

    rows = []
    for smear in smear_values:
        row = [f"| {smear}"]
        for sigma in sigma_values:
            found = False
            for label, exit_code in status.items():
                l_smear, l_sigma = _parse_label(label)
                if l_smear == smear and l_sigma == sigma:
                    row.append(f"| {exit_code}")
                    found = True
                    break
            if not found:
                row.append("| -")
        row.append("|")
        rows.append(" ".join(row))

    return "\n".join([header, separator] + rows)


RY_TO_EV = 13.605693


def find_optimal_sigma(
    eentropy_per_atom: Dict[str, float],
    threshold: float = 0.001,
    sigma_unit: str = "ev",
) -> Dict[str, float | None]:
    """Find the optimal sigma for each smear method.

    According to VASP Wiki: SIGMA should be as large as possible while keeping
    the difference between free energy and total energy (entropy T*S term)
    negligible (< 1 meV/atom = 0.001 eV/atom).

    Recommendation algorithm
    ------------------------
    Iterating over ascending ``sigma`` values, we mark a sigma as the
    candidate *iff* its per-atom entropy is below ``threshold`` AND
    the delta against the previous sigma is below ``threshold`` as
    well.  If no sigma satisfies this — for instance ``smear=gauss``
    where the entropy is always well above 1 meV/atom — the
    recommendation for that smear method is ``None``.

    The function returns sigma values **in the input unit**; the
    caller is responsible for any unit formatting in the rendered
    table.  ``abacus`` reports sigma in Rydberg, ``vasp`` in eV.

    Args:
        eentropy_per_atom: Dict of label -> eentropy per atom values
        threshold: Maximum allowed eentropy (default: 0.001 eV/atom)
        sigma_unit: Unit of sigma values, 'ev' or 'ry' (Rydberg).
            Reported back to the caller as-is.

    Returns:
        Dict of smear_method -> optimal sigma value (None when the
        criterion is never met)
    """
    del sigma_unit  # kept for API compatibility; sigma values are
    # returned in the input unit, no implicit conversion happens here.
    smear_data: Dict[str, list[tuple[float, float]]] = {}

    for label, value in eentropy_per_atom.items():
        if value is None or isinstance(value, str):
            continue
        smear, sigma = _parse_label(label)
        if sigma is not None:
            smear_data.setdefault(smear, []).append((sigma, value))

    optimal_sigma: Dict[str, float | None] = {}

    for smear, data in smear_data.items():
        if not data:
            optimal_sigma[smear] = None
            continue

        data.sort(key=lambda x: x[0])

        accepted: float | None = None
        prev_value: float | None = None
        for sigma, eentropy in data:
            per_atom = abs(eentropy)
            if prev_value is None:
                prev_value = per_atom
                continue
            diff = abs(per_atom - prev_value)
            if diff < threshold and per_atom < threshold:
                accepted = sigma
            prev_value = per_atom

        optimal_sigma[smear] = round(accepted, 2) if accepted is not None else None

    return optimal_sigma


def generate_optimal_sigma_section(
    eentropy_per_atom: Dict[str, float], sigma_unit: str = "ev"
) -> str:
    """Generate the optimal sigma recommendation section.

    The unit annotation in the table header reflects ``sigma_unit``:
    ``"ry"`` for ABACUS (sigma is in Rydberg), ``"ev"`` for VASP.
    """
    lines = []
    lines.append("## Recommended Sigma Values")
    lines.append("")
    lines.append(
        "According to the VASP Wiki Smearing technique documentation: "
        "SIGMA should be as large as possible while keeping the entropy T*S term "
        "negligible (< 1 meV/atom = 0.001 eV/atom)."
    )
    lines.append("")
    lines.append(
        "This section recommends the largest sigma for each smear method "
        "where the eentropy change from the smallest sigma is < 0.001 eV/atom "
        "AND the absolute eentropy value < 0.001 eV/atom."
    )
    lines.append("")

    optimal = find_optimal_sigma(eentropy_per_atom, sigma_unit=sigma_unit)

    unit_label = "Ry" if sigma_unit == "ry" else "eV"

    if not optimal:
        lines.append("No valid data available for recommendation.")
        return "\n".join(lines)

    lines.append(f"| smearing_method | recommended_sigma ({unit_label}) |")
    lines.append("| --- | --- |")

    for smear in sorted(optimal.keys()):
        sigma = optimal[smear]
        lines.append(f"| {smear} | {sigma} |")

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

    # ABACUS smearing_sigma is in Rydberg; VASP's SIGMA is in eV.
    sigma_unit = "ry" if workflow_type == "abacus" else "ev"

    report_lines = []

    report_lines.append(f"# Smear WorkChain Report (PK: {pk})")
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
                sigma_unit=sigma_unit,
            )
        )
        report_lines.append("")

    if "eentropy_per_atom" in output_params:
        report_lines.append("## Electronic Entropy per Atom (eV/atom)")
        report_lines.append("")
        report_lines.append(
            generate_eentropy_table(
                output_params["eentropy_per_atom"],
                sigma_unit=sigma_unit,
            )
        )
        report_lines.append("")

    if "eentropy" in output_params:
        report_lines.append("## Electronic Entropy (eV)")
        report_lines.append("")
        report_lines.append(
            generate_eentropy_table(
                output_params["eentropy"],
                sigma_unit=sigma_unit,
            )
        )
        report_lines.append("")

    if "eentropy_per_atom" in output_params and output_params["eentropy_per_atom"]:
        report_lines.append(
            generate_optimal_sigma_section(
                output_params["eentropy_per_atom"], sigma_unit=sigma_unit
            )
        )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("*Generated by aiida-uranium-workflow*")

    return "\n".join(report_lines)
