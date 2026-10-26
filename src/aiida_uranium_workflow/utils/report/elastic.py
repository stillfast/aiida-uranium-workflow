"""Elastic WorkChain report generation module.

Renders the elastic tensor (GPa) and derived polycrystalline moduli
from ``AbacusElasticWorkChain`` / ``FleurElasticWorkChain``
``output_parameters``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ._common import (
    render_report_footer,
    render_report_header,
)

#: Voigt index labels for the 6x6 tensor display. Standard Voigt
#: numbering: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy (pymatgen / Nye
#: convention; shear strain components are doubled, 2ε_yz etc.).
_VOIGT_LABELS = ["1", "2", "3", "4", "5", "6"]


def _fmt(value: Any, prec: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


def _symmetrized(tensor_gpa: Any) -> Any:
    """Return the symmetrized 6×6 tensor for display.

    The elastic tensor must satisfy Cij = Cji, but legacy fits (before
    the symmetrization fix in ``utils.elastic``) can carry asymmetric
    noise in the underdetermined normal-shear coupling elements
    (e.g. C14 = 0 while C41 ≈ 0.01 GPa). Symmetrizing at display time
    is idempotent and makes old reports consistent with new ones.
    """
    try:
        arr = np.asarray(list(tensor_gpa), dtype=float)
        if arr.shape == (6, 6):
            return ((arr + arr.T) / 2.0).tolist()
    except (TypeError, ValueError):
        pass
    return tensor_gpa


def generate_tensor_table(tensor_gpa: Any) -> str:
    """Render the 6×6 Voigt elastic tensor as a Markdown table."""
    if not tensor_gpa:
        return "No elastic tensor available."
    try:
        c = _symmetrized(tensor_gpa)
        rows = list(c)
    except TypeError:
        return "No elastic tensor available."

    lines = [
        "| Cij (GPa) | " + " | ".join(_VOIGT_LABELS) + " |",
        "| --- |" + " --- |" * 6,
    ]
    for i, row in enumerate(rows):
        cells = " | ".join(_fmt(v) for v in row)
        lines.append(f"| {_VOIGT_LABELS[i]} | {cells} |")
    return "\n".join(lines)


def _young_modulus_gpa(output_params: Dict[str, Any]):
    """Return the VRH Young modulus in GPa.

    Prefers the workchain-stored ``young_modulus_gpa``; falls back to
    deriving it from the stored VRH bulk / shear moduli
    (``Y = 9KG/(3K+G)``) so reports for workchains run before the key
    existed still show it.
    """
    young = output_params.get("young_modulus_gpa")
    if young is not None:
        return young
    k = output_params.get("bulk_modulus_gpa")
    g = output_params.get("shear_modulus_gpa")
    if k is not None and g is not None:
        try:
            return 9.0 * float(k) * float(g) / (3.0 * float(k) + float(g))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def stability_report(output_params: Dict[str, Any]):
    """Return ``(stable, min_eigenvalue, cubic_born)`` for the stored tensor.

    Prefers the values recorded by the workchain (``stable`` /
    ``min_eigenvalue_gpa`` / ``born_cubic``); recomputes them from
    ``elastic_tensor_gpa`` for nodes created before those keys existed, so
    old reports get the same verdict.
    """
    if "stable" in output_params:
        return (
            bool(output_params["stable"]),
            output_params.get("min_eigenvalue_gpa"),
            output_params.get("born_cubic") or {},
        )

    tensor = output_params.get("elastic_tensor_gpa")
    if not tensor:
        return None, None, {}
    import numpy as np

    try:
        c = np.asarray(tensor, dtype=float)
        c = (c + c.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(c)
    except (TypeError, ValueError):
        return None, None, {}
    min_eigenvalue = float(eigenvalues.min())
    born = {}
    c11, c12, c44 = float(c[0, 0]), float(c[0, 1]), float(c[3, 3])
    if (
        abs(c[0, 0] - c[1, 1]) < 1e-3 * (abs(c11) + 1.0)
        and abs(c[0, 0] - c[2, 2]) < 1e-3 * (abs(c11) + 1.0)
        and abs(c[0, 1] - c[1, 2]) < 1e-3 * (abs(c12) + 1.0)
        and abs(c[0, 1] - c[0, 2]) < 1e-3 * (abs(c12) + 1.0)
        and abs(c[3, 3] - c[4, 4]) < 1e-3 * (abs(c44) + 1.0)
    ):
        born = {
            "C11_minus_C12": c11 - c12,
            "C11_plus_2C12": c11 + 2.0 * c12,
            "C44": c44,
        }
    return bool(min_eigenvalue > 1e-6), min_eigenvalue, born


def generate_summary_table(output_params: Dict[str, Any], workflow_type: str) -> str:
    """Render bulk / shear moduli + derived properties."""
    method = output_params.get("method", "?")
    stable, min_eigenvalue, cubic_born = stability_report(output_params)
    # ``flag`` marks the rows that are meaningless for an unstable tensor.
    flag = "" if stable is not False else " ⚠️"
    unstable_note = ""
    if stable is False:
        detail = ""
        if cubic_born:
            detail = (
                f" (C11−C12 = {_fmt(cubic_born.get('C11_minus_C12'))} GPa, "
                f"C44 = {_fmt(cubic_born.get('C44'))} GPa, "
                f"C11+2C12 = {_fmt(cubic_born.get('C11_plus_2C12'))} GPa)"
            )
        unstable_note = (
            f"| stability | **mechanically UNSTABLE** — the elastic tensor is "
            f"not positive definite (min eigenvalue {_fmt(min_eigenvalue)} GPa)"
            f"{detail}; the Born criterion C' = (C11−C12)/2 > 0 is violated, "
            "so the Reuss shear modulus (4(S11−S12)+3S44 ≈ 0) and everything "
            "derived from it (G_VRH, Y_VRH, Poisson ratio, anisotropy) are "
            "not physically meaningful — report the Cij only, or re-evaluate "
            "at a (meta)stable structure |"
        )

    lines = [
        "| Property | Value |",
        "| --- | --- |",
        f"| backend | {workflow_type} |",
        f"| method | {method} |",
        f"| Bulk modulus K_VRH | {_fmt(output_params.get('bulk_modulus_gpa'))} GPa |",
        f"| Shear modulus G_VRH{flag} | {_fmt(output_params.get('shear_modulus_gpa'))} GPa |",
        f"| Young modulus Y_VRH{flag} | {_fmt(_young_modulus_gpa(output_params))} GPa |",
        f"| K_Voigt | {_fmt(output_params.get('bulk_modulus_voigt'))} GPa |",
        f"| K_Reuss | {_fmt(output_params.get('bulk_modulus_reuss'))} GPa |",
        f"| G_Voigt{flag} | {_fmt(output_params.get('shear_modulus_voigt'))} GPa |",
        f"| G_Reuss{flag} | {_fmt(output_params.get('shear_modulus_reuss'))} GPa |",
        f"| Universal anisotropy{flag} | {_fmt(output_params.get('universal_anisotropy'))} |",
        f"| Poisson ratio{flag} | {_fmt(output_params.get('poisson_ratio'))} |",
    ]
    if unstable_note:
        lines.append(unstable_note)
    source = output_params.get("equilibrium_structure_source")
    if source is not None:
        label = {
            "lattice_relax": "relaxed (lattice_relax)",
            "input": "INPUT cell (not relaxed)",
        }.get(source, source)
        lines.append(f"| deformed lattice | {label} |")
        if source == "input":
            lines.append(
                "| NOTE | the strains were applied to the INPUT cell — the "
                "lattice pre-relaxation was skipped or failed, so the "
                "constants are not evaluated at the equilibrium lattice |"
            )
    if output_params.get("diagonal_only"):
        lines.append(
            "| NOTE | diagonal-only (energy method; off-diagonal needs "
            "combined strains) |"
        )
    return "\n".join(lines)


def generate_report(output_params: Dict[str, Any], pk: int, workflow_type: str) -> str:
    """Generate a complete Markdown report for an elastic WorkChain."""
    lines: List[str] = [
        render_report_header(
            title="Elastic Constants Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Summary",
        "",
        generate_summary_table(output_params, workflow_type),
        "",
        "## Elastic Tensor (Voigt, GPa)",
        "",
        generate_tensor_table(output_params.get("elastic_tensor_gpa")),
        "",
        "> Voigt order: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy "
        "(shear strain components doubled).",
        "",
    ]
    note = output_params.get("note")
    if note:
        lines += ["## Note", "", note, ""]
    lines += [render_report_footer()]
    return "\n".join(lines)
