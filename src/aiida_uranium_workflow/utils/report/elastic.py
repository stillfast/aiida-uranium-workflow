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


def generate_summary_table(output_params: Dict[str, Any], workflow_type: str) -> str:
    """Render bulk / shear moduli + derived properties."""
    method = output_params.get("method", "?")
    lines = [
        "| Property | Value |",
        "| --- | --- |",
        f"| backend | {workflow_type} |",
        f"| method | {method} |",
        f"| Bulk modulus K_VRH | {_fmt(output_params.get('bulk_modulus_gpa'))} GPa |",
        f"| Shear modulus G_VRH | {_fmt(output_params.get('shear_modulus_gpa'))} GPa |",
        f"| Young modulus Y_VRH | {_fmt(_young_modulus_gpa(output_params))} GPa |",
        f"| K_Voigt | {_fmt(output_params.get('bulk_modulus_voigt'))} GPa |",
        f"| K_Reuss | {_fmt(output_params.get('bulk_modulus_reuss'))} GPa |",
        f"| G_Voigt | {_fmt(output_params.get('shear_modulus_voigt'))} GPa |",
        f"| G_Reuss | {_fmt(output_params.get('shear_modulus_reuss'))} GPa |",
        f"| Universal anisotropy | {_fmt(output_params.get('universal_anisotropy'))} |",
        f"| Poisson ratio | {_fmt(output_params.get('poisson_ratio'))} |",
    ]
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
