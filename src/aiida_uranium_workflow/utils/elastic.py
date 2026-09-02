"""Elastic-constant utilities shared by the ABACUS / FLEUR elastic workflows.

Implements the Materials Project deformation methodology:

1. Generate deformed structures via pymatgen's ``DeformedStructureSet``
   (3 normal strains × ``norm_strains`` + 3 shear strains ×
   ``shear_strains`` = 24 structures).
2. For each deformed structure run a fixed-lattice SCF.
3. Fit the elastic tensor:
   * ABACUS / VASP — stress method: pymatgen
     ``ElasticTensor.from_independent_strains`` (needs the DFT stress
     tensor, which ABACUS prints in the ``TOTAL-STRESS`` block and VASP
     in OUTCAR / vasprun.xml; both report positive = compression).
   * FLEUR — energy method: the SCF total energy of each deformed
     structure is fit to the Voigt quadratic form
     ``E = E0 + V/2 · εᵀ·C·ε`` (FLEUR prints no stress tensor).

Units: stresses are converted to GPa (ABACUS and VASP print kbar;
1 kbar = 0.1 GPa); energies in eV; volumes in Å³.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: eV per Hartree.
HA_TO_EV = 27.211386245988
#: 1 kbar = 0.1 GPa (ABACUS stress unit).
KBAR_TO_GPA = 0.1
#: elementary charge (eV per J).
_EV_PER_J = 1.602176634e-19

#: Cubic space-group numbers (195–230) — used to enforce the cubic
#: symmetry (C11=C22=C33, C12=C13=C23, C44=C55=C66, cross terms = 0)
#: on the fitted tensor. Without this, stress/energy noise in the
#: individual strain states leaks into the "independent" constants
#: (e.g. C11 ≠ C33 or C45 ≠ 0 for a nominally cubic cell).
_CUBIC_SPACEGROUPS = set(range(195, 231))


def crystal_system_of(structure) -> str:
    """Return the crystal system ("cubic", …) of an AiiDA StructureData.

    Used by the elastic workflows to constrain the fitted tensor to the
    lattice symmetry (only the cubic case is implemented so far).
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pmg = structure_to_pymatgen(structure)
    try:
        number = SpacegroupAnalyzer(pmg).get_space_group_number()
    except Exception:
        return "unknown"
    if number in _CUBIC_SPACEGROUPS:
        return "cubic"
    return "other"


def cubic_symmetrize(c_voigt: np.ndarray) -> np.ndarray:
    """Average a 6×6 Voigt tensor onto the cubic manifold.

    Returns a tensor with ``C11=C22=C33``, ``C12=C13=C23``,
    ``C44=C55=C66`` and every normal-shear / shear-shear cross term
    zero. The three averages are over the three symmetry-equivalent
    entries, so noise in individual strain states is suppressed instead
    of being reported as a symmetry breaking.
    """
    c = np.asarray(c_voigt, dtype=float)
    c11 = (c[0, 0] + c[1, 1] + c[2, 2]) / 3.0
    c12 = (c[0, 1] + c[0, 2] + c[1, 2]) / 3.0
    c44 = (c[3, 3] + c[4, 4] + c[5, 5]) / 3.0
    out = np.zeros((6, 6))
    out[:3, :3] = c12
    out[0, 0] = out[1, 1] = out[2, 2] = c11
    out[3, 3] = out[4, 4] = out[5, 5] = c44
    return out


def structure_to_pymatgen(structure) -> Any:
    """Convert an AiiDA StructureData to a pymatgen Structure."""
    from pymatgen.core import Structure as PmgStructure

    return PmgStructure(
        lattice=structure.cell,
        species=[site.kind_name for site in structure.sites],
        coords=[site.position for site in structure.sites],
        coords_are_cartesian=True,
    )


def pymatgen_to_structure(pmg_structure) -> Any:
    """Convert a pymatgen Structure to an AiiDA StructureData.

    Uses the **cartesian** ``site.coords`` — ``append_atom(position=…)``
    expects cartesian coordinates, and feeding the fractional
    coordinates instead (``site.frac_coords``) piles every atom up
    inside the cell corner (~0–1 Å), which trips ABACUS's
    "atoms too close" check.
    """
    from aiida.orm import StructureData

    sd = StructureData(cell=pmg_structure.lattice.matrix, pbc=True)
    for site in pmg_structure:
        sd.append_atom(
            position=site.coords,
            symbols=[site.specie.symbol],
        )
    return sd


def generate_deformations(
    norm_strains: List[float],
    shear_strains: List[float],
    combined_strains: Optional[List[List[float]]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Return ``[(strain_voigt, deformation_matrix, label)]`` for every state.

    Mirrors pymatgen's ``DeformedStructureSet``: the 3 normal strains
    (ε₁ = ε_xx, ε₂ = ε_yy, ε₃ = ε_zz) take ``norm_strains``; the 3
    shear strains (ε₄ = 2ε_yz, ε₅ = 2ε_xz, ε₆ = 2ε_xy) take
    ``shear_strains``. Each ``deformation_matrix`` is the Green-Lagrange
    deformation gradient F (the lattice is deformed by F·a_i).

    ``combined_strains`` (optional) adds user-method combined modes:
    each entry is a 6-component Voigt template (e.g. ``(1, 1, 0, 0, 0,
    0)`` for the biaxial ε = (δ, δ, 0, 0, 0, 0)) and is applied at every
    ``norm_strains`` magnitude. These modes probe the off-diagonal
    elastic constants C12 / C13 / C23 (single normal/shear strains
    cannot).

    ``strain_voigt`` is the 6-component Voigt strain (shear components
    already doubled, matching ``ElasticTensor.from_independent_strains``).
    """
    from pymatgen.analysis.elasticity import Strain

    deformations: List[Tuple[np.ndarray, np.ndarray, str]] = []
    # Normal strains: (i,i) diagonal.
    for idx in [(0, 0), (1, 1), (2, 2)]:
        for amount in norm_strains:
            strain = Strain.from_index_amount(idx, amount)
            deformations.append(
                (np.asarray(strain.voigt), strain.get_deformation_matrix(), f"n{idx}")
            )
    # Shear strains: off-diagonal (i,j).
    for idx in [(0, 1), (0, 2), (1, 2)]:
        for amount in shear_strains:
            strain = Strain.from_index_amount(idx, amount)
            deformations.append(
                (np.asarray(strain.voigt), strain.get_deformation_matrix(), f"s{idx}")
            )
    # Combined strains: user-method biaxial / triaxial modes.
    for cidx, mode in enumerate(combined_strains or []):
        mode = np.asarray(mode, dtype=float)
        for amount in norm_strains:
            voigt = mode * amount
            strain = Strain.from_voigt(voigt)
            deformations.append(
                (np.asarray(voigt), strain.get_deformation_matrix(), f"c{cidx}")
            )
    return deformations


#: User-method combined-strain modes (Voigt templates, × magnitude).
#: Covers the orthorhombic biaxial set (ε7..ε9) and the cubic set
#: (ε1 triaxial, ε2 triple shear) — see the elastic-constants notes.
COMBINED_STRAIN_MODES: List[List[float]] = [
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # ε7 = (δ, δ, 0, …) → C11 + C22 + 2·C12
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # ε8 = (δ, 0, δ, …) → C11 + C33 + 2·C13
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # ε9 = (0, δ, δ, …) → C22 + C33 + 2·C23
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # cubic ε1 → (3·C11 + 6·C12)·δ²/2
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # cubic ε2 → (3·C44)·δ²/2
]


def apply_deformation_matrix(structure, deformation: np.ndarray):
    """Return a deformed AiiDA StructureData (lattice vectors F·a_i).

    A uniform strain keeps the **fractional** coordinates of every site
    fixed; the cartesian positions must be re-derived from the deformed
    cell. Keeping the raw cartesian coordinates instead (the old
    behaviour) displaced atoms off their ideal sites, introducing
    spurious internal displacements that corrupted the energy / stress
    response — for FLEUR shear strains this produced negative C44 and
    non-zero C45, and it biased the ABACUS stress-method constants too.
    """
    from aiida.orm import StructureData

    cell = np.asarray(structure.cell, dtype=float)
    new_cell = np.asarray(deformation, dtype=float) @ cell
    sd = StructureData(cell=new_cell, pbc=True)
    for site in structure.sites:
        frac = np.linalg.solve(cell.T, np.asarray(site.position, dtype=float))
        sd.append_atom(position=frac @ new_cell, symbols=[site.kind_name])
    return sd


# ---------------------------------------------------------------------------
# ABACUS — stress method
# ---------------------------------------------------------------------------


def fit_elastic_from_stress(
    strain_voigts: List[np.ndarray],
    stress_tensors_kbar: List[np.ndarray],
    symprec: float = 1e-3,
    crystal_system: Optional[str] = None,
    compression_positive: bool = True,
) -> Dict[str, Any]:
    """Fit the elastic tensor from (strain, stress) pairs (stress method).

    ``stress_tensors_kbar`` are the 3×3 stress tensors in kbar.
    **Sign convention**: ABACUS and VASP both report positive stress as
    *compression* (a positive hydrostatic stress drives the volume
    relax to expand; verified for VASP against real OUTCAR/vasprun
    output — a compressed cell shows positive stress), while pymatgen's
    ``Stress`` uses positive = *tension*. With the default
    ``compression_positive=True`` the tensor is therefore **negated**
    before the fit — the official ABACUS elastic example
    (``compute_dfm.py``) applies the same flip
    (``Stress(stress * (-1000))``), and pymatgen's own
    ``ElasticTensor.from_independent_strains`` negates VASP data via
    its ``vasp`` flag. Without it every fitted elastic constant comes
    out with the wrong sign (e.g. negative bulk moduli for a stable
    lattice). Only pass ``compression_positive=False`` for data that is
    already tension-positive.

    Stresses (kbar) are converted to GPa (1 kbar = 0.1 GPa) and passed
    to pymatgen's ``ElasticTensor.from_independent_strains``.

    ``crystal_system`` ("cubic" / "other" / None) optionally enforces
    the lattice symmetry on the fitted tensor (see
    :func:`cubic_symmetrize`) — without it, stress noise in the
    individual strain states shows up as a spurious symmetry breaking
    (C11 ≠ C33 or C45 ≠ 0 for a cubic cell).
    """
    from pymatgen.analysis.elasticity import ElasticTensor, Strain, Stress

    sign = -1.0 if compression_positive else 1.0
    strains = [Strain.from_voigt(sv) for sv in strain_voigts]
    stresses = [
        # ABACUS positive = compression → negate for the pymatgen
        # (positive = tension) convention; VASP is already tension
        # positive, so leave it untouched.
        Stress(sign * np.asarray(s, dtype=float) * KBAR_TO_GPA)
        for s in stress_tensors_kbar
    ]

    ct = ElasticTensor.from_independent_strains(strains, stresses)
    if crystal_system == "cubic":
        ct = ElasticTensor.from_voigt(cubic_symmetrize(ct.voigt))
    return _elastic_result(ct, "stress")


# ---------------------------------------------------------------------------
# FLEUR — energy method
# ---------------------------------------------------------------------------


def _fit_cubic_from_energy(
    strains: np.ndarray, energies_ev: np.ndarray, volume_ang3: float
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Fit the 3 cubic constants (C11, C12, C44) by per-family ±-symmetric
    differences, bypassing the global E0 extrapolation.

    Each strain family (ε1=(δ,0,0), ε4=(0,0,0,δ,0,0), the biaxial
    (δ,δ,0,0,0,0) and the triaxial (δ,δ,δ,0,0,0) modes) is fit
    independently to ``dE_sym(δ) = [E(+δ)+E(−δ)]/2 = a·δ²``::

        (δ,0,0,0,0,0)          →  C11  = 2a/V
        (0,0,0,δ,0,0)          →  C44  = 2a/V
        (δ,δ,0,0,0,0)          →  a/V  = C11 + C12
        (δ,δ,δ,0,0,0)          →  a/V  = (3C11 + 6C12)/2

    The linear (residual-stress) term cancels in the symmetric
    difference and no global E0 needs to be extrapolated, so each
    constant is determined by its own family only — a global 21-term
    least squares over inconsistent meV-level energies (as produced by
    FLEUR for a cubic cell) can otherwise flip the sign of C44 or
    invent non-zero C45.
    """
    a_ev = volume_ang3 * 1e-30 * 1e9 / _EV_PER_J  # eV per GPa per unit strain

    families: Dict[Tuple[int, ...], List[Tuple[np.ndarray, float]]] = {}
    for sv, e in zip(strains, energies_ev):
        nz = tuple(np.nonzero(np.abs(sv) > 1e-12)[0])
        families.setdefault(nz, []).append((sv, float(e)))

    def fam_coef(states: List[Tuple[np.ndarray, float]]) -> Optional[float]:
        """a (eV per δ²) from the symmetric part at two strain magnitudes."""
        pairs: Dict[float, List[Tuple[float, float]]] = {}
        for sv, e in states:
            vi = next(i for i, x in enumerate(sv) if abs(x) > 1e-12)
            pairs.setdefault(abs(sv[vi]), []).append((np.sign(sv[vi]), e))
        pts = []
        for m, se in sorted(pairs.items()):
            pos = [e for s, e in se if s > 0]
            neg = [e for s, e in se if s < 0]
            if pos and neg:
                sym = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2.0
                pts.append((m, sym))
        if len(pts) >= 2:
            (m1, s1), (m2, s2) = pts[0], pts[1]
            return (s2 - s1) / (m2 * m2 - m1 * m1)
        return None

    c11 = c44 = c12 = None
    for nz, states in families.items():
        a = fam_coef(states)
        if a is None:
            continue
        val = 2.0 * a / a_ev  # GPa carried by this family
        if nz in ((0,), (1,), (2,)):
            c11 = val if c11 is None else (c11 + val) / 2.0
        elif nz in ((3,), (4,), (5,)):
            c44 = val if c44 is None else (c44 + val) / 2.0
        elif nz in ((0, 1), (0, 2), (1, 2)) and c11 is not None:
            # a/V = C11 + C12  →  C12 = val/2 − C11
            cand = val / 2.0 - c11
            c12 = cand if c12 is None else (c12 + cand) / 2.0
        elif nz == (0, 1, 2) and c11 is not None:
            # 3C11 + 6C12 = val → C12 = (val − 3C11)/6
            cand = (val - 3.0 * c11) / 6.0
            c12 = cand if c12 is None else (c12 + cand) / 2.0

    missing = [
        name for name, v in (("C11", c11), ("C44", c44), ("C12", c12)) if v is None
    ]
    note = (
        None
        if not missing
        else f"cubic fit: missing strain families for {', '.join(missing)}"
    )
    if c11 is None:
        c11 = 0.0
    if c44 is None:
        c44 = 0.0
    if c12 is None:
        c12 = 0.0

    c = np.zeros((6, 6))
    c[:3, :3] = c12
    c[0, 0] = c[1, 1] = c[2, 2] = c11
    c[3, 3] = c[4, 4] = c[5, 5] = c44
    return c, None, note


def fit_elastic_from_energy(
    strain_voigts: List[np.ndarray],
    energies_ev: List[float],
    volume_ang3: float,
    crystal_system: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit the full 6×6 elastic tensor from (strain, total-energy) pairs.

    The SCF total energy of each deformed structure is fit to the Voigt
    quadratic form::

        E(ε) = E0 + V·σ₀·ε + (V/2) · εᵀ · C · ε

    with ``ε`` the 6-component Voigt strain (shear doubled), ``C`` in
    GPa and ``σ₀`` the residual stress of the reference structure (also
    GPa). A least-squares design matrix is built from the linear strain
    terms (6, absorbing any initial stress) and the quadratic strain
    products over the 21 independent elements of the symmetric tensor
    (E0 included as an extra unknown), so **combined strains** (e.g.
    ε1+ε2, ε1+ε2+ε3) automatically deliver the off-diagonal constants
    C12 / C13 / C23. Elements never probed by the strain set (e.g. C14
    when no mixed normal+shear state is included) come out zero.

    The linear (initial-stress) term matters whenever the strain set is
    not exactly symmetric about ε = 0 — e.g. combined modes applied at
    both signs still cancel in the pairwise average, but a set with
    only +δ combined states would otherwise bias every quadratic
    coefficient. For perfectly ±-symmetric sets the linear columns are
    orthogonal to the quadratic ones and the fit is unaffected.

    All columns are scaled into eV so that the least-squares system is
    well conditioned (the raw Pa-scaled columns span ~34 orders of
    magnitude) and the fitted coefficients come out directly in GPa.

    Returns the same dict shape as :func:`fit_elastic_from_stress`.
    """
    from pymatgen.analysis.elasticity import ElasticTensor

    strains = np.asarray(strain_voigts, dtype=float)  # (n, 6)
    energies = np.asarray(energies_ev, dtype=float)  # (n,) in eV

    if crystal_system == "cubic":
        c_voigt, sigma0_gpa, note = _fit_cubic_from_energy(
            strains, energies, volume_ang3
        )
        ct = ElasticTensor.from_voigt(c_voigt)
        result = _elastic_result(ct, "energy")
        result["energy_fit_e0_ev"] = float(np.mean(energies))
        result["diagonal_only"] = False
        result["note"] = note
        return result

    v_m3 = volume_ang3 * 1e-30
    #: eV per GPa per unit strain: V·σ = v_m3·(σ_GPa·1e9) [J] → /_EV_PER_J.
    a_ev = v_m3 * 1e9 / _EV_PER_J

    pairs = [(i, j) for i in range(6) for j in range(i, 6)]
    n = len(strains)
    # Unknowns: E0 + 6 linear (initial stress) + 21 quadratic C_ij.
    # The linear columns are dropped when the system would be
    # underdetermined (fewer states than 1 + 6 + 21).
    n_lin = 6 if n >= 1 + 6 + len(pairs) else 0
    design = np.zeros((n, 1 + n_lin + len(pairs)))
    design[:, 0] = 1.0  # E0 [eV]
    if n_lin:
        design[:, 1:7] = a_ev * strains  # linear term, coef → GPa
    for k, (i, j) in enumerate(pairs):
        factor = 1.0 if i == j else 2.0
        design[:, 1 + n_lin + k] = a_ev / 2.0 * factor * strains[:, i] * strains[:, j]

    coef, *_ = np.linalg.lstsq(design, energies, rcond=None)  # all in eV
    e0_ev = float(coef[0])

    c_voigt = np.zeros((6, 6))
    for k, (i, j) in enumerate(pairs):
        c_voigt[i, j] = c_voigt[j, i] = coef[1 + n_lin + k]  # already GPa
    ct = ElasticTensor.from_voigt(c_voigt)

    sigma0_gpa = np.asarray(coef[1:7]) if n_lin else None

    # Did the strain set probe any off-diagonal element? Without a
    # combined strain the corresponding design columns are all zero and
    # the fit returns 0 for them.
    probed_off_diag = any(
        i != j and np.any(np.abs(design[:, 1 + n_lin + k]) > 1e-30)
        for k, (i, j) in enumerate(pairs)
    )

    result = _elastic_result(ct, "energy")
    result["energy_fit_e0_ev"] = e0_ev
    if sigma0_gpa is not None:
        result["initial_stress_gpa"] = [float(s) for s in sigma0_gpa]
    result["diagonal_only"] = not probed_off_diag
    result["note"] = (
        None
        if probed_off_diag
        else "FLEUR energy method: the strain set contains no combined "
        "strains, so the off-diagonal constants (C12/C13/C23) are zero."
    )
    return result


def _elastic_result(ct, method: str) -> Dict[str, Any]:
    """Serialize an ElasticTensor into the report dict.

    Derived moduli (bulk / shear / anisotropy) are evaluated defensively:
    a tensor with zero elements (e.g. unprobed C14 in an energy-method
    fit) is singular, and pymatgen's compliance inversion then raises —
    such derived quantities are reported as ``None``.
    """
    c_voigt = np.asarray(ct.voigt, dtype=float)
    # The elastic tensor must be symmetric (Cij = Cji). pymatgen's
    # ``from_independent_strains`` least-squares fit does not enforce
    # this, and with a strain set that lacks normal-shear combinations
    # (ABACUS preset: pure normal + pure shear only) the coupling
    # elements C14/C15/C16/… are underdetermined — stress noise then
    # leaks into them asymmetrically (e.g. C14 = 0 but C41 ≈ 0.01 GPa).
    # Symmetrize the raw fit before deriving any moduli; the symmetric
    # part is unaffected (verified: K/G identical before/after).
    c_voigt = (c_voigt + c_voigt.T) / 2.0

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _safe(fn):
        try:
            return _f(fn())
        except Exception:
            return None

    return {
        "method": method,
        "elastic_tensor_gpa": c_voigt.tolist(),
        "bulk_modulus_gpa": _safe(lambda: ct.k_vrh),
        "shear_modulus_gpa": _safe(lambda: ct.g_vrh),
        # ``y_mod`` is returned in Pa (SI) while k_vrh / g_vrh come back
        # as GPa numbers — divide by 1e9 for a consistent GPa value.
        "young_modulus_gpa": _safe(lambda: ct.y_mod / 1e9),
        "bulk_modulus_voigt": _safe(lambda: ct.k_voigt),
        "bulk_modulus_reuss": _safe(lambda: ct.k_reuss),
        "shear_modulus_voigt": _safe(lambda: ct.g_voigt),
        "shear_modulus_reuss": _safe(lambda: ct.g_reuss),
        "universal_anisotropy": _safe(lambda: ct.universal_anisotropy),
        "poisson_ratio": _safe(lambda: ct.homogeneous_poisson),
    }
