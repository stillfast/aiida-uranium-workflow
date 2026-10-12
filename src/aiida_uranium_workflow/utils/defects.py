"""Defect-structure utilities shared by the ABACUS / FLEUR defect workflows.

Structure generation follows the standard defect methodology (same as
aiida-defects / pymatgen):

1. ``make_supercell`` — scale the (primitive) cell via pymatgen's
   ``make_supercell`` (the supercell is reduced back to the primitive
   cell of the super-lattice).
2. ``create_vacancy`` — remove the atom at ``site_index`` of the
   supercell (a vacancy).
3. ``create_interstitial`` — insert an atom of ``element`` at a
   fractional coordinate inside the supercell (an interstitial).

The neutral formation energy follows aiida-defects::

    E_f = E_defect − E_host − Σ_i n_i·μ_i

with ``n_i`` the atom-count change of species ``i`` (+1 inserted,
−1 removed) and ``μ_i`` its chemical potential (eV, per-atom reference
energy; 0 when not supplied).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def make_supercell(structure, supercell_matrix: List[int]):
    """Return the supercell of ``structure`` as an AiiDA StructureData.

    ``supercell_matrix`` is the [2, 2, 2]-style diagonal repetition
    (``pymatgen.make_supercell`` accepts it directly).
    """
    from aiida.orm import StructureData
    from pymatgen.core import Structure as PmgStructure

    pmg = PmgStructure(
        lattice=structure.cell,
        species=[site.kind_name for site in structure.sites],
        coords=[site.position for site in structure.sites],
        coords_are_cartesian=True,
    )
    pmg.make_supercell(list(supercell_matrix))
    return StructureData(pymatgen=pmg)


def _is_diagonal_triple(value) -> bool:
    """``[n, n, n]`` — the shorthand for a diagonal supercell."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
        and value[0] == value[1] == value[2]
    )


def normalize_supercell_matrices(spec) -> List[list]:
    """Normalise a preset ``supercell_matrix`` / ``supercell_matrices`` value.

    Accepted forms::

        None                                  -> []
        [2, 2, 2]                             -> [[2, 2, 2]]        (one cell)
        [[2,0,0], [0,2,0], [0,0,2]]           -> [that 3x3 matrix]  (one cell)
        [[2,2,2], [3,3,3]]                    -> two diagonal cells
        [[[2,0,0],[0,2,0],[0,0,2]], [3,3,3]]  -> a 3x3 and a diagonal cell
        [[2,2,2], [3,3,3], [4,4,4]]           -> three diagonal cells
          (three ``[n, n, n]`` rows are read as a *list*; to pass a real
           3x3 matrix use ``supercell_matrices: [[[...]]]`` or write it with
           non-uniform rows)

    Returns a list of matrices, each either a 3-element diagonal shorthand or
    a full 3x3 nested list (both accepted by ``pymatgen.make_supercell``).
    """
    if spec is None:
        return []
    if isinstance(spec, (str, bytes)) or not isinstance(spec, (list, tuple)):
        raise TypeError(
            f"supercell matrix must be a list, got {type(spec).__name__}"
        )
    items = list(spec)
    if not items:
        return []

    # flat [a, b, c] — a single diagonal supercell
    if all(isinstance(v, (int, float)) for v in items):
        if len(items) != 3:
            raise ValueError(
                f"diagonal supercell must have 3 entries, got {items!r}"
            )
        return [items]

    if not all(isinstance(v, (list, tuple)) for v in items):
        raise TypeError(f"malformed supercell matrix {spec!r}")

    # [[a,b,c], [d,e,f], [g,h,i]]: a full 3x3 matrix OR three diagonal cells
    if len(items) == 3 and all(len(v) == 3 for v in items):
        if all(_is_diagonal_triple(v) for v in items):
            return [list(v) for v in items]
        return [[list(v) for v in items]]

    # list of matrices (each a 3x3 or a diagonal triple)
    out = []
    for item in items:
        item = list(item)
        if all(isinstance(v, (int, float)) for v in item):
            if len(item) != 3:
                raise ValueError(
                    f"diagonal supercell must have 3 entries, got {item!r}"
                )
            out.append(item)
        elif len(item) == 3 and all(len(row) == 3 for row in item):
            out.append([list(row) for row in item])
        else:
            raise ValueError(f"malformed supercell matrix entry {item!r}")
    return out


#: ABACUS ``INPUT`` keys a *single* supercell entry may override on top of
#: the shared SCF base preset (``parameters/<backend>/scf.yml``). The key
#: use case is the mixing history: a 249-atom vacancy converges with
#: ``mixing_ndim: 16`` while the 431-atom one needs 20-24 — see the defect
#: protocol header for the measured values.
PER_CELL_SCF_KEYS = (
    "scf_thr",
    "scf_nmax",
    "mixing_type",
    "mixing_beta",
    "mixing_ndim",
    "mixing_gg0",
    "mixing_restart",
    "mixing_dmr",
    "smearing_method",
    "smearing_sigma",
    "ks_solver",
    "nbands",
    "symmetry",
    "gamma_only",
    "nelec",
)


def normalize_single_matrix(value) -> list:
    """Normalise one supercell matrix: ``[2, 2, 2]`` or a full 3x3 list."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"supercell matrix must be a list, got {value!r}")
    items = list(value)
    if all(isinstance(v, (int, float)) for v in items):
        if len(items) != 3:
            raise ValueError(f"diagonal supercell must have 3 entries, got {items!r}")
        return items
    if len(items) == 3 and all(
        isinstance(row, (list, tuple)) and len(row) == 3 for row in items
    ):
        return [[float(v) if isinstance(v, float) else int(v) for v in row] for row in items]
    raise ValueError(f"malformed supercell matrix {value!r}")


def normalize_supercell_cells(spec) -> List[dict]:
    """Normalise a preset cell list into ``[{'matrix': …, …}, …]``.

    Every entry is either a bare matrix (``[2, 2, 2]`` or a 3x3 list) or a
    mapping with the matrix plus optional per-cell settings::

        - {matrix: [2, 2, 2], kpoints_mesh: [7, 7, 7]}
        - {matrix: [3, 3, 3], kpoints_mesh: [5, 5, 5], label: "3x3x3"}
        - {matrix: [4, 4, 4], kpoints_distance: 0.1}   # 1/A
        - [2, 2, 2]                                    # bare matrix
        - {matrix: null}                               # input cell as host

    Returns a list of dicts with the keys ``matrix`` (or ``None``),
    ``kpoints_mesh``, ``kpoints_distance`` and ``label``.
    """
    if spec is None:
        return []
    if isinstance(spec, (str, bytes)) or not isinstance(spec, (list, tuple)):
        raise TypeError(f"supercell cells must be a list, got {type(spec).__name__}")
    items = list(spec)
    if not items:
        return []

    # a bare matrix given directly ([2,2,2] / 3x3) → single cell
    if all(isinstance(v, (int, float)) for v in items):
        return [{"matrix": normalize_single_matrix(items)}]
    if len(items) == 3 and all(
        isinstance(v, (list, tuple)) and len(v) == 3 for v in items
    ) and not all(_is_diagonal_triple(v) for v in items):
        # rows with off-diagonal elements → one real 3x3 matrix
        return [{"matrix": normalize_single_matrix(items)}]

    out = []
    for item in items:
        if isinstance(item, dict):
            kwargs = dict(item)
            matrix = kwargs.pop("matrix", None)
            cell = {"matrix": normalize_single_matrix(matrix) if matrix is not None else None}
            mesh = kwargs.pop("kpoints_mesh", None)
            if mesh is not None:
                cell["kpoints_mesh"] = [int(v) for v in mesh]
            distance = kwargs.pop("kpoints_distance", None)
            if distance is not None:
                cell["kpoints_distance"] = float(distance)
            label = kwargs.pop("label", None)
            if label is not None:
                cell["label"] = str(label)
            # per-cell SCF overrides (ABACUS INPUT keys) — the larger
            # supercells need a longer mixing history to converge:
            #   - {matrix: [6, 6, 6], mixing_ndim: 24}
            scf = {k: kwargs.pop(k) for k in list(kwargs) if k in PER_CELL_SCF_KEYS}
            # Idempotency: the CLI adapter normalises the protocol once
            # already, so the WorkChain receives entries that carry the
            # collected overrides under the 'scf' key. Accept them as-is
            # (instead of failing with "unknown per-cell keys ['scf']").
            existing = kwargs.pop("scf", None)
            if existing is not None:
                if not isinstance(existing, dict):
                    raise TypeError(
                        f"per-cell 'scf' must be a mapping of ABACUS INPUT "
                        f"keys, got {existing!r}"
                    )
                unknown = sorted(set(existing) - set(PER_CELL_SCF_KEYS))
                if unknown:
                    raise ValueError(
                        f"unknown SCF override keys {unknown} in {item!r} "
                        f"(allowed: {list(PER_CELL_SCF_KEYS)})"
                    )
                scf = {**existing, **scf}
            if scf:
                cell["scf"] = scf
            if kwargs:
                raise ValueError(
                    f"unknown per-cell keys {sorted(kwargs)} in {item!r} "
                    "(allowed: matrix, kpoints_mesh, kpoints_distance, label, "
                    f"plus SCF overrides {list(PER_CELL_SCF_KEYS)})"
                )
            out.append(cell)
        else:
            out.append({"matrix": normalize_single_matrix(item)})
    return out


def supercell_label(matrix) -> str:
    """Short label for a supercell matrix: ``2x2x2`` (diagonal) or ``m3x3``."""
    import numpy as np

    arr = np.asarray(matrix, dtype=float)
    if arr.shape == (3,):
        if arr[0] == arr[1] == arr[2]:
            return f"{int(arr[0])}x{int(arr[1])}x{int(arr[2])}"
        return "diag" + "-".join(str(int(v)) for v in arr)
    if arr.shape == (3, 3):
        diag = np.diag(arr)
        off = arr - np.diag(diag)
        if np.allclose(off, 0):
            return f"{int(diag[0])}x{int(diag[1])}x{int(diag[2])}"
    return "m3x3"


def _to_pymatgen(structure):
    from pymatgen.core import Structure as PmgStructure

    return PmgStructure(
        lattice=structure.cell,
        species=[site.kind_name for site in structure.sites],
        coords=[site.position for site in structure.sites],
        coords_are_cartesian=True,
    )


def _to_aiida(pmg_structure):
    from aiida.orm import StructureData

    return StructureData(pymatgen=pmg_structure)


def create_vacancy(supercell, site_index: int):
    """Return ``supercell`` with the atom at ``site_index`` removed.

    ``site_index`` indexes the sites of the supercell (0-based). The
    cell is unchanged — only the atom is removed.
    """
    pmg = _to_pymatgen(supercell)
    n_sites = len(pmg)
    if not 0 <= site_index < n_sites:
        raise ValueError(
            f"vacancy site_index {site_index} out of range "
            f"(supercell has {n_sites} sites)"
        )
    removed = pmg.sites[site_index]
    pmg.remove_sites([site_index])
    return _to_aiida(pmg), removed.species_string


def create_interstitial(supercell, element: str, position_frac: List[float]):
    """Return ``supercell`` with an atom of ``element`` inserted.

    ``position_frac`` is the fractional coordinate of the interstitial
    site inside the supercell cell (e.g. ``[0.5, 0.5, 0.5]`` for a body
    centre). The user is responsible for choosing a sensible interstitial
    site (tetrahedral / octahedral / high-symmetry position).
    """
    pmg = _to_pymatgen(supercell)
    pmg.append(element, list(position_frac), coords_are_cartesian=False)
    return _to_aiida(pmg)


def formation_energy(
    defect_energy_ev: float,
    host_energy_ev: float,
    defect_natoms: int,
    host_natoms: int,
) -> Dict[str, Any]:
    """Compute the defect formation energy without chemical potentials.

    Uses the atom-count-scaled formula::

        E_f = E_defect − E_host × (N_defect / N_host)

    where the perfect-cell energy is scaled to the defect cell's atom
    count (each atom contributes the same average energy), so no chemical
    potential (μ) is needed. For a vacancy (N_defect = N_host − 1) this
    equals E_defect − E_host×(N−1)/N.

    Returns the formation energy plus the raw energy difference
    (E_defect − E_host, unscaled) as a reference.
    """
    scale = defect_natoms / host_natoms if host_natoms else 1.0
    e_f = defect_energy_ev - host_energy_ev * scale
    return {
        "formation_energy_ev": float(e_f),
        "formula": "E_defect − E_host × N_defect/N_host",
        "defect_energy_ev": float(defect_energy_ev),
        "host_energy_ev": float(host_energy_ev),
        "host_natoms": int(host_natoms),
        "defect_natoms": int(defect_natoms),
        "energy_difference_ev": float(defect_energy_ev - host_energy_ev),
    }


def analyse_defect_species(host, defect) -> Dict[str, Dict[str, int]]:
    """Return ``{"removed": {...}, "inserted": {...}}`` atom-count changes.

    Compares the host and defect structures element-wise; used to build
    the chemical-potential terms of the formation energy.
    """
    from collections import Counter

    host_counts = Counter(site.kind_name for site in host.sites)
    defect_counts = Counter(site.kind_name for site in defect.sites)

    removed: Dict[str, int] = {}
    inserted: Dict[str, int] = {}
    for species in set(host_counts) | set(defect_counts):
        diff = defect_counts.get(species, 0) - host_counts.get(species, 0)
        if diff < 0:
            removed[species] = -diff
        elif diff > 0:
            inserted[species] = diff
    return {"removed": removed, "inserted": inserted}


def count_atoms(structure) -> Dict[str, int]:
    """Element → atom-count map of a structure."""
    from collections import Counter

    return dict(Counter(site.kind_name for site in structure.sites))
