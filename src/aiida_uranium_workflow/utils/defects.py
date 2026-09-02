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
