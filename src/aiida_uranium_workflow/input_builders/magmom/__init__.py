"""Magmom-workflow input builders (ABACUS / VASP)."""

from .abacus import AbacusMagmomAdapter
from .vasp import VaspMagmomAdapter

__all__ = ["AbacusMagmomAdapter", "VaspMagmomAdapter"]
