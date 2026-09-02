"""eos-workflow input builders (ABACUS custom WorkChain / FLEUR plugin reuse)."""

from .abacus import AbacusEosAdapter
from .fleur import FleurEosAdapter

__all__ = ["AbacusEosAdapter", "FleurEosAdapter"]
