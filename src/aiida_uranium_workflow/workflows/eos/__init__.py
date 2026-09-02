"""eos-workflow WorkChains.

* :mod:`.abacus` :class:`AbacusEosWorkChain` — energy–volume scan with
  ABACUS SCFs + an EOS fit for the equilibrium volume.

The FLEUR side reuses the plugin ``fleur.eos``
(:class:`aiida_fleur.workflows.eos.FleurEosWorkChain`) directly — no
custom FLEUR EOS WorkChain is defined here.
"""

from .abacus import AbacusEosWorkChain

__all__ = ["AbacusEosWorkChain"]
