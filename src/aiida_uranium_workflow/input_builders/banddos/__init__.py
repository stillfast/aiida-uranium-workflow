"""banddos-workflow input builders.

* :mod:`.abacus` :class:`AbacusBandAdapter` — wraps
  ``abacus.band`` (``AbacusBandWorkChain``)
* :mod:`.fleur`  :class:`FleurBandAdapter`  — wraps
  ``aiida_uranium_workflow.workflows.banddos.FleurBandAndDosWorkChain``
  (runs ``fleur.banddos`` twice, once for band and once for DOS).
"""
from .abacus import AbacusBandAdapter
from .fleur import FleurBandAdapter

__all__ = ["AbacusBandAdapter", "FleurBandAdapter"]