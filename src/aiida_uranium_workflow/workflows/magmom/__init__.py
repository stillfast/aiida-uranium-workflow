"""Magmom workflow — AiiDA WorkChains.

* :mod:`.abacus` :class:`AbacusMagmomWorkChain`
* :mod:`.vasp`   :class:`VaspMagmomWorkChain`

The orchestrator for this workflow lives in
:mod:`aiida_uranium_workflow.schedulers.magmom`.
"""

from .abacus import AbacusMagmomWorkChain
from .vasp import VaspMagmomWorkChain

__all__ = ["AbacusMagmomWorkChain", "VaspMagmomWorkChain"]