"""banddos-workflow input builders.

* :mod:`.abacus` :class:`AbacusBandAdapter` — wraps
  ``abacus.band`` (``AbacusBandWorkChain``)
* :mod:`.fleur`  :class:`FleurBandAdapter`  — wraps
  ``aiida_uranium_workflow.workflows.banddos.FleurBandAndDosWorkChain``
  (runs ``fleur.banddos`` twice, once for band and once for DOS).
* :mod:`.vasp`   :class:`VaspBandAdapter`   — wraps aiida-vasp's
  ``vasp.v2.bands`` (``VaspBandsWorkChain``: SCF + seekpath k-path +
  non-SCF band pass + optional constant-charge DOS pass).
"""
from .abacus import AbacusBandAdapter
from .fleur import FleurBandAdapter
from .vasp import VaspBandAdapter

__all__ = ["AbacusBandAdapter", "FleurBandAdapter", "VaspBandAdapter"]