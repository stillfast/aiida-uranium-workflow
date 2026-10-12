"""phonopy-workflow input builders (ABACUS / FLEUR / QE).

* :mod:`.abacus` :class:`AbacusPhonopyAdapter` — wraps
  ``AbacusPhonopyWorkChain`` (ABACUS SCF + forces for displaced
  supercells, phonopy band / DOS post-processing).
* :mod:`.fleur` :class:`FleurPhonopyAdapter` — wraps
  ``FleurPhonopyWorkChain`` (FLEUR SCF in ``density`` mode with
  ``l_f`` forces + phonopy post-processing).
* :mod:`.qe` :class:`QePhonopyAdapter` — wraps
  ``QePhonopyWorkChain`` (QE-native DFPT chain: pw.x SCF → ph.x →
  q2r.x → matdyn.x; codes ``ph`` / ``q2r`` / ``matdyn`` come from
  ``input.json["code"]``).
"""

from .abacus import AbacusPhonopyAdapter
from .fleur import FleurPhonopyAdapter
from .qe import QePhonopyAdapter

__all__ = ["AbacusPhonopyAdapter", "FleurPhonopyAdapter", "QePhonopyAdapter"]
