"""phonopy-workflow WorkChains (ABACUS / FLEUR / QE).

* :mod:`.abacus` :class:`AbacusPhonopyWorkChain` — frozen-phonon
  phonon bands / DOS with ABACUS forces + phonopy post-processing.
* :mod:`.fleur` :class:`FleurPhonopyWorkChain` — frozen-phonon
  phonon bands / DOS with FLEUR forces (density-mode SCF with
  ``l_f`` forces from ``output_parameters['force_atoms']``) + phonopy
  post-processing.
* :mod:`.qe` :class:`QePhonopyWorkChain` — QE-native **DFPT** phonon
  bands / DOS: pw.x SCF → ``quantumespresso.ph.base`` (ph.x on a
  q-mesh) → ``quantumespresso.q2r.base`` → ``quantumespresso.matdyn.base``
  (no supercells, no phonopy code).
"""

from .abacus import AbacusPhonopyWorkChain
from .fleur import FleurPhonopyWorkChain
from .qe import QePhonopyWorkChain

__all__ = ["AbacusPhonopyWorkChain", "FleurPhonopyWorkChain", "QePhonopyWorkChain"]
