"""Input builders — translate a ParamBundle into AiiDA inputs for one WorkChain.

Each input builder knows how to assemble the AiiDA ``inputs`` dict for
exactly one workflow × backend pair. They are strategy objects used by
the scheduler.

The directory layout mirrors the workflow sub-packages::

    input_builders/
    ├── base.py                 (abstract base)
    ├── base_workchain.py       (the ``base`` workflow adapters)
    ├── smear/{abacus,vasp}.py
    ├── convergence/{abacus,vasp}.py
    ├── magmom/{abacus,vasp}.py
    └── banddos/{abacus}.py     (ABACUS only — VASP band WorkChain not yet registered)
"""
from .base import AdaptedInputs, SoftwareAdapter
from .base_workchain import AbacusBaseWorkChainAdapter, VaspBaseWorkChainAdapter
from .smear import AbacusAdapter, VaspAdapter
from .convergence import AbacusConvergenceAdapter, VaspConvergenceAdapter
from .magmom import AbacusMagmomAdapter, VaspMagmomAdapter
from .banddos import AbacusBandAdapter, FleurBandAdapter

__all__ = [
    "SoftwareAdapter",
    "AdaptedInputs",
    "AbacusAdapter",
    "VaspAdapter",
    "AbacusBaseWorkChainAdapter",
    "VaspBaseWorkChainAdapter",
    "AbacusConvergenceAdapter",
    "VaspConvergenceAdapter",
    "AbacusMagmomAdapter",
    "VaspMagmomAdapter",
    "AbacusBandAdapter",
    "FleurBandAdapter",
]
