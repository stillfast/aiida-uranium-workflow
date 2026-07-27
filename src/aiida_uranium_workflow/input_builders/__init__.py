"""Input builders — translate a ParamBundle into AiiDA inputs for one WorkChain.

Each input builder knows how to assemble the AiiDA ``inputs`` dict for
exactly one backend. They are strategy objects used by the scheduler.
"""

from .abacus import AbacusAdapter
from .base import AdaptedInputs, SoftwareAdapter
from .base_workchain import AbacusBaseWorkChainAdapter, VaspBaseWorkChainAdapter
from .vasp import VaspAdapter

__all__ = [
    "SoftwareAdapter",
    "AdaptedInputs",
    "AbacusAdapter",
    "VaspAdapter",
    "AbacusBaseWorkChainAdapter",
    "VaspBaseWorkChainAdapter",
]
