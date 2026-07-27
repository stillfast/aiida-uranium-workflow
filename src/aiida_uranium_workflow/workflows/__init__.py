"""AiiDA WorkChain definitions for each workflow (smear, …).

This package contains the ``WorkChain`` classes. It does NOT contain
scheduling / submission logic — that lives in
:mod:`aiida_uranium_workflow.schedulers`.
"""

from . import smear as _smear  # noqa: F401
from . import convergence as _convergence  # noqa: F401
from . import magmom as _magmom  # noqa: F401
from .smear.abacus import AbacusSmearWorkChain
from .smear.vasp import VaspSmearWorkChain
from .convergence.abacus import AbacusConvergenceWorkChain
from .convergence.vasp import VaspConvergenceWorkChain
from .magmom.abacus import AbacusMagmomWorkChain
from .magmom.vasp import VaspMagmomWorkChain

__all__ = [
    "AbacusSmearWorkChain",
    "VaspSmearWorkChain",
    "AbacusConvergenceWorkChain",
    "VaspConvergenceWorkChain",
    "AbacusMagmomWorkChain",
    "VaspMagmomWorkChain",
]
