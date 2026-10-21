"""Relax workflows (no custom WorkChains).

The relax WorkChains are **provided by the plugins** and called
directly through the aiida-uranium layer:

* ABACUS — ``abacus.relax`` (``aiida_abacus.workflows.relax.AbacusRelaxWorkChain``)
* VASP   — ``vasp.relax`` (``aiida_vasp.workchains.v2.relax.VaspRelaxWorkChain``)
* FLEUR  — ``fleur.relax`` (``aiida_fleur.workflows.relax.FleurRelaxWorkChain``)
* QE     — ``quantumespresso.pw.relax``
  (``aiida_quantumespresso.workflows.relax.PwRelaxWorkChain``)

The input builders (:mod:`aiida_uranium_workflow.input_builders.relax`)
assemble the plugin inputs from ``parameters/relax.yml`` +
``parameters/<backend>/scf.yml``; the report generator
(:mod:`aiida_uranium_workflow.utils.report.relax`) derives the relaxed
lattice constants / volume / energy from the plugin node outputs.
"""
