"""Scheduler for the banddos workflow.

Binds the banddos workflow to:

* ``parameters/banddos.yml`` (protocol — one entry per preset with
  per-backend blocks: ``abacus`` (the ``band_settings`` namespace:
  ``run_bands`` / ``run_dos`` / ``band_kpoints_distance`` /
  ``dos_kpoints_distance`` / …), ``fleur`` (``band_wf`` /
  ``dos_wf`` overrides) and ``vasp`` (``run_bands`` / ``run_dos`` /
  ``run_proj_dos`` + the same k-path settings, forwarded to
  aiida-vasp's ``band_settings``). The per-backend layout mirrors
  ``parameters/magmom.yml``.
* ``parameters/abacus/banddos.yml`` / ``parameters/fleur/banddos.yml``
  (per-backend SCF presets — physical / SCF parameters only)
* a parser hook that splits the protocol section into the per-backend
  blocks inside ``workflow_data``
* the orchestrator below

Unlike smear / convergence / magmom, the banddos workflow is **not** a
parameter sweep — one ``AbacusBandWorkChain`` is submitted per preset
(ABACUS) or one ``FleurBandAndDosWorkChain`` per preset (FLEUR).
"""
from __future__ import annotations

from aiida_uranium_workflow.input_builders.banddos import (
    AbacusBandAdapter,
    FleurBandAdapter,
    VaspBandAdapter,
)
from aiida_uranium_workflow.schedulers.base import (
    register_workflow,
    WorkflowOrchestrator,
)
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Banddos-specific protocol parser
# ---------------------------------------------------------------------------


def parse_banddos_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Split the protocol section into per-backend blocks.

    The ``parameters/banddos.yml`` file (protocol path) now uses the
    same per-backend layout as ``parameters/magmom.yml`` — each preset
    carries an ``abacus`` and a ``fleur`` block::

        tdos:
          abacus:
            run_bands: True
            run_dos: True
            band_kpoints_distance: 0.02
            dos_kpoints_distance: 0.1
            ...
          fleur:
            band_wf: {...}
            dos_wf: {...}

    The ``abacus`` block is forwarded verbatim as the
    ``AbacusBandWorkChain``'s ``band_settings`` namespace; the ``fleur``
    block (``band_wf`` / ``dos_wf``) goes to the FLEUR adapter; the
    ``vasp`` block goes to the VASP adapter, which maps
    ``run_bands``/``run_dos``/``run_proj_dos`` onto aiida-vasp's
    ``only_dos``/``run_dos`` pair.
    """
    if not protocol:
        return {}
    # ``or {}`` guards the YAML placeholders (a bare ``vasp:`` line is
    # ``None``, not a dict).
    return {
        "band_settings": dict(protocol.get("abacus") or {}),
        "fleur": dict(protocol.get("fleur") or {}),
        "vasp": dict(protocol.get("vasp") or {}),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BanddosWorkflowOrchestrator(WorkflowOrchestrator):
    """Orchestrator for the banddos workflow.

    Supports ABACUS (``AbacusBandWorkChain``), FLEUR
    (``FleurBandAndDosWorkChain``) and VASP (aiida-vasp's
    ``vasp.v2.bands`` / ``VaspBandsWorkChain``). The
    ``parameters/banddos.yml`` protocol presets carry per-backend
    blocks — ``abacus`` (the ``band_settings`` namespace), ``fleur``
    (``band_wf`` / ``dos_wf``) and ``vasp`` (``run_bands`` /
    ``run_dos`` / ``run_proj_dos`` + the k-path settings); the
    per-backend preset files under ``parameters/<backend>/scf.yml``
    carry the SCF / physical parameters (shared with relax).
    """

    ADAPTERS = {
        "abacus": AbacusBandAdapter,
        "fleur":  FleurBandAdapter,
        "vasp":   VaspBandAdapter,
    }

    BACKENDS = ("abacus", "fleur", "vasp")

    #: Per-backend preset-subkey mapping — all backends share the
    #: unified ``parameters/<backend>/scf.yml`` (same file as relax):
    #: ``"abacus": {"scf": [...]}``, ``"vasp": {"scf": "standard"}``.
    PRESET_SUBKEYS: dict[str, str] = {
        "abacus": "scf",
        "fleur":  "scf",
        "vasp":   "scf",
    }


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------


register_workflow(
    name="banddos",
    protocol_file="banddos.yml",
    parser_hook=parse_banddos_protocol,
    orchestrator_cls=BanddosWorkflowOrchestrator,
)