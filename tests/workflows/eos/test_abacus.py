"""Tests for the ``AbacusEosWorkChain``.

Regression coverage:

* ``collect_fit`` used to read the scaled cell from
  ``child.inputs.structure`` and crash with ``AttributeError`` (the child
  is an ``AbacusBaseWorkChain``, which exposes the ``AbacusCalculation``
  inputs — including ``structure`` — under the ``abacus`` namespace, so
  the correct path is ``child.inputs.abacus.structure``). See the
  aiida-abacus plugin ``workflows/base.py`` ``define()``:
  ``spec.expose_inputs(AbacusCalculation, namespace='abacus', ...)``.
* Failed scan points are skipped and the EOS is fitted with the
  remaining successful points (per-atom energy / volume), like the
  aiida-fleur ``FleurEosWorkChain``.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from types import SimpleNamespace

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_uranium_workflow.workflows.eos.abacus import (
    MIN_EOS_POINTS,
    AbacusEosWorkChain,
)


def _make_workchain(aiida_profile_clean):
    """A 2-atom bcc-U conventional cell (per-atom quantities differ from
    cell totals, so the per-atom fit is actually exercised)."""
    from ase.build import bulk

    structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45, cubic=True))
    assert len(structure.sites) == 2
    runner = get_manager().get_runner()
    return instantiate_process(
        runner, AbacusEosWorkChain, structure=structure, base={}
    )


def _stub_child(total_energy: float, volume: float):
    """A minimal stand-in for a finished, successful ``AbacusBaseWorkChain``.

    Mirrors the real access pattern of ``collect_fit``:
    ``child.outputs.misc.get_dict()`` and
    ``child.inputs.abacus.structure.get_cell_volume()``.
    """
    return SimpleNamespace(
        is_finished_ok=True,
        outputs=SimpleNamespace(
            misc=SimpleNamespace(get_dict=lambda: {"total_energy": total_energy})
        ),
        inputs=SimpleNamespace(
            abacus=SimpleNamespace(
                structure=SimpleNamespace(get_cell_volume=lambda: volume)
            )
        ),
    )


def _stub_failed_child():
    """A stand-in for a failed SCF child (skipped by the EOS fit)."""
    return SimpleNamespace(is_finished_ok=False)


def test_workchain_definition():
    spec = AbacusEosWorkChain.spec()
    assert "structure" in spec.inputs
    assert "base" in spec.inputs
    assert "output_parameters" in spec.outputs
    assert hasattr(AbacusEosWorkChain.exit_codes, "ERROR_CHILD")
    assert hasattr(AbacusEosWorkChain.exit_codes, "ERROR_PARSER")


def test_collect_fit_reads_volume_from_abacus_namespace(aiida_profile_clean):
    """``collect_fit`` must read the volume from ``inputs.abacus.structure``
    (regression: ``inputs.structure`` raised AttributeError → the whole
    workchain 'Excepted' in step collect_fit)."""
    process = _make_workchain(aiida_profile_clean)
    # 5 points shaped like a parabola with a minimum near V = 43.0 Å³.
    process.ctx.scales = [0.98, 0.99, 1.00, 1.01, 1.02]
    process.ctx.scaled_structures = [None] * 5  # only len() is used
    volumes = [41.0, 42.0, 43.0, 44.0, 45.0]
    energies = [-27000.0, -27001.0, -27001.5, -27001.0, -27000.0]
    for idx in range(5):
        process.ctx[f"scf_{idx}"] = _stub_child(
            total_energy=energies[idx], volume=volumes[idx]
        )

    exit_code = process.collect_fit()
    # The regression: before the fix, ``child.inputs.structure`` raised
    # AttributeError and the workchain 'Excepted' in collect_fit. Now it
    # completes and the fit runs to the end (exit_code None).
    assert exit_code is None, f"collect_fit failed: {exit_code}"

    # ``self.out`` on a non-running instantiated process stashes the
    # output on the process rather than the stored node, so verify the
    # fit result there instead of via node links.
    outputs = getattr(process, "_outputs", {})
    para = outputs.get("output_parameters")
    assert para is not None, "collect_fit did not produce output_parameters"
    para = para.get_dict() if hasattr(para, "get_dict") else dict(para)
    assert para["fit"] == "birchmurnaghan"
    assert para["n_points"] == 5
    assert para["natoms"] == 2
    assert para["volumes"] == volumes
    # The fit is per atom: the minimum sits at V/N = 21.5 Å³/atom
    # (total volumes 41–45 Å³ over 2 atoms), and the reported cell
    # total is the per-atom value × natoms.
    assert para["volumes_per_atom"] == [v / 2 for v in volumes]
    assert abs(para["volume_gs_per_atom"] - 21.5) < 0.5
    assert abs(para["volume_gs"] - 43.0) < 1.0
    assert abs(para["energy_gs_per_atom_ev"] - (-27001.5 / 2)) < 0.5
    assert abs(para["energy_gs_ev"] - (-27001.5)) < 1.0


def test_collect_fit_uses_remaining_points(aiida_profile_clean):
    """Failed SCF points are skipped — the fit uses the remaining ones."""
    process = _make_workchain(aiida_profile_clean)
    process.ctx.scales = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02]
    process.ctx.scaled_structures = [None] * 6  # only len() is used
    volumes = [39.0, 41.0, 42.0, 43.0, 44.0, 45.0]
    energies = [-26999.0, -27000.0, -27001.0, -27001.5, -27001.0, -27000.0]
    for idx in range(6):
        if idx in (1, 4):  # two scan points fail to converge
            process.ctx[f"scf_{idx}"] = _stub_failed_child()
        else:
            process.ctx[f"scf_{idx}"] = _stub_child(
                total_energy=energies[idx], volume=volumes[idx]
            )

    exit_code = process.collect_fit()
    assert exit_code is None, f"collect_fit failed: {exit_code}"

    outputs = getattr(process, "_outputs", {})
    para = outputs["output_parameters"]
    para = para.get_dict() if hasattr(para, "get_dict") else dict(para)
    # The two failed points (idx 1 and 4) are excluded from the fit.
    assert para["n_points"] == 4
    assert para["volumes"] == [39.0, 42.0, 43.0, 45.0]
    assert para["scales"] == [0.97, 0.99, 1.00, 1.02]
    assert "error" not in para  # the fit on the remaining points succeeded
    # Minimum near per-atom V = 21.5 Å³ (2 atoms), total V = 43 Å³.
    assert abs(para["volume_gs_per_atom"] - 21.5) < 0.5
    assert abs(para["volume_gs"] - 43.0) < 1.0


def test_collect_fit_too_few_points_returns_error_child(aiida_profile_clean):
    """Fewer than MIN_EOS_POINTS successful points → ERROR_CHILD (the fit
    is not possible), not a crash and not a misleading fit."""
    process = _make_workchain(aiida_profile_clean)
    process.ctx.scales = [0.98, 0.99, 1.00]
    process.ctx.scaled_structures = [None] * 3
    process.ctx.scf_0 = _stub_failed_child()
    process.ctx.scf_1 = _stub_failed_child()
    process.ctx.scf_2 = _stub_child(total_energy=-27000.0, volume=41.0)

    exit_code = process.collect_fit()
    assert exit_code == process.exit_codes.ERROR_CHILD
    assert 1 < MIN_EOS_POINTS  # sanity: the test setup really is too few
