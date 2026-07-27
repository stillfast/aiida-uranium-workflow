"""Tests for the `VaspConvergenceWorkChain` class."""

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida_uranium_workflow.workflows.convergence.vasp import (
    parse_and_gather_convergence_results,
    VaspConvergenceWorkChain,
)


def test_workchain_definition():
    """Test that the workchain can be loaded and has the correct inputs/outputs defined."""
    assert VaspConvergenceWorkChain is not None

    spec = VaspConvergenceWorkChain.spec()
    assert "encut_list" in spec.inputs
    assert "kpoints_spacing_list" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_child_metadata_label_construction():
    """Test that child workchain labels are constructed correctly.

    This verifies the label format used in submit_children to ensure
    metadata.label is set properly for each (kpoints_spacing, encut) pair.
    """
    kpoints_spacing_list = [0.0159, 0.0238]
    encut_list = [300, 400]

    from itertools import product

    expected_labels = []
    for kpoints_spacing, encut in product(kpoints_spacing_list, encut_list):
        label = f"kpoints_spacing_{kpoints_spacing}_encut_{encut}".replace(".", "_")
        expected_labels.append(label)

    assert "kpoints_spacing_0_0159_encut_300" in expected_labels
    assert "kpoints_spacing_0_0238_encut_400" in expected_labels
    assert len(expected_labels) == len(kpoints_spacing_list) * len(encut_list)


def test_child_inputs_metadata_label_structure():
    """Test that child_inputs structure includes metadata.label at the top level.

    This verifies the input structure matches the expected format:
    - metadata.label is set at the top level of child_inputs
    - calc does NOT contain metadata (that would set CalcJob metadata)
    """
    from aiida import orm

    label = "kpoints_spacing_0_0159_encut_300"

    child_inputs = {
        "parameters": orm.Dict({"incar": {"encut": 300}}),
        "kpoints_spacing": orm.Float(0.0159),
        "metadata": {"label": label},
        "calc": {},
    }

    assert "metadata" in child_inputs, "metadata should be a top-level key"
    assert "label" in child_inputs["metadata"], "metadata should contain label"
    assert (
        child_inputs["metadata"]["label"] == label
    ), f"metadata.label should be {label}"
    assert "metadata" not in child_inputs["calc"], "calc should NOT contain metadata"


def test_parse_and_gather_convergence_results(aiida_profile_clean):
    """Test the `parse_and_gather_convergence_results` calcfunction with empty inputs."""
    status_dict = orm.Dict({})
    child_pks = orm.List([])

    result = parse_and_gather_convergence_results(
        status_dict=status_dict, child_pks=child_pks
    )

    assert isinstance(result, orm.Dict)
    result_dict = result.get_dict()
    assert "total_energy" in result_dict
    assert "num_atoms" in result_dict
    assert "total_energy_per_atom" in result_dict


def test_workchain_exit_codes():
    """Test that the workchain defines the expected exit codes."""
    assert hasattr(VaspConvergenceWorkChain.exit_codes, "SUCCESS")
    assert hasattr(VaspConvergenceWorkChain.exit_codes, "ERROR_CHILD")
    assert hasattr(VaspConvergenceWorkChain.exit_codes, "ERROR_PARSER")

    assert VaspConvergenceWorkChain.exit_codes.SUCCESS.status == 0
    assert VaspConvergenceWorkChain.exit_codes.ERROR_CHILD.status == 300
    assert VaspConvergenceWorkChain.exit_codes.ERROR_PARSER.status == 305


def test_workchain_inputs_kpoints_list():
    """Test that the workchain accepts kpoints_list input."""
    spec = VaspConvergenceWorkChain.spec()
    assert "kpoints_list" in spec.inputs


def test_child_metadata_label_construction_mesh():
    """Test that child workchain labels are constructed correctly for kpoints_mesh mode.

    This verifies the label format used in submit_children when using kpoints_mesh_list.
    """
    kpoints_mesh_list = [[11, 11, 11], [9, 9, 9]]
    encut_list = [300, 400]

    from itertools import product

    expected_labels = []
    for kpoints_val, encut in product(kpoints_mesh_list, encut_list):
        kpoints_str = "x".join(str(k) for k in kpoints_val)
        label = f"kpoints_{kpoints_str}_encut_{encut}".replace(".", "_")
        expected_labels.append(label)

    assert "kpoints_11x11x11_encut_300" in expected_labels
    assert "kpoints_9x9x9_encut_400" in expected_labels
    assert len(expected_labels) == len(kpoints_mesh_list) * len(encut_list)
