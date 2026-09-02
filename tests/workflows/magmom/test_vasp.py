"""Tests for the `VaspMagmomWorkChain` class."""

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida_uranium_workflow.workflows.magmom.vasp import (
    VaspMagmomWorkChain,
    parse_and_gather_magmom_results,
)


# ---------------------------------------------------------------------------
# Definition / inputs / outputs
# ---------------------------------------------------------------------------


def test_workchain_definition():
    """Test that the workchain can be loaded and has the correct inputs/outputs defined."""
    assert VaspMagmomWorkChain is not None

    spec = VaspMagmomWorkChain.spec()
    assert "magmom_list" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_workchain_exit_codes():
    """Test that the workchain defines the expected exit codes."""
    assert hasattr(VaspMagmomWorkChain.exit_codes, "SUCCESS")
    assert hasattr(VaspMagmomWorkChain.exit_codes, "ERROR_CHILD")
    assert hasattr(VaspMagmomWorkChain.exit_codes, "ERROR_PARSER")

    assert VaspMagmomWorkChain.exit_codes.SUCCESS.status == 0
    assert VaspMagmomWorkChain.exit_codes.ERROR_CHILD.status == 300
    assert VaspMagmomWorkChain.exit_codes.ERROR_PARSER.status == 305


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------








def test_generate_inputs(aiida_profile_clean):
    """Test that an input dict for `VaspMagmomWorkChain` can be constructed."""
    from aiida.orm import KpointsData, StructureData

    structure = StructureData(cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4])

    parameters = orm.Dict(
        {
            "incar": {
                "encut": 400,
                "ismear": 0,
                "sigma": 0.01,
                "nsw": 0,
                "ibrion": -1,
                "ispin": 2,
            }
        }
    )

    inputs = {
        "code": orm.load_code("test.vasp.vasp") if orm.QueryBuilder().append(
            orm.Code, filters={"label": "test.vasp.vasp"}
        ).count() else None,
        "structure": structure,
        "kpoints": kpoints,
        "parameters": parameters,
        "potential_family": orm.Str("LDA"),
        "potential_mapping": orm.Dict({"Si": "Si"}),
        "calc": {"metadata": {"options": {"resources": {"num_machines": 1}}}},
        "magmom_list": orm.List([{"Si": 1.0}, {"Si": -1.0}]),
    }

    assert "magmom_list" in inputs
    assert isinstance(inputs["magmom_list"], orm.List)
    assert inputs["magmom_list"].get_list() == [{"Si": 1.0}, {"Si": -1.0}]


def test_child_inputs_metadata_label_structure():
    """Test that child_inputs structure includes metadata.label at the top level.

    This verifies the input structure matches the expected format:
    - metadata.label is set at the top level of child_inputs
    - calc does NOT contain metadata (that would set CalcJob metadata)
    - magmom_mapping is included on every child
    """
    label = "magmom_000_Si_1"

    child_inputs = {
        "parameters": orm.Dict({"incar": {"encut": 400}}),
        "magmom_mapping": orm.Dict({"Si": 1.0}),
        "metadata": {"label": label},
        "calc": {},
    }

    assert "metadata" in child_inputs, "metadata should be a top-level key"
    assert "label" in child_inputs["metadata"], "metadata should contain label"
    assert (
        child_inputs["metadata"]["label"] == label
    ), f"metadata.label should be {label}"
    assert "metadata" not in child_inputs["calc"], "calc should NOT contain metadata"
    assert "magmom_mapping" in child_inputs, "magmom_mapping should be set"


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_parse_and_gather_magmom_results(aiida_profile_clean):
    """Test the `parse_and_gather_magmom_results` calcfunction with empty inputs."""
    child_pks = orm.List([])

    result = parse_and_gather_magmom_results(child_pks=child_pks)

    assert isinstance(result, orm.Dict)
    result_dict = result.get_dict()
    assert "magnetization" in result_dict
    assert "site_magnetization" in result_dict
    assert result_dict["magnetization"] == {}
    assert result_dict["site_magnetization"] == {}