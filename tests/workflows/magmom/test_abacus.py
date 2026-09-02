"""Tests for the `AbacusMagmomWorkChain` class."""

import os
from pathlib import Path

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida_pseudo.data.pseudo import UpfData
from aiida_pseudo.groups.family import PseudoPotentialFamily
from aiida_uranium_workflow.workflows.magmom.abacus import (
    AbacusMagmomWorkChain,
    parse_and_gather_magmom_results,
)

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(aiida_code_installed):
    """Return an ``InstalledCode`` instance configured to run calculations of given entry point on localhost."""

    def _fixture_code(entry_point_name):
        return aiida_code_installed(
            label=f"test.{entry_point_name}", default_calc_job_plugin=entry_point_name
        )

    return _fixture_code


@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing bulk silicon."""

    def _generate_structure(structure_id="silicon"):
        """Return a ``StructureData`` representing bulk silicon."""
        from aiida.orm import StructureData

        if structure_id.startswith("silicon"):
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0),
                symbols="Si",
                name="Si",
            )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return structure

    return _generate_structure


@pytest.fixture
def pseudo_family(aiida_profile_clean):
    """Create a PseudoPotentialFamily for testing."""
    data_folder = Path(os.path.join(TEST_DIR, "test_data", "pseudos"))
    family = PseudoPotentialFamily.create_from_folder(
        data_folder, "aiida-uranium-test", pseudo_type=UpfData
    )
    return family


@pytest.fixture
def generate_inputs_abacus_magmom(fixture_code, generate_structure, pseudo_family):
    """Generate default inputs for `AbacusMagmomWorkChain`."""

    def _generate_inputs_abacus_magmom():
        """Generate default inputs for `AbacusMagmomWorkChain`."""
        from aiida.orm import Dict, KpointsData, List

        parameters = Dict(
            {
                "input": {
                    "nspin": 2,
                    "symmetry": 1,
                    "basis_type": "lcao",
                    "ecutwfc": 80,
                    "scf_thr": 1e-7,
                    "scf_nmax": 100,
                    "device": "cpu",
                    "precision": "double",
                },
                "stru": {
                    "mag": [[1.0], [1.0]],
                },
            }
        )
        structure = generate_structure()

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([4, 4, 4])

        return {
            "abacus": {
                "code": fixture_code("abacus"),
                "parameters": parameters,
                "structure": structure,
                "metadata": {
                    "options": {
                        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}
                    }
                },
            },
            "kpoints": kpoints,
            "pseudo_family": orm.Str(pseudo_family.label),
            "magmom_list": List([[[1.0], [1.0]], [[1.0], [-1.0]]]),
        }

    return _generate_inputs_abacus_magmom


@pytest.fixture
def generate_workchain_abacus_magmom(generate_inputs_abacus_magmom):
    """Generate an instance of a `AbacusMagmomWorkChain`."""

    def _generate_workchain_abacus_magmom(inputs=None):
        """Generate an instance of a `AbacusMagmomWorkChain`.

        :param inputs: inputs for the `AbacusMagmomWorkChain`.
        """
        if inputs is None:
            inputs = generate_inputs_abacus_magmom()

        runner = get_manager().get_runner()
        return instantiate_process(runner, AbacusMagmomWorkChain, **inputs)

    return _generate_workchain_abacus_magmom


# ---------------------------------------------------------------------------
# Definition / inputs / outputs
# ---------------------------------------------------------------------------


def test_workchain_definition():
    """Test that the workchain can be loaded and has the correct inputs/outputs defined."""
    assert AbacusMagmomWorkChain is not None

    spec = AbacusMagmomWorkChain.spec()
    assert "magmom_list" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_workchain_exit_codes():
    """Test that the workchain defines the expected exit codes."""
    assert hasattr(AbacusMagmomWorkChain.exit_codes, "SUCCESS")
    assert hasattr(AbacusMagmomWorkChain.exit_codes, "ERROR_CHILD")
    assert hasattr(AbacusMagmomWorkChain.exit_codes, "ERROR_PARSER")

    assert AbacusMagmomWorkChain.exit_codes.SUCCESS.status == 0
    assert AbacusMagmomWorkChain.exit_codes.ERROR_CHILD.status == 300
    assert AbacusMagmomWorkChain.exit_codes.ERROR_PARSER.status == 305


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def test_generate_inputs(aiida_profile_clean, generate_inputs_abacus_magmom):
    """Test that the input generator produces valid inputs."""
    inputs = generate_inputs_abacus_magmom()

    assert "abacus" in inputs
    assert "code" in inputs["abacus"]
    assert "parameters" in inputs["abacus"]
    assert "structure" in inputs["abacus"]
    assert "metadata" in inputs["abacus"]
    assert "kpoints" in inputs
    assert "pseudo_family" in inputs
    assert "magmom_list" in inputs

    assert isinstance(inputs["magmom_list"], orm.List)
    assert isinstance(inputs["pseudo_family"], orm.Str)


def test_workchain_instantiation(
    aiida_profile_clean, generate_workchain_abacus_magmom
):
    """Test that the workchain can be instantiated with valid inputs."""
    process = generate_workchain_abacus_magmom()
    assert isinstance(process, AbacusMagmomWorkChain)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------




def test_gather_results_no_children(aiida_profile_clean, generate_workchain_abacus_magmom):
    """Test `gather_results` when no children were submitted.

    With no children appended to ``self.ctx``, every label lookup returns
    ``None`` and the workchain should return the ``ERROR_CHILD`` exit code.
    """
    process = generate_workchain_abacus_magmom()

    exit_code = process.gather_results()

    assert exit_code == process.exit_codes.ERROR_CHILD


def test_parse_and_gather_magmom_results(aiida_profile_clean):
    """Test the `parse_and_gather_magmom_results` calcfunction with empty inputs."""
    child_pks = orm.List([])

    result = parse_and_gather_magmom_results(child_pks=child_pks)

    assert isinstance(result, orm.Dict)
    result_dict = result.get_dict()
    assert "magnetism" in result_dict
    assert "final_magnetism" in result_dict
    assert result_dict["magnetism"] == {}
    assert "nspin" in result_dict
    assert result_dict["magnetism"] == {}
    assert result_dict["final_magnetism"] == {}
    assert result_dict["nspin"] == {}