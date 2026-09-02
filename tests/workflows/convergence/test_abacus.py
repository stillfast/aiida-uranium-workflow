"""Tests for the `AbacusConvergenceWorkChain` class."""

from pathlib import Path

import os
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida.engine import ExitCode
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida_pseudo.data.pseudo import UpfData
from aiida_pseudo.groups.family import PseudoPotentialFamily
from aiida_uranium_workflow.workflows.convergence.abacus import (
    AbacusConvergenceWorkChain,
    parse_and_gather_convergence_results,
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
def generate_inputs_abacus_convergence(fixture_code, generate_structure, pseudo_family):
    """Generate default inputs for `AbacusConvergenceWorkChain`."""

    def _generate_inputs_abacus_convergence():
        """Generate default inputs for `AbacusConvergenceWorkChain`."""
        from aiida.orm import Dict, List

        parameters = Dict(
            {
                "input": {
                    "symmetry": 1,
                    "basis_type": "lcao",
                    "scf_thr": 1e-7,
                    "scf_nmax": 100,
                    "device": "cpu",
                    "precision": "double",
                }
            }
        )
        structure = generate_structure()

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
            "kpoints_distance": orm.Float(0.2),
            "pseudo_family": orm.Str(pseudo_family.label),
            "ecutwfc_list": List([80, 100]),
            "kpoints_distance_list": List([0.15, 0.2]),
        }

    return _generate_inputs_abacus_convergence


@pytest.fixture
def generate_workchain_abacus_convergence(generate_inputs_abacus_convergence):
    """Generate an instance of a `AbacusConvergenceWorkChain`."""

    def _generate_workchain_abacus_convergence(inputs=None):
        """Generate an instance of a `AbacusConvergenceWorkChain`.

        :param inputs: inputs for the `AbacusConvergenceWorkChain`.
        """
        if inputs is None:
            inputs = generate_inputs_abacus_convergence()

        runner = get_manager().get_runner()
        return instantiate_process(runner, AbacusConvergenceWorkChain, **inputs)

    return _generate_workchain_abacus_convergence


def test_workchain_definition():
    """Test that the workchain can be loaded and has the correct inputs/outputs defined."""
    assert AbacusConvergenceWorkChain is not None

    spec = AbacusConvergenceWorkChain.spec()
    assert "ecutwfc_list" in spec.inputs
    assert "kpoints_distance_list" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_generate_inputs(aiida_profile_clean, generate_inputs_abacus_convergence):
    """Test that the input generator produces valid inputs."""
    inputs = generate_inputs_abacus_convergence()

    assert "abacus" in inputs
    assert "code" in inputs["abacus"]
    assert "parameters" in inputs["abacus"]
    assert "structure" in inputs["abacus"]
    assert "metadata" in inputs["abacus"]
    assert "kpoints_distance" in inputs
    assert "pseudo_family" in inputs
    assert "ecutwfc_list" in inputs
    assert "kpoints_distance_list" in inputs

    assert isinstance(inputs["ecutwfc_list"], orm.List)
    assert isinstance(inputs["kpoints_distance_list"], orm.List)
    assert isinstance(inputs["pseudo_family"], orm.Str)


def test_workchain_instantiation(
    aiida_profile_clean, generate_workchain_abacus_convergence
):
    """Test that the workchain can be instantiated with valid inputs."""
    process = generate_workchain_abacus_convergence()
    assert isinstance(process, AbacusConvergenceWorkChain)


def test_submit_children(aiida_profile_clean, generate_workchain_abacus_convergence):
    """Test the `submit_children` method records planned pairs."""
    process = generate_workchain_abacus_convergence()

    process.submit_children()

    ecutwfc_list = process.inputs.ecutwfc_list.get_list()
    kpoints_distance_list = process.inputs.kpoints_distance_list.get_list()

    expected_pairs = []
    for ecutwfc in ecutwfc_list:
        for kpoints_distance in kpoints_distance_list:
            expected_pairs.append((ecutwfc, kpoints_distance))

    assert hasattr(process.ctx, "convergence_pairs")
    assert process.ctx.convergence_pairs == expected_pairs


def test_gather_results_no_children(
    aiida_profile_clean, generate_workchain_abacus_convergence
):
    """Test `gather_results` when no children were submitted."""
    process = generate_workchain_abacus_convergence()

    exit_code = process.gather_results()

    assert exit_code == process.exit_codes.ERROR_CHILD


def test_parse_and_gather_convergence_results(aiida_profile_clean):
    """Test the `parse_and_gather_convergence_results` calcfunction with empty inputs."""
    child_pks = orm.List([])

    result = parse_and_gather_convergence_results(child_pks=child_pks)

    assert isinstance(result, orm.Dict)
    result_dict = result.get_dict()
    assert "total_energy" in result_dict
    assert "num_atoms" in result_dict
    assert "total_energy_per_atom" in result_dict
    assert result_dict["total_energy"] == {}


def test_workchain_exit_codes():
    """Test that the workchain defines the expected exit codes."""
    assert hasattr(AbacusConvergenceWorkChain.exit_codes, "SUCCESS")
    assert hasattr(AbacusConvergenceWorkChain.exit_codes, "ERROR_CHILD")
    assert hasattr(AbacusConvergenceWorkChain.exit_codes, "ERROR_PARSER")

    assert AbacusConvergenceWorkChain.exit_codes.SUCCESS.status == 0
    assert AbacusConvergenceWorkChain.exit_codes.ERROR_CHILD.status == 300
    assert AbacusConvergenceWorkChain.exit_codes.ERROR_PARSER.status == 305


def test_workchain_inputs_kpoints_list():
    """Test that the workchain accepts kpoints_list input."""
    spec = AbacusConvergenceWorkChain.spec()
    assert "kpoints_list" in spec.inputs


@pytest.fixture
def generate_inputs_abacus_convergence_mesh(
    fixture_code, generate_structure, pseudo_family
):
    """Generate inputs for `AbacusConvergenceWorkChain` with kpoints_mesh_list."""

    def _generate_inputs_abacus_convergence_mesh():
        """Generate inputs for `AbacusConvergenceWorkChain` with kpoints_mesh_list."""
        from aiida.orm import Dict, List

        parameters = Dict(
            {
                "input": {
                    "symmetry": 1,
                    "basis_type": "lcao",
                    "scf_thr": 1e-7,
                    "scf_nmax": 100,
                    "device": "cpu",
                    "precision": "double",
                }
            }
        )
        structure = generate_structure()

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
            "pseudo_family": orm.Str(pseudo_family.label),
            "ecutwfc_list": List([80, 100]),
            "kpoints_list": List([[11, 11, 11], [9, 9, 9]]),
        }

    return _generate_inputs_abacus_convergence_mesh


def test_generate_inputs_mesh(
    aiida_profile_clean, generate_inputs_abacus_convergence_mesh
):
    """Test that the input generator produces valid inputs with kpoints_list."""
    inputs = generate_inputs_abacus_convergence_mesh()

    assert "abacus" in inputs
    assert "pseudo_family" in inputs
    assert "ecutwfc_list" in inputs
    assert "kpoints_list" in inputs

    assert isinstance(inputs["ecutwfc_list"], orm.List)
    assert isinstance(inputs["kpoints_list"], orm.List)
    assert isinstance(inputs["pseudo_family"], orm.Str)


def test_workchain_instantiation_mesh(
    aiida_profile_clean, generate_inputs_abacus_convergence_mesh
):
    """Test that the workchain can be instantiated with kpoints_list."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    inputs = generate_inputs_abacus_convergence_mesh()
    runner = get_manager().get_runner()
    process = instantiate_process(runner, AbacusConvergenceWorkChain, **inputs)
    assert isinstance(process, AbacusConvergenceWorkChain)
