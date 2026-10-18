"""Guard: per-atom magmom seeds must match the structure they seed.

Run 5b295317 (``alpha-uranium``, 4 sites) reused the ``test_mag`` protocol,
whose ``abacus.mag_list`` was written for the 2-atom bcc cell.  Nothing
checked the lengths, so all three children were submitted and then excepted
at upload with ``IndexError: list index out of range`` inside
aiida-abacus' STRU writer.  The adapters now validate the seeds up front.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.magmom.checks import (
    validate_per_atom_seeds,
)

#: The shipped ``test_mag`` protocol: one 2-atom configuration per row.
TWO_ATOM = [[[0.0], [0.0]], [[4.0], [4.0]], [[4.0], [-4.0]]]
#: The same sweep written for the 4-atom alpha-uranium cell.
FOUR_ATOM = [
    [[0.0], [0.0], [0.0], [0.0]],
    [[4.0], [4.0], [4.0], [4.0]],
    [[4.0], [-4.0], [4.0], [-4.0]],
]


def test_matching_length_passes() -> None:
    validate_per_atom_seeds(FOUR_ATOM, 4, field="abacus.mag_list", backend="abacus")
    validate_per_atom_seeds(TWO_ATOM, 2, field="abacus.mag_list", backend="abacus")


def test_empty_seeds_are_ignored() -> None:
    validate_per_atom_seeds([], 4, field="abacus.mag_list", backend="abacus")
    validate_per_atom_seeds(None, 4, field="abacus.mag_list", backend="abacus")


def test_too_few_atoms_is_reported_with_the_fix() -> None:
    # Exactly the alpha-uranium failure.
    with pytest.raises(ValueError) as excinfo:
        validate_per_atom_seeds(
            TWO_ATOM, 4, field="abacus.mag_list", backend="abacus",
            structure_label="U4",
        )
    message = str(excinfo.value)
    assert "configuration #1" in message
    assert "seeds 2 atoms" in message
    assert "structure has 4 sites (U4)" in message
    assert "parameters/magmom.yml" in message
    # The suggestion keeps the seed pattern and pads it to the cell size.
    assert "Expected: [[0.0], [0.0], [0.0], [0.0]]" in message


def test_suggestion_cycles_the_pattern() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_per_atom_seeds([[[4.0], [-4.0]]], 4, field="abacus.mag_list", backend="abacus")
    assert "Expected: [[4.0], [-4.0], [4.0], [-4.0]]" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        validate_per_atom_seeds([[[4.0]]], 4, field="abacus.mag_list", backend="abacus")
    assert "Expected: [[4.0], [4.0], [4.0], [4.0]]" in str(excinfo.value)


def test_too_many_atoms_is_reported() -> None:
    with pytest.raises(ValueError, match="seeds 5 atoms"):
        validate_per_atom_seeds([[4.0] * 5], 4, field="abacus.mag_list", backend="abacus")


def test_flat_scalars_vasp_style() -> None:
    validate_per_atom_seeds([[4.0, 4.0, 4.0, 4.0]], 4, field="vasp.magmom_per_atom_list", backend="vasp")
    with pytest.raises(ValueError, match="seeds 2 atoms"):
        validate_per_atom_seeds([[0.0, 0.0]], 4, field="vasp.magmom_per_atom_list", backend="vasp")


def test_soc_vectors_accepted() -> None:
    validate_per_atom_seeds(
        [[[0.0, 0.0, 4.0], [0.0, 0.0, -4.0]]], 2, field="abacus.mag_list", backend="abacus"
    )


@pytest.mark.parametrize(
    "config",
    [
        [["x"]],                       # not numeric
        [[4.0, 4.0]],                  # 2-component vector: collinear=1, SOC=3
        [[1.0, 2.0, 3.0, 4.0]],        # 4-component vector
        [True],                        # bool is not a moment
    ],
)
def test_malformed_moment_is_reported(config) -> None:
    with pytest.raises(ValueError, match="not a magnetic moment"):
        validate_per_atom_seeds([config], 1, field="abacus.mag_list", backend="abacus")


# ---------------------------------------------------------------------------
# Wired into the adapters
# ---------------------------------------------------------------------------


@pytest.fixture
def four_atom_u(aiida_profile):
    from aiida import orm
    from ase.build import bulk

    # 4 sites: the alpha-uranium cell size that broke run 5b295317.
    return orm.StructureData(ase=bulk("U", "bcc", a=3.45, cubic=True).repeat((2, 1, 1)))


@pytest.fixture
def code_label(aiida_profile, aiida_localhost, request):
    from aiida import orm

    label = "seed_" + request.node.name.replace("[", "_").replace("]", "")
    orm.InstalledCode(
        label=label, computer=aiida_localhost, filepath_executable="/bin/true"
    ).store()
    return f"{label}@{aiida_localhost.label}"


def _abacus_adapter(code_label, mag_list):
    from aiida_uranium_workflow.input_builders import magmom

    return magmom.AbacusMagmomAdapter(
        code_label=code_label,
        software_params={"parameters": {"input": {"calculation": "scf", "nspin": 2}}},
        metadata={},
        workflow_data={"magmom_lists": {"abacus": {"mag_list": mag_list}}},
        extra_codes={},
    )


def test_abacus_adapter_rejects_two_atom_seeds(code_label, four_atom_u) -> None:
    adapter = _abacus_adapter(code_label, TWO_ATOM)
    with pytest.raises(ValueError, match="seeds 2 atoms"):
        adapter.adapt(four_atom_u)


def test_abacus_adapter_accepts_matching_seeds(code_label, four_atom_u) -> None:
    adapter = _abacus_adapter(code_label, FOUR_ATOM)
    adapted = adapter.adapt(four_atom_u)
    assert adapted.inputs["magmom_list"].get_list() == FOUR_ATOM


def test_vasp_adapter_rejects_two_atom_seeds(code_label, four_atom_u) -> None:
    from aiida_uranium_workflow.input_builders import magmom

    adapter = magmom.VaspMagmomAdapter(
        code_label=code_label,
        software_params={
            "parameters": {"incar": {"ISPIN": 2}},
            "potential_family": "test",
            "potential_mapping": {},
        },
        metadata={},
        workflow_data={
            "magmom_lists": {"vasp": {"magmom_per_atom_list": [[0.0, 0.0], [4.0, 4.0]]}}
        },
        extra_codes={},
    )
    with pytest.raises(ValueError, match="seeds 2 atoms"):
        adapter.adapt(four_atom_u)
