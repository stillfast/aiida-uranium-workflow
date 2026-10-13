"""Regression: an ABACUS magmom sweep must run spin polarised.

Run ``39df4b1f`` produced a report whose ``M_cell`` column was empty.  The
pipeline was fine — the data was never computed: the base SCF preset
(``magmom_lcao`` in the v14 data root) set ``nspin: 1``, so ABACUS skipped
the spin density, the initial moments shipped in ``stru.mag`` were ignored,
and the parser emitted neither ``magnetism`` nor ``final_magnetism``.

The magmom adapter now repairs such a preset to ``nspin: 2`` and warns; a
deliberate ``nspin: 2`` / ``nspin: 4`` (SOC) preset is left untouched.  The
caller's parameter dict is never mutated (the adapter works on a deep copy),
so a repaired sweep cannot leak back into the preset object.
"""

from __future__ import annotations

import warnings

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def _make_adapter(code_label: str, input_block: dict):
    from aiida_uranium_workflow.input_builders import magmom

    return magmom.AbacusMagmomAdapter(
        code_label=code_label,
        software_params={"parameters": {"input": dict(input_block)}},
        metadata={"options": {"queue_name": "q_test", "max_wallclock_seconds": 600}},
        workflow_data={},
        extra_codes={},
    )


@pytest.fixture
def code_label(aiida_profile, aiida_localhost, request):
    from aiida import orm

    # One code per test: the profile persists across tests, so a fixed label
    # would collide on the second store().
    label = "magmom_nspin_" + request.node.name.replace("[", "_").replace("]", "")
    orm.InstalledCode(
        label=label,
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()
    return f"{label}@{aiida_localhost.label}"


@pytest.fixture
def u_structure(aiida_profile):
    from aiida import orm
    from ase.build import bulk

    return orm.StructureData(ase=bulk("U", "bcc", a=3.45))


def _adapted_input_block(adapter, structure) -> dict:
    adapted = adapter.adapt(structure)
    return adapted.inputs["abacus"]["parameters"].get_dict()["input"]


def test_unpolarised_preset_is_repaired(code_label, u_structure) -> None:
    adapter = _make_adapter(code_label, {"calculation": "scf", "nspin": 1})

    with pytest.warns(UserWarning, match="nspin=1"):
        block = _adapted_input_block(adapter, u_structure)

    assert block["nspin"] == 2
    # The rest of the preset survives untouched.
    assert block["calculation"] == "scf"
    # The caller's dict is untouched (deep copy inside the adapter).
    assert adapter.software_params["parameters"]["input"]["nspin"] == 1


def test_missing_nspin_is_repaired(code_label, u_structure) -> None:
    adapter = _make_adapter(code_label, {"calculation": "scf"})

    with pytest.warns(UserWarning, match="nspin=unset"):
        block = _adapted_input_block(adapter, u_structure)

    assert block["nspin"] == 2


@pytest.mark.parametrize("nspin", [2, 4])
def test_polarised_preset_is_kept(code_label, u_structure, nspin) -> None:
    adapter = _make_adapter(code_label, {"calculation": "scf", "nspin": nspin})

    # Record instead of escalating: AiiDA itself emits unrelated warnings
    # during ``adapt()``, and only the magmom repair warning matters here.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = _adapted_input_block(adapter, u_structure)

    assert [str(w.message) for w in caught if "magmom sweep" in str(w.message)] == []
    assert block["nspin"] == nspin


def test_scheduler_options_still_forwarded(code_label, u_structure) -> None:
    """The repair must not disturb the option injection that runs after it."""
    adapter = _make_adapter(code_label, {"calculation": "scf", "nspin": 1})

    with pytest.warns(UserWarning):
        adapted = adapter.adapt(u_structure)

    metadata = adapted.inputs["abacus"]["metadata"]
    assert metadata["options"]["queue_name"] == "q_test"
