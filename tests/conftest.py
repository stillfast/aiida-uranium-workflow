"""Shared pytest fixtures for the aiida-uranium-workflow test-suite.

This module provides a :class:`ConfigLoader` fixture wired to the
``parameters/convergence.yml`` fixture file.  ``convergence.yml`` is the
shared-preset layout used by the convergence workflow:

.. code-block:: yaml

    test:
      abacus:
        ecutwfc_list: [...]
        kpoints_distance_list: [...]
      vasp:
        encut_list: [...]
        kpoints_spacing_list: [...]

Because no ``convergence`` workflow is registered by the package itself,
the fixture registers one locally for the lifetime of the test session.
The fixture is intentionally narrow: it only populates the protocol and
software-params sections of :class:`ParamBundle` so the YAML parsing
logic inside :class:`ConfigLoader` can be exercised without touching the
AiiDA / scheduler layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.schedulers import register_workflow
from aiida_uranium_workflow.schedulers.base import _WORKFLOW_REGISTRY
from aiida_uranium_workflow.utils.config import ConfigLoader


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
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        return instantiate_process(runner, process_class, **inputs)

    return _generate_workchain


@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing bulk silicon or uranium."""

    def _generate_structure(structure_id="silicon"):
        """Return a ``StructureData`` representing bulk silicon or uranium.

        :param structure_id: identifies the ``StructureData`` you want to generate. Either 'silicon' or 'uranium'.
        """
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
        elif structure_id == "uranium":
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="U", name="U")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0), symbols="U", name="U"
            )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return structure

    return _generate_structure


# ---------------------------------------------------------------------------
# Convergence-workflow registration
# ---------------------------------------------------------------------------

#: Protocol YAML for the temporary ``convergence`` workflow.
CONVERGENCE_PROTOCOL_FILE = "convergence.yml"

#: Protocol preset name (must exist as a top-level key in convergence.yml).
CONVERGENCE_PROTOCOL_NAME = "test"


def _ensure_convergence_workflow() -> None:
    """Register the ``convergence`` workflow if it isn't already.

    The package only ships the ``smear`` workflow.  Registering
    ``convergence`` here lets :class:`ConfigLoader` discover
    :data:`CONVERGENCE_PROTOCOL_FILE` via the registry without modifying
    the production source tree.
    """
    if "convergence" in _WORKFLOW_REGISTRY:
        return
    register_workflow(
        name="convergence",
        protocol_file=CONVERGENCE_PROTOCOL_FILE,
        workflow_key="convergence",
        parser_hook=None,
        orchestrator_cls=None,
    )


# ---------------------------------------------------------------------------
# input.json fixture
# ---------------------------------------------------------------------------

#: Minimal input.json body for the convergence-workflow fixture.
#:
#: - ``workflow``     -> triggers registry lookup of ``convergence``
#: - ``parameters``   -> both the protocol name (``convergence: test``)
#:   and the backend preset (``abacus: test``, ``vasp: test``)
#: - ``static``       -> references metadata so ``_load_metadata`` runs
#: - ``profile`` / ``code`` -> required by ``_validate``
_CONVERGENCE_INPUT: Dict[str, Any] = {
    "workflow": "convergence",
    "parameters": {
        "convergence": CONVERGENCE_PROTOCOL_NAME,
        "abacus": "test",
        "vasp": "test",
    },
    "static": {
        "structure": "bcc-uranium",
        "metadata": "yeesuan",
    },
    "profile": "aiida_profile",
    "code": {
        "abacus": "abacus@yeesuan",
        "vasp": "vaspstd@yeesuan",
    },
}


@pytest.fixture(scope="session")
def convergence_input_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the convergence input.json to a temp dir and return its path.

    The file is created once per session because the contents are
    static.  Using ``tmp_path_factory`` keeps the test hermetic — no
    pollution of the source tree.
    """
    _ensure_convergence_workflow()
    out_dir = tmp_path_factory.mktemp("convergence-input")
    out_path = out_dir / "input.json"
    out_path.write_text(json.dumps(_CONVERGENCE_INPUT, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# ConfigLoader fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config_loader(convergence_input_json: Path) -> ConfigLoader:
    """A ready-to-use :class:`ConfigLoader` pointed at ``convergence.yml``.

    The loader is constructed but not yet ``load_all``'d so individual
    tests can call whichever accessor they need.  ``input_params`` is
    already populated by ``__init__``.
    """
    return ConfigLoader(convergence_input_json)


@pytest.fixture(scope="session")
def loaded_bundle(config_loader: ConfigLoader):
    """A fully-populated :class:`ParamBundle` for the convergence fixture.

    Equivalent to ``config_loader.load_all()`` but cached for the
    session so the YAML files are only read once.
    """
    return config_loader.load_all()


# ---------------------------------------------------------------------------
# Convenience assertions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def expected_convergence_yaml() -> Dict[str, Any]:
    """Hand-checked snapshot of ``parameters/convergence.yml``.

    Kept here rather than re-parsed so a regression in the YAML parser
    cannot silently propagate into the expected values.
    """
    return {
        CONVERGENCE_PROTOCOL_NAME: {
            "abacus": {
                "ecutwfc_list": [40, 60, 80, 100, 120, 150],
                "kpoints_distance_list": [0.1, 0.15, 0.2, 0.25, 0.3],
            },
            "vasp": {
                "encut_list": [252.502, 300, 350, 400, 450, 500],
                "kpoints_spacing_list": [
                    0.0159154943,
                    0.0238732415,
                    0.0318309886,
                    0.0397887358,
                    0.0477464829,
                ],
            },
        }
    }


__all__ = [
    "CONVERGENCE_PROTOCOL_FILE",
    "CONVERGENCE_PROTOCOL_NAME",
    "convergence_input_json",
    "config_loader",
    "loaded_bundle",
    "expected_convergence_yaml",
]
