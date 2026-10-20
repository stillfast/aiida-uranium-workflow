"""A missing ``parameters/<backend>/<category>.yml`` must explain itself.

Case 3-elastic/vasp used ``{"vasp": {"defect": "standard"}}`` — the dict form
of ``input.json["parameters"]`` names the preset *file*, so the loader looked
for ``parameters/vasp/defect.yml`` and died with a bare
``FileNotFoundError``. The key belongs to another workflow; the error now
names the file the current workflow expects and lists what exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "aiida_uranium_workflow"


def _write_input(tmp_path: Path, parameters: dict) -> Path:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "workflow": "elastic",
                "parameters": parameters,
                "static": {"structure": "bcc-uranium", "metadata": "yeesuan"},
                "profile": "aiida_profile",
                "code": {"vasp": "vaspstd@yeesuan"},
            },
            indent=2,
        )
    )
    return path


def test_missing_category_file_is_explained(tmp_path):
    from aiida_uranium_workflow.utils.config import ConfigLoader

    loader = ConfigLoader(
        _write_input(tmp_path, {"vasp": {"defect": "standard"}}),
        data_root=PACKAGE_ROOT,
    )
    with pytest.raises(ValueError) as excinfo:
        loader.load_all()

    message = str(excinfo.value)
    assert "parameters/vasp/defect.yml not found" in message
    assert "is the preset FILE to read, not a preset name" in message
    # the elastic workflow expects parameters/vasp/elastic.yml
    assert 'For the \'elastic\' workflow the \'vasp\' entry must be' in message
    assert '"vasp": {"elastic": <preset>}' in message
    assert "Available categories for vasp:" in message
    assert "elastic" in message


def test_correct_category_loads(tmp_path):
    from aiida_uranium_workflow.utils.config import ConfigLoader

    loader = ConfigLoader(
        _write_input(tmp_path, {"vasp": {"elastic": "standard"}, "elastic": "test"}),
        data_root=PACKAGE_ROOT,
    )
    bundle = loader.load_all()
    assert bundle.software_params["vasp"]
