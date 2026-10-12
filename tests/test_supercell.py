"""Tests for the supercell workflow (protocol parsing, adapters, report,
supercell structure generation)."""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

import numpy as np
import pytest

from aiida_uranium_workflow.input_builders.supercell.abacus import AbacusSupercellAdapter
from aiida_uranium_workflow.schedulers.supercell import parse_supercell_protocol
from aiida_uranium_workflow.utils.report.supercell import generate_report
from aiida_uranium_workflow.workflows.supercell.abacus import (
    SupercellScfWorkChain,
    make_supercell_structure,
)


class TestProtocolParser:
    def test_splits_backends(self):
        proto = {"abacus": {"supercells": [{"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}}
        out = parse_supercell_protocol(proto)
        assert out == proto

    def test_empty(self):
        assert parse_supercell_protocol(None) == {}
        assert parse_supercell_protocol({}) == {}


class TestMakeSupercell:
    def test_identity_keeps_cell(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert len(sc.sites) == len(cell.sites)
        assert sc.get_cell_volume() == pytest.approx(cell.get_cell_volume(), rel=1e-6)

    def test_2x2x2_octuples_volume(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        assert len(sc.sites) == 8 * len(cell.sites)
        assert sc.get_cell_volume() == pytest.approx(8 * cell.get_cell_volume(), rel=1e-6)

    def test_coordinates_cartesian_not_fractional(self, aiida_profile):
        """Regression: pymatgen_to_structure must write CARTESIAN site
        coordinates — fractional coords piled atoms into the cell corner
        and ABACUS aborted with "atoms too close"."""
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        positions = np.array([s.position for s in sc.sites])
        # The bcc primitive basis has negative components, but the sites
        # must still span the whole 2× supercell (≈6.9 Å), not be piled
        # into the cell corner (~0–1 Å).
        span = positions.max() - positions.min()
        assert span.max() > 3.0
        # Minimum periodic nearest-neighbour distance ≈ bcc NN (a√3/2).
        frac = positions @ np.linalg.inv(np.asarray(sc.cell))
        dmin = 1e9
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                d = frac[j] - frac[i]
                d -= np.round(d)
                dmin = min(dmin, np.linalg.norm(d @ np.asarray(sc.cell)))
        assert dmin == pytest.approx(3.45 * np.sqrt(3) / 2, rel=0.05)

    def test_non_diagonal_matrix(self, aiida_profile):
        from ase.build import bulk
        from aiida import orm

        cell = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        sc = make_supercell_structure(cell, [[2, 1, 0], [1, 2, 0], [0, 0, 2]])        # det([[2,1,0],[1,2,0],[0,0,2]]) = 6 primitive cells.
        assert sc.get_cell_volume() == pytest.approx(6 * cell.get_cell_volume(), rel=1e-6)


class TestAdapter:
    def test_abacus_inputs(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="supercell_test_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusSupercellAdapter(
            code_label=f"supercell_test_abacus@{computer.label}",
            software_params={
                "parameters": {"input": {"nspin": 1}},
                "kpoints_mesh": [11, 11, 11],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data={
                "abacus": {
                    "supercells": [
                        {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "label": "1x1x1", "kpoints_mesh": [11, 11, 11]},
                        {"matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                         "label": "2x2x2", "kpoints_mesh": [7, 7, 7],
                         "scf_thr": 1e-6, "mixing_beta": 0.4},
                    ],
                }
            },
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"] == "scf"
        # The adapter does not inject a restart budget — the workflow
        # forces max_iterations=0 on the child AbacusBaseWorkChain.
        assert "max_iterations" not in inputs["base"]
        cells = inputs["supercell_parameters"].get_dict()["supercells"]
        assert len(cells) == 2
        assert cells[1]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert cells[1]["scf_thr"] == 1e-6
        err = SupercellScfWorkChain.spec().inputs.validate(inputs)
        assert err is None, f"spec validation failed: {err}"

    def test_preset_calculation_is_honoured(self, aiida_profile, aiida_localhost):
        """An explicit ``calculation`` in the preset (e.g. nscf running on
        a stored density) must NOT be overwritten with scf."""
        from aiida import orm
        from aiida_uranium_workflow.input_builders.supercell.abacus import (
            AbacusSupercellAdapter,
        )
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="supercell_nscf_test",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        proto = {
            "abacus": {
                "supercells": [
                    {"matrix": [[8, 0, 0], [0, 8, 0], [0, 0, 8]], "label": "8x8x8"}
                ]
            }
        }
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))

        # explicit nscf survives
        nscf = AbacusSupercellAdapter(
            code_label=f"supercell_nscf_test@{computer.label}",
            software_params={
                "parameters": {"input": {"calculation": "nscf", "nbands": 18842}},
                "kpoints_mesh": [1, 1, 1],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data=proto,
        )._build_workchain_inputs(structure)
        assert nscf["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"] == "nscf"

        # absent -> scf default
        default = AbacusSupercellAdapter(
            code_label=f"supercell_nscf_test@{computer.label}",
            software_params={
                "parameters": {"input": {"nspin": 1}},
                "kpoints_mesh": [1, 1, 1],
                "pseudo_family": "sg15_sz",
            },
            metadata={},
            workflow_data=proto,
        )._build_workchain_inputs(structure)
        assert default["base"]["abacus"]["parameters"].get_dict()["input"]["calculation"] == "scf"

    def test_missing_supercells_raises(self, aiida_profile, aiida_localhost):
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="supercell_test_empty",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusSupercellAdapter(
            code_label=f"supercell_test_empty@{computer.label}",
            software_params={"parameters": {"input": {"nspin": 1}}},
            metadata={},
            workflow_data={"abacus": {}},
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        with pytest.raises(ValueError):
            adapter._build_workchain_inputs(structure)


class TestScfInputsFor:
    """Per-cell SCF input assembly: restart budget forced to 0."""

    @staticmethod
    def _process():
        # ``_scf_inputs_for`` only touches ``self.inputs.base`` and cell
        # dicts — a bare instance suffices (no AiiDA instantiation).
        from aiida_uranium_workflow.workflows.supercell.abacus import (
            SupercellScfWorkChain,
        )

        return object.__new__(SupercellScfWorkChain)

    @staticmethod
    def _base():
        from aiida import orm

        return {
            "abacus": {
                "code": "fake",
                "parameters": orm.Dict(dict={"input": {"nspin": 1}}),
            },
            "kpoints": orm.List(list=[1, 1, 1]),
        }

    def test_max_iterations_forced_to_zero(self):
        """The child AbacusBaseWorkChain gets max_iterations=0 (no
        auto-restart of a failed supercell SCF)."""
        process = self._process()
        cell = {"entry": {}, "label": "1x1x1", "structure": object()}
        inputs = process._scf_inputs_for(cell, self._base())
        assert inputs["max_iterations"].value == 0

    def test_zero_overrides_base_value(self):
        """Even if the shared base carries a restart budget, the child
        still receives 0."""
        from aiida import orm

        process = self._process()
        base = self._base()
        base["max_iterations"] = orm.Int(5)
        cell = {"entry": {}, "label": "1x1x1", "structure": object()}
        inputs = process._scf_inputs_for(cell, base)
        assert inputs["max_iterations"].value == 0

    def test_smearing_sigma_override_reaches_input(self):
        """A per-supercell smearing_sigma must land in the child's ABACUS
        ``input`` block (sigma scans run one entry per sigma)."""
        process = self._process()
        base = self._base()
        cell = {"entry": {"smearing_sigma": 0.015, "scf_thr": 1e-4},
                "label": "8x8x8_sigma0p015", "structure": object()}
        inputs = process._scf_inputs_for(cell, base)
        pinput = inputs["abacus"]["parameters"].get_dict()["input"]
        assert pinput["smearing_sigma"] == 0.015
        assert pinput["scf_thr"] == 1e-4
        # The rest of the base SCF input survives.
        assert pinput["nspin"] == 1

    def test_pexsi_overrides_reach_input(self):
        """PEXSI pole-expansion keys are per-cell overridable: the pole
        count has to grow with the cell size, so a scan cannot freeze it
        in the shared base preset."""
        process = self._process()
        base = self._base()
        from aiida import orm

        base["abacus"]["parameters"] = orm.Dict(
            dict={
                "input": {
                    "nspin": 1,
                    "ks_solver": "pexsi",
                    "gamma_only": 1,
                    "pexsi_npole": 60,
                    "pexsi_temp": 0.0073529411764706,
                    "pexsi_nproc_pole": 2,
                    "smearing_method": "fd",
                    "smearing_sigma": 0.0073529411764706,
                }
            }
        )
        cell = {
            "entry": {"pexsi_npole": 120, "pexsi_elec_thr": 1e-5},
            "label": "8x8x8_npole120",
            "structure": object(),
        }
        pinput = process._scf_inputs_for(cell, base)["abacus"][
            "parameters"
        ].get_dict()["input"]
        assert pinput["pexsi_npole"] == 120
        assert pinput["pexsi_elec_thr"] == 1e-5
        # untouched base keys survive
        assert pinput["ks_solver"] == "pexsi"
        assert pinput["pexsi_temp"] == 0.0073529411764706

    def test_mixing_restart_preset_reaches_child_input(self):
        """``test_mixing_restart`` must actually vary ``mixing_restart``.

        ``_scf_inputs_for`` copies only ``_SCF_OVERRIDE_KEYS``; a key
        outside that tuple is dropped silently, so the scan would submit
        four identical 1024-atom SCFs (and 'test' nothing). Read the
        preset straight from ``supercell.yml`` so the yml and the
        whitelist cannot drift apart again.
        """
        import yaml
        from aiida import orm

        from aiida_uranium_workflow.utils.common import PARAMETERS_DIR

        proto = yaml.safe_load(
            (PARAMETERS_DIR / "supercell.yml").read_text()
        )["test_mixing_restart"]["abacus"]["supercells"]

        process = self._process()
        base = self._base()
        base["abacus"]["parameters"] = orm.Dict(
            dict={"input": {"nspin": 1, "mixing_beta": 0.5, "mixing_ndim": 8}}
        )

        seen = []
        for entry in proto:
            cell = {
                "entry": entry,
                "label": entry["label"],
                "structure": object(),
            }
            pinput = process._scf_inputs_for(cell, base)["abacus"][
                "parameters"
            ].get_dict()["input"]
            assert pinput["mixing_restart"] == entry["mixing_restart"]
            assert pinput["mixing_dmr"] == entry["mixing_dmr"]
            assert pinput["mixing_beta"] == 0.5  # base survives
            seen.append(pinput["mixing_restart"])

        # The whole point of the preset: one distinct threshold per cell.
        # (The yml is edited by hand — do not pin the exact cell count.)
        assert len(proto) >= 2
        assert len(set(map(str, seen))) == len(proto)

    def test_every_preset_entry_key_reaches_child_input(self):
        """Invariant across the whole protocol file: every per-cell key in
        ``supercell.yml`` must reach the child's ABACUS ``input`` block.

        ``_scf_inputs_for`` copies only ``_SCF_OVERRIDE_KEYS``, so a key
        missing from that tuple is dropped **silently** — the cell then
        just re-runs the base preset. This has already bitten
        ``mixing_restart`` and ``mixing_gg0``: both scans would have
        submitted four identical 1024-atom SCFs. Reading the presets
        straight from the yml keeps the file and the whitelist in sync.
        """
        import yaml
        from aiida import orm

        from aiida_uranium_workflow.utils.common import PARAMETERS_DIR

        presets = yaml.safe_load((PARAMETERS_DIR / "supercell.yml").read_text())
        # Handled outside the plain ``input`` merge.
        special = {"matrix", "label", "kpoints_mesh", "kpoints_distance"}

        process = self._process()
        base = self._base()
        base["abacus"]["parameters"] = orm.Dict(dict={"input": {"nspin": 1}})

        checked = 0
        for preset, block in presets.items():
            for entry in (block.get("abacus") or {}).get("supercells") or []:
                cell = {
                    "entry": entry,
                    "label": str(entry.get("label") or preset),
                    "structure": object(),
                }
                pinputs = process._scf_inputs_for(cell, base)["abacus"][
                    "parameters"
                ].get_dict()["input"]
                for key, value in entry.items():
                    if key in special:
                        continue
                    assert key in pinputs, (
                        f"{preset}/{cell['label']}: '{key}' is not in "
                        f"_SCF_OVERRIDE_KEYS — it would be silently dropped"
                    )
                    assert pinputs[key] == value
                    checked += 1
        # Guard against a yml refactor quietly emptying this test.
        assert checked > 40


class TestReport:
    def test_table_renders_rows(self):
        out = {
            "workflow": "supercell",
            "cells": [
                {"label": "1x1x1", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "natoms": 2, "volume": 41.086, "energy": -27065.73,
                 "time_s": 120.5, "scf_steps": 21, "scf_pk": 1},
                {"label": "2x2x2", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                 "natoms": 16, "volume": 328.69, "energy": -27065.73 * 8,
                 "time_s": 3600.0, "scf_steps": 35, "scf_pk": 2},
            ],
        }
        report = generate_report(out, pk=7, workflow_type="abacus")
        assert "Supercell SCF WorkChain Report" in report
        assert "| 1x1x1 |" in report
        assert "| 2x2x2 |" in report
        assert "41.086" in report
        # time + scf steps columns present; scf pk column dropped.
        assert "| energy/atom (eV) | time (s) | scf steps |" in report
        # wall time is rendered as integer seconds
        assert "| 120 | 21 |" in report
        assert "| 3600 | 35 |" in report
        # average atomic energy column (energy / natoms, derived when the
        # stored energy_per_atom is missing)
        assert "| -13532.865000 |" in report
        assert "scf pk" not in report

    def test_time_and_steps_default_to_dash(self):
        """Rows without time / scf steps render dashes when no child is
        reachable (legacy data, child fetch fails)."""
        out = {
            "workflow": "supercell",
            "cells": [
                {"label": "1x1x1", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "natoms": 2, "volume": 41.086, "energy": -27065.73,
                 "scf_pk": 999999},  # no such node -> fetch fails
            ],
        }
        report = generate_report(out, pk=7, workflow_type="abacus")
        assert "| — | — |" in report

    def test_child_fetch_fills_missing_stats(self, monkeypatch):
        """When the output dict lacks time/scf steps, the report fetches
        them from the SCF child (scf_pk) instead of showing dashes."""
        out = {
            "workflow": "supercell",
            "cells": [
                {"label": "1x1x1", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "natoms": 2, "volume": 41.086, "energy": -27065.73,
                 "scf_pk": 12345},
            ],
        }
        fetched = {"called_with": []}

        class _FakeChild:
            pk = 12345

        def _fake_fetch_summary(node, backend):
            fetched["called_with"].append((node.pk, backend))
            return {"energy_ev": -1.0, "time_s": 3600.0, "natoms": 2,
                    "scf_steps": 42}

        import aiida.orm as aiida_orm
        import aiida_uranium_workflow.utils.parsers as parsers
        import aiida_uranium_workflow.utils.report.supercell as report_mod

        monkeypatch.setattr(aiida_orm, "load_node", lambda pk: _FakeChild())
        monkeypatch.setattr(parsers, "fetch_summary", _fake_fetch_summary)

        table = report_mod.generate_summary_table(out)
        assert "| 3600 | 42 |" in table
        assert fetched["called_with"] == [(12345, "abacus")]

    def test_empty(self):
        report = generate_report({"cells": []}, pk=1, workflow_type="abacus")
        assert "No supercell data available." in report


class TestPartialFailureReporting:
    """A scan whose cells partly fail must still be tabulated.

    The PEXSI verification steps contain cells that are *expected* to fail
    (pole count outside PEXSI's table, an SCF that cannot converge). Aborting
    in ``inspect_scfs`` left the WorkChain without ``output_parameters``, so
    the report showed nothing at all for that step.
    """

    def test_collect_results_records_failed_cells(self, monkeypatch):
        from types import SimpleNamespace

        from aiida import orm
        from ase.build import bulk
        from aiida_uranium_workflow.utils import parsers
        from aiida_uranium_workflow.workflows.supercell import abacus as wc_mod

        class _FakeChild:
            def __init__(self, pk, energy):
                self.pk = pk
                self.exit_status = 0 if energy is not None else 302
                self._energy = energy

        chain = wc_mod.SupercellScfWorkChain
        process = object.__new__(chain)
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        ctx = SimpleNamespace(
            cells=[
                {"idx": 0, "label": "ok",
                 "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "structure": structure},
                {"idx": 1, "label": "bad",
                 "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "structure": structure},
            ],
            scf_0=_FakeChild(1, -123.0),
            scf_1=_FakeChild(2, None),
        )
        monkeypatch.setattr(chain, "ctx", property(lambda self: ctx), raising=False)

        collected = {}
        monkeypatch.setattr(process, "out", lambda key, value: collected.__setitem__(key, value))
        monkeypatch.setattr(process, "report", lambda _msg: None)
        monkeypatch.setattr(
            parsers, "fetch_summary",
            lambda node, backend: {"energy_ev": node._energy, "time_s": 1.0,
                                   "scf_steps": 3, "natoms": 2},
        )

        assert process.collect_results() is chain.exit_codes.ERROR_CHILD

        payload = collected["output_parameters"].get_dict()
        assert [c["label"] for c in payload["cells"]] == ["ok", "bad"]
        assert payload["cells"][0]["status"] == "ok"
        assert payload["cells"][1]["status"] == "failed"
        assert payload["cells"][1]["energy"] is None
        assert payload["cells"][1]["exit_status"] == 302
        assert payload["failed_cells"] == ["bad"]
        assert payload["n_cells"] == 2 and payload["n_failed"] == 1
