"""Tests for the banddos input builders.

Regression: the ABACUS banddos adapter used to hardcode an empty
``base.abacus.metadata`` and drop the scheduler options from
``static/metadata.yml`` entirely — the generic ``_inject_options`` in
the base adapter only looks for a *top-level* ``abacus`` / ``calc`` key,
which the banddos layout (``base.abacus``) does not have. The submitted
job then ran with default resources (1 task), no queue and no wall-time
instead of the ``yeesuan`` / ``xinghe`` settings.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders.banddos.abacus import AbacusBandAdapter


class TestAbacusBandAdapterOptions:
    def _make_adapter(self, code_label: str, inpgen_label: str = ""):
        return AbacusBandAdapter(
            code_label=code_label,
            software_params={
                "parameters": {
                    "input": {"calculation": "scf", "nspin": 2, "scf_thr": 1e-7}
                },
                "kpoints_mesh": [4, 4, 4],
                "pseudo_family": "sg15_sz_soc",
            },
            metadata={
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 56,
                        "tot_num_mpiprocs": 56,
                    },
                    "max_wallclock_seconds": 7200,
                    "queue_name": "q_ysuan",
                    "withmpi": True,
                }
            },
            workflow_data={"band_settings": {"run_bands": True, "run_dos": True}},
            extra_codes={"inpgen": inpgen_label} if inpgen_label else {},
        )

    def test_options_land_in_base_abacus_metadata(self, aiida_profile, aiida_localhost):
        """The yeesuan-style scheduler options must reach
        ``inputs.base.abacus.metadata.options`` (not be dropped)."""
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="band_test_abacus2",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = self._make_adapter(f"band_test_abacus2@{computer.label}")
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)

        options = inputs["base"]["abacus"]["metadata"]["options"]
        assert options["queue_name"] == "q_ysuan"
        assert options["max_wallclock_seconds"] == 7200
        assert options["resources"]["num_mpiprocs_per_machine"] == 56

    def test_no_options_keeps_empty_metadata(self, aiida_profile, aiida_localhost):
        """Without metadata options the namespace stays empty (no crash)."""
        from aiida import orm
        from ase.build import bulk

        computer = aiida_localhost
        orm.InstalledCode(
            label="band_test_abacus",
            computer=computer,
            filepath_executable="/bin/true",
        ).store()

        adapter = AbacusBandAdapter(
            code_label=f"band_test_abacus@{computer.label}",
            software_params={
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints_mesh": [4, 4, 4],
            },
            metadata={},
            workflow_data={},
            extra_codes={},
        )
        structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        inputs = adapter._build_workchain_inputs(structure)
        assert inputs["base"]["abacus"]["metadata"] == {}


class TestAbacusBandAdapterKpoints:
    """Regression: the base namespace must receive the k-point strategy.

    Run 522011 (preset ``lcao`` in parameters/abacus/scf.yml) exited the
    inner AbacusBaseWorkChain with 202 — "Neither the `kpoints` nor the
    `kpoints_distance` input was specified" — because the adapter only
    forwarded ``kpoints_mesh`` and silently dropped the preset's
    ``kpoints_distance``. Nothing was submitted.
    """

    def _inputs(self, aiida_localhost, software_params: dict, label: str = "band_kp_abacus"):
        from aiida import orm
        from ase.build import bulk

        # One code per test: the profile persists across tests in a module.
        orm.InstalledCode(
            label=label,
            computer=aiida_localhost,
            filepath_executable="/bin/true",
        ).store()
        adapter = AbacusBandAdapter(
            code_label=f"{label}@{aiida_localhost.label}",
            software_params=software_params,
            metadata={"options": {"queue_name": "q_test"}},
            workflow_data={"band_settings": {"run_bands": True, "run_dos": True}},
            extra_codes={},
        )
        return adapter._build_workchain_inputs(
            orm.StructureData(ase=bulk("U", "bcc", a=3.45))
        )

    def test_kpoints_distance_is_forwarded(self, aiida_profile, aiida_localhost):
        inputs = self._inputs(
            aiida_localhost,
            {
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints_distance": 0.12,
                "pseudo_family": "sg15_sz",
            },
            label="band_kp_dist",
        )
        assert inputs["base"]["kpoints_distance"].value == 0.12
        assert "kpoints" not in inputs["base"]

    def test_kpoints_mesh_still_wins(self, aiida_profile, aiida_localhost):
        inputs = self._inputs(
            aiida_localhost,
            {
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints_mesh": [4, 4, 4],
                "kpoints_distance": 0.12,  # ignored, mesh takes precedence
            },
            label="band_kp_mesh",
        )
        assert inputs["base"]["kpoints"].get_kpoints_mesh()[0] == [4, 4, 4]
        assert "kpoints_distance" not in inputs["base"]

    def test_explicit_kpoints_list(self, aiida_profile, aiida_localhost):
        inputs = self._inputs(
            aiida_localhost,
            {
                "parameters": {"input": {"calculation": "scf"}},
                "kpoints": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            },
            label="band_kp_list",
        )
        assert len(inputs["base"]["kpoints"].get_kpoints()) == 2
