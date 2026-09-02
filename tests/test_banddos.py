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
