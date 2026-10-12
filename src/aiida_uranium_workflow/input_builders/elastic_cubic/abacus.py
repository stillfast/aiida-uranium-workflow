"""ABACUS input builder for the cubic elastic-constant workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` (``software_params``)
and assembles inputs for
:class:`aiida_uranium_workflow.workflows.elastic_cubic.abacus.AbacusCubicElasticWorkChain`.

Optional equilibrium data (``bulk_modulus`` / ``volume_gs`` /
``energy_gs_ev``) and the crystal-system marker may be supplied through
the workflow_data (``parameters/elastic_cubic.yml`` preset once one is
defined); without them the WorkChain runs the 7-point EOS fallback.
"""

from __future__ import annotations

from ..base import filter_scheduler_options, SoftwareAdapter
from typing import Any

import copy

#: Optional per-run knobs read from the protocol/workflow_data.
_EQUILIBRIUM_KEYS = ("bulk_modulus", "volume_gs", "energy_gs_ev")


class AbacusCubicElasticAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS cubic-elastic WorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.elastic_cubic.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        params = copy.deepcopy(self.software_params["parameters"])

        # Fixed-lattice SCF — the distorted cells must stay at the
        # imposed strain (no relaxation).
        params["input"]["calculation"] = "scf"

        # aiida-abacus's ``abacus.base`` metadata.options namespace is not
        # dynamic — drop non-scheduler keys from the shared metadata
        # presets (e.g. ``optimize_resources`` in the ``yunhe`` preset).
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )
        base: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(params),
                "metadata": {"options": options} if options else {},
            },
        }
        if "kpoints_distance" in self.software_params:
            base["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            base["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            base["pseudo_family"] = orm.Str(str(self.software_params["pseudo_family"]))

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "crystal_system": orm.Str("cubic"),
        }

        preset = dict(self.workflow_data.get("elastic_cubic", {}) or {})
        equilibrium = dict(preset.get("equilibrium", {}) or {})
        for key in _EQUILIBRIUM_KEYS:
            if equilibrium.get(key) is not None:
                inputs[key] = orm.Float(float(equilibrium[key]))
        if "relax_internal" in preset:
            inputs["relax_internal"] = orm.Bool(bool(preset["relax_internal"]))
        if preset.get("eos_parameters"):
            inputs["eos_parameters"] = orm.Dict(dict=preset["eos_parameters"])
        if preset.get("deltas"):
            inputs["deltas"] = orm.List(list=[float(d) for d in preset["deltas"]])

        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation (resolves the WorkChain from
    # ``_workchain_entry_point``).
