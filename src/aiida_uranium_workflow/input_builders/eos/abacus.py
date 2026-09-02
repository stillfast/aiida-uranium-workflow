"""ABACUS input builder for the eos workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the EOS scan
settings from the workflow protocol (``parameters/eos.yml``'s ``abacus``
block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.eos.abacus.AbacusEosWorkChain`.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any

import copy


class AbacusEosAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS EOS WorkChain inputs."""

    name = "abacus"

    #: SCF-input keys the EOS protocol may override on top of the
    #: shared ``scf.yml`` preset (EOS-grade convergence is looser than
    #: the generic SCF threshold).
    _SCF_OVERRIDE_KEYS = (
        "scf_thr",
        "scf_nmax",
        "mixing_beta",
        "smearing_method",
        "smearing_sigma",
    )

    def _workchain_entry_point(self) -> str:
        return "uranium.eos.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        params = copy.deepcopy(self.software_params["parameters"])

        # Fixed-lattice SCF — the cell must stay at the scaled volume.
        params["input"]["calculation"] = "scf"

        # EOS scan settings + optional SCF convergence overrides from
        # the protocol's ``abacus`` block.
        eos_proto = dict(self.workflow_data.get("abacus", {}) or {})
        defaults = {"points": 9, "step": 0.005, "guess": 1.00}
        eos_settings = dict(defaults)
        eos_settings.update({k: v for k, v in eos_proto.items() if k in defaults})
        for key in self._SCF_OVERRIDE_KEYS:
            if key in eos_proto:
                params["input"][key] = eos_proto[key]

        options = self.metadata.get("options", {})
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
            "eos_parameters": orm.Dict(dict=eos_settings),
        }
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation (submits the WorkChain
    # resolved from ``_workchain_entry_point``).
