"""ABACUS input builder for the elastic workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the strain
lists from the workflow protocol (``parameters/elastic.yml``'s
``abacus`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.elastic.abacus.AbacusElasticWorkChain`.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Dict


class AbacusElasticAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS elastic WorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.elastic.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        params = copy.deepcopy(self.software_params["parameters"])
        options = self.metadata.get("options", {})

        # Fixed-lattice SCF: no cell relaxation, stress output on.
        params["input"]["calculation"] = "scf"
        # ABACUS defaults ``cal_stress`` to 0, so without this the run
        # never prints the TOTAL-STRESS block the parser needs.
        params["input"]["cal_stress"] = 1

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
            kpoints_mesh.set_kpoints_mesh(
                list(self.software_params["kpoints_mesh"])
            )
            base["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            base["pseudo_family"] = orm.Str(
                str(self.software_params["pseudo_family"])
            )

        elastic_proto = dict(self.workflow_data.get("abacus", {}) or {})
        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
        }
        norm = elastic_proto.get("norm_strains") or [-0.010, -0.005, 0.005, 0.010]
        shear = elastic_proto.get("shear_strains") or [-0.010, -0.005, 0.005, 0.010]
        inputs["norm_strains"] = orm.List(list=norm)
        inputs["shear_strains"] = orm.List(list=shear)
        # Relax internal coordinates under each strain (abacus.relax,
        # positions only, cell fixed) — default True, the official ABACUS
        # elastic example's method. The protocol may set it to False for
        # clamped-ion constants.
        inputs["relax_internal"] = orm.Bool(
            elastic_proto.get("relax_internal", True)
        )
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.elastic.abacus import (
            AbacusElasticWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=AbacusElasticWorkChain,
            inputs=inputs,
        )
