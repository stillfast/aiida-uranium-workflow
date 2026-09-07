"""ABACUS input builder for the supercell workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the supercell
protocol (``parameters/supercell.yml``'s ``abacus`` block — the
supercell matrix list with per-cell SCF overrides), then assembles
inputs for
:class:`aiida_uranium_workflow.workflows.supercell.abacus.SupercellScfWorkChain`.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any

import copy


class AbacusSupercellAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS supercell WorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.supercell.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        params = copy.deepcopy(self.software_params["parameters"])

        # Fixed-lattice SCF — the cell stays at the supercell volume.
        params["input"]["calculation"] = "scf"

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

        # Supercell protocol from the workflow protocol's ``abacus`` block.
        proto = dict(self.workflow_data.get("abacus", {}) or {})
        supercells = proto.get("supercells") or []
        if not supercells:
            raise ValueError(
                "supercell protocol has no 'supercells' list; check "
                "parameters/supercell.yml."
            )

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "supercell_parameters": orm.Dict(dict={"supercells": supercells}),
        }
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation (submits the WorkChain
    # resolved from ``_workchain_entry_point``).
