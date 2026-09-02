"""ABACUS input builder for the relax workflow.

Reads the SCF base from ``parameters/abacus/scf.yml`` and the
volume-relax settings from the workflow protocol
(``parameters/relax.yml``'s ``abacus`` block), then assembles inputs
for the **plugin** ``abacus.relax`` WorkChain
(:class:`aiida_abacus.workflows.relax.AbacusRelaxWorkChain`) — no
custom WorkChain is needed.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any


class AbacusRelaxAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the plugin ``abacus.relax`` inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "abacus.relax"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        params = copy.deepcopy(self.software_params["parameters"])

        # The relax settings (volume-only) come from the protocol's
        # abacus block; the SCF base from the scf.yml preset.
        relax_proto = dict(self.workflow_data.get("abacus", {}) or {})

        # ABACUS calculation mode must be a cell-relax one so the
        # volume can change (RelaxType.VOLUME). The protocol does not
        # carry 'calculation' — the preset does; force it here.
        params["input"]["calculation"] = "cell-relax"

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
            "relax_settings": orm.Dict(dict=relax_proto),
        }
        # Optional top-level relax controls.
        for key in ("meta_convergence",):
            if key in relax_proto:
                inputs[key] = orm.Bool(bool(relax_proto[key]))
        for key in ("volume_convergence",):
            if key in relax_proto:
                inputs[key] = orm.Float(float(relax_proto[key]))
        for key in ("max_meta_convergence_iterations",):
            if key in relax_proto:
                inputs[key] = orm.Int(int(relax_proto[key]))
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation, which submits the plugin
    # WorkChain resolved from ``_workchain_entry_point`` ("abacus.relax").
