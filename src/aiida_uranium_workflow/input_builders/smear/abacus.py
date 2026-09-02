"""ABACUS input builder for the smear workflow."""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, List, Tuple

_ABACUS_SMEAR_KW = {
    "gauss": "gauss",
    "mp": "mp",
    "mp2": "mp2",
}

# Conversion: canonical sigma is in eV; ABACUS wants Ry.
_EV_TO_RY = 1 / 13.605693


class AbacusAdapter(SoftwareAdapter):
    """Translate a ParamBundle into AbacusSmearWorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.smear.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        params = copy.deepcopy(self.software_params["parameters"])

        options = self.metadata.get("options", {})
        code_label = self.code_label

        inputs: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(code_label),
                # ``AbacusCalculation.spec.parameters`` requires ``orm.Dict``.
                # The plugin store the raw Python dict internally; the
                # ``AbacusBaseWorkChain.setup`` step calls ``.get_dict()`` to
                # convert it back.  Wrap here so the workchain schema
                # validation passes.
                "parameters": orm.Dict(params),
                "structure": structure,
                # ``metadata`` mirrors the shape used in
                # ``test_abacus_base.py`` — scheduler ``options`` are
                # placed here by ``SoftwareAdapter._inject_options``.
                "metadata": {},
            },
        }
        if "kpoints_distance" in self.software_params:
            inputs["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(
                list(self.software_params["kpoints_mesh"])
            )
            inputs["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            inputs["pseudo_family"] = orm.Str(
                str(self.software_params["pseudo_family"])
            )
        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[List, List[float]]:
        """Translate canonical smear/sigma into ABACUS keywords + Ry sigma."""
        lists = self.workflow_data.get("smear_lists", {})
        if not lists:
            return [], []
        smear = [_ABACUS_SMEAR_KW[s] for s in lists.get("smear", [])]
        sigma = [s * _EV_TO_RY for s in lists.get("sigma", [])]
        return smear, sigma
