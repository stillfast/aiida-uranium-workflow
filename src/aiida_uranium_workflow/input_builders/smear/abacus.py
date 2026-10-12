"""ABACUS input builder for the smear workflow.

The sweep grid comes from the ``abacus`` section of the protocol preset
(``parameters/smear.yml``), which carries ABACUS-native keywords and units::

    test_simple:
      abacus:
        smearing_method_list:      # ABACUS smearing_method keywords
          - mp
          - gauss
        smearing_sigma_list:       # the unit is Ry
          - 0.0225
          - 0.0100

Both lists are handed to :class:`AbacusSmearWorkChain` verbatim — the
WorkChain overwrites ``smearing_method`` / ``smearing_sigma`` of the SCF
preset for every child of the sweep.

The legacy shared protocol layout (canonical method names ``gauss`` /
``mp`` / ``mp2`` plus a sigma in eV) is still translated here, so the
``test`` / ``gauss`` protocol presets keep working unchanged.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
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
        """Return ``(smearing_method_list, smearing_sigma_list)``.

        The protocol's ``abacus`` section is already ABACUS-native (method
        keywords + sigma in Ry), so it is passed through untouched.  Only
        the legacy shared layout — canonical method names with sigma in
        eV — needs the keyword/unit translation.
        """
        lists = self.workflow_data.get("smear_lists", {})

        native = lists.get("abacus")
        if isinstance(native, dict):
            method = list(native.get("smearing_method_list", []))
            sigma = [float(s) for s in native.get("smearing_sigma_list", [])]
            return method, sigma

        if "smear" not in lists:
            return [], []
        method = [_ABACUS_SMEAR_KW[s] for s in lists.get("smear", [])]
        sigma = [float(s) * _EV_TO_RY for s in lists.get("sigma", [])]
        return method, sigma

    def adapt(self, structure) -> AdaptedInputs:
        """Compose the final AiiDA inputs + workchain class.

        The sweep grid is written to the ``smearing_method_list`` /
        ``smearing_sigma_list`` ports of :class:`AbacusSmearWorkChain`,
        which are named after the ABACUS keywords they drive (and after
        the protocol keys they come from).  The generic
        :meth:`SoftwareAdapter.adapt` cannot be reused here because it
        hard-codes the ``smear`` / ``sigma`` port names.
        """
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        method_list, sigma_list = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        if method_list:
            inputs["smearing_method_list"] = orm.List(list=method_list)
        if sigma_list:
            inputs["smearing_sigma_list"] = orm.List(list=sigma_list)

        self._inject_options(inputs, options)

        return AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )
