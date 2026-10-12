"""VASP input builder for the smear workflow.

The sweep grid comes from the ``vasp`` section of the protocol preset
(``parameters/smear.yml``), which carries VASP-native values::

    test_simple:
      vasp:
        smear_list:      # ISMEAR integers
          - 0
          - 1
        sigma_list:      # the unit is eV
          - 0.3
          - 0.2

Both lists are handed to :class:`VaspSmearWorkChain` verbatim — the
WorkChain writes ``ismear`` / ``sigma`` into the INCAR of every child of
the sweep.

The legacy shared protocol layout (canonical method names ``gauss`` /
``mp`` / ``mp2`` plus a sigma in eV) is still translated here, so the
``test`` / ``gauss`` protocol presets keep working unchanged.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, List, Tuple

_VASP_ISMEAR_KW = {
    "gauss": 0,
    "mp": 1,
    "mp2": 2,
}


class VaspAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspSmearWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "uranium.smear.vasp"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        entry = self.software_params
        incar = copy.deepcopy(entry.get("parameters", {}).get("incar", {}))

        options = self.metadata.get("options", {})

        inputs: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "structure": structure,
            # ``VaspWorkChain.spec`` types this port as ``orm.Dict``; a plain
            # dict is rejected at submit-time port validation.
            "parameters": orm.Dict(dict={"incar": incar}),
            "potential_family": orm.Str(entry["potential_family"]),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
            "calc": {"metadata": {"options": options} if options else {}},
        }
        if "kpoints_spacing" in entry:
            inputs["kpoints_spacing"] = orm.Float(entry["kpoints_spacing"])
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            inputs["kpoints"] = kpoints_mesh
        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[List, List[float]]:
        """Return ``(smear_list, sigma_list)`` for the VASP sweep.

        The protocol's ``vasp`` section is already VASP-native (ISMEAR
        integers + sigma in eV), so it is passed through untouched.  Only
        the legacy shared layout — canonical method names — needs the
        ``ismear`` translation.
        """
        lists = self.workflow_data.get("smear_lists", {})

        native = lists.get("vasp")
        if isinstance(native, dict):
            smear = [int(s) for s in native.get("smear_list", [])]
            sigma = [float(s) for s in native.get("sigma_list", [])]
            return smear, sigma

        if "smear" not in lists:
            return [], []
        smear = [_VASP_ISMEAR_KW[s] for s in lists.get("smear", [])]
        sigma = [float(s) for s in lists.get("sigma", [])]  # VASP wants eV
        return smear, sigma

    def adapt(self, structure) -> AdaptedInputs:
        """Compose the final AiiDA inputs + workchain class.

        The sweep grid is written to the ``smear_list`` / ``sigma_list``
        ports of :class:`VaspSmearWorkChain`, which are named after the
        protocol keys they come from.  The generic
        :meth:`SoftwareAdapter.adapt` cannot be reused here because it
        hard-codes the ``smear`` / ``sigma`` port names.
        """
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        smear_list, sigma_list = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        if smear_list:
            inputs["smear_list"] = orm.List(list=smear_list)
        if sigma_list:
            inputs["sigma_list"] = orm.List(list=sigma_list)

        self._inject_options(inputs, options)

        return AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )
