"""ABACUS input builder for the magmom workflow.

Reads the per-backend ``mag_list`` from the workflow protocol section
(e.g. ``parameters/magmom.yml``) and assembles inputs for
``AbacusMagmomWorkChain``.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, Dict, List


class AbacusMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into AbacusMagmomWorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.magmom.abacus"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy
        import warnings

        params = copy.deepcopy(self.software_params["parameters"])

        # A magmom sweep is only meaningful with spin polarisation: under
        # ``nspin: 1`` ABACUS computes no spin density, the initial moments
        # shipped in ``stru.mag`` are ignored, and its parser drops
        # ``magnetism`` / ``final_magnetism`` — the report then has no M_cell
        # to show.  Spin polarise the children (``nspin: 4``/SOC is kept).
        pinput = params.setdefault("input", {})
        nspin = pinput.get("nspin")
        if nspin is None or int(nspin) < 2:
            pinput["nspin"] = 2
            warnings.warn(
                "magmom sweep: the base SCF preset has "
                f"nspin={nspin if nspin is not None else 'unset'} (unpolarised) — "
                "the initial magnetic moments would be ignored and the result "
                "would report no M_cell. Using nspin=2 for this sweep; set "
                "`nspin: 2` in the preset to make it explicit.",
                stacklevel=2,
            )

        code_label = self.code_label

        inputs: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(code_label),
                "parameters": orm.Dict(params),
                "structure": structure,
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
        inputs["pseudo_family"] = orm.Str(
            str(self.software_params.get("pseudo_family", ""))
        )
        return inputs

    def _prepare_workflow_inputs(self) -> List[Any]:
        """Extract the ``mag_list`` from workflow_data.

        Each entry is a nested list of per-atom magnetizations, e.g.
        ``[[1.0, 1.0], [1.0, -1.0]]`` matching ABACUS ``stru.mag``.
        """
        lists = self.workflow_data.get("magmom_lists", {}).get("abacus", {})
        if not lists:
            return []
        return list(lists.get("mag_list", []))

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        mag_list = self._prepare_workflow_inputs()

        # Per-atom seeds: fail here, with the YAML path and the required
        # length, instead of inside aiida-abacus' STRU writer at upload time
        # (IndexError: list index out of range — see magmom.checks).
        from .checks import validate_structure_seeds

        validate_structure_seeds(
            mag_list, structure, field="abacus.mag_list", backend="abacus"
        )

        inputs = self._build_workchain_inputs(structure)
        if mag_list:
            inputs["magmom_list"] = orm.List(list=mag_list)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
