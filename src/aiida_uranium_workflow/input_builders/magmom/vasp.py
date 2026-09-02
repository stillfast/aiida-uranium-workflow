"""VASP input builder for the magmom workflow.

Reads the per-backend ``magmom_mapping_list`` from the workflow protocol
section (e.g. ``parameters/magmom.yml``) and assembles inputs for
``VaspMagmomWorkChain``.
"""

from __future__ import annotations

from ..base import SoftwareAdapter
from typing import Any, Dict


class VaspMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspMagmomWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "uranium.magmom.vasp"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        entry = self.software_params
        incar = copy.deepcopy(entry.get("parameters", {}).get("incar", {}))

        options = self.metadata.get("options", {})

        inputs: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "structure": structure,
            "parameters": {"incar": incar},
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

    def _prepare_workflow_inputs(self) -> dict[str, list]:
        """Extract the vasp magmom lists from workflow_data.

        Returns ``{"magmom_mapping_list": [...], "magmom_per_atom_list": [...]}``.
        ``magmom_mapping_list`` entries are per-species dicts like
        ``{"Si": 1.0}`` / ``{"U": [1.0, -1.0]}``; ``magmom_per_atom_list``
        entries are per-site lists like ``[0.0, 0.0]`` / ``[4.0, -4.0]``.
        """
        lists = self.workflow_data.get("magmom_lists", {}).get("vasp", {})
        if not lists:
            return {}
        return {
            "magmom_mapping_list": list(lists.get("magmom_mapping_list", [])),
            "magmom_per_atom_list": list(lists.get("magmom_per_atom_list", [])),
        }

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        magmom_lists = self._prepare_workflow_inputs()

        inputs = self._build_workchain_inputs(structure)
        magmom_mapping_list = magmom_lists.get("magmom_mapping_list") or []
        magmom_per_atom_list = magmom_lists.get("magmom_per_atom_list") or []
        if magmom_per_atom_list:
            # Per-atom (site-order) initial moments: use the
            # ``magmom_per_atom_list`` port (aiida-vasp's
            # ``magmom_per_atom`` takes precedence over ``magmom_mapping``).
            inputs["magmom_per_atom_list"] = orm.List(list=magmom_per_atom_list)
        elif magmom_mapping_list:
            inputs["magmom_list"] = orm.List(list=magmom_mapping_list)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
