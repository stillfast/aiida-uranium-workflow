"""VASP input builder for the elastic workflow.

Reads the VASP SCF / relaxation parameters from
``parameters/vasp/elastic.yml`` (per-backend preset, self-contained
INCAR) and the strain lists from the workflow protocol
(``parameters/elastic.yml``'s ``vasp`` block), then assembles inputs
for
:class:`aiida_uranium_workflow.workflows.elastic.vasp.VaspElasticWorkChain`.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Dict


class VaspElasticAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the VASP elastic WorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "uranium.elastic.vasp"

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

        elastic_proto = dict(self.workflow_data.get("vasp", {}) or {})
        norm = elastic_proto.get("norm_strains") or [-0.010, -0.005, 0.005, 0.010]
        shear = elastic_proto.get("shear_strains") or [-0.010, -0.005, 0.005, 0.010]
        inputs["norm_strains"] = orm.List(list=norm)
        inputs["shear_strains"] = orm.List(list=shear)
        # Relax internal coordinates under each strain (child INCAR
        # ``ISIF=2`` + ``IBRION=2``, cell fixed) — default True, the VASP
        # counterpart of the official ABACUS elastic example's
        # ``calculation relax``. The protocol may set it to False for
        # clamped-ion constants.
        inputs["relax_internal"] = orm.Bool(
            elastic_proto.get("relax_internal", True)
        )
        # Fully relax the lattice (vasp.relax, cell + ions) before the
        # deformations — default True, the VASP counterpart of the
        # official ABACUS elastic example's ``prepare_elastic.sh``
        # (``../OPT/``, ``calculation cell-relax``). The optional
        # ``relax_settings`` dict is forwarded verbatim to the
        # VaspRelaxWorkChain RelaxOptions.
        inputs["relax_lattice"] = orm.Bool(
            elastic_proto.get("relax_lattice", True)
        )
        relax_settings = elastic_proto.get("relax_settings")
        if relax_settings:
            inputs["relax_settings"] = orm.Dict(dict=relax_settings)
        return inputs

    def _prepare_workflow_inputs(self):
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.elastic.vasp import (
            VaspElasticWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=VaspElasticWorkChain,
            inputs=inputs,
        )
