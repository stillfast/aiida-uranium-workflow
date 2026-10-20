"""VASP input builder for the elastic workflow.

Reads the VASP SCF / relaxation parameters from
``parameters/vasp/elastic.yml`` (per-backend preset, self-contained
INCAR) and the strain lists from the workflow protocol
(``parameters/elastic.yml``'s ``vasp`` block), then assembles inputs
for
:class:`aiida_uranium_workflow.workflows.elastic.vasp.VaspElasticWorkChain`.
"""

from __future__ import annotations

from ..base import filter_scheduler_options, AdaptedInputs, SoftwareAdapter
from typing import Any


#: Keys of ``parameters/elastic.yml``'s vasp block that configure the
#: ``vasp.relax`` lattice-relaxation child — the same names the relax
#: protocol (``parameters/relax.yml``) uses.
try:  # pragma: no cover - aiida-vasp independent
    from ..relax.vasp import PROTOCOL_KEYS as _RELAX_PROTOCOL_KEYS
except ImportError:  # pragma: no cover
    _RELAX_PROTOCOL_KEYS = frozenset({"relax_type", "relax_settings"})

#: ``relax_settings`` alone is a raw RelaxOptions override, not a spec.
_RELAX_SPEC_KEYS = frozenset(_RELAX_PROTOCOL_KEYS) - {"relax_settings"}


def _has_relax_spec(block: dict) -> bool:
    """True when the block carries relax-yml style keys (``relax_type``, …)."""
    return bool(_RELAX_SPEC_KEYS & set(block))


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
        options = filter_scheduler_options(
            self.metadata.get("options", {}),
            entry_point=self._workchain_entry_point(),
            backend=self.name,
        )

        inputs: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "structure": structure,
            # orm.Dict, not a plain dict: the WorkChain exposes the
            # VaspWorkChain ``parameters`` port (valid_type=orm.Dict), and
            # ``aiida-uranium check --out`` validates the adapted inputs
            # against the spec — a plain dict fails that check even though
            # submit() would serialise it.
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
        # Fail hard when the lattice relaxation fails? Default False: the
        # 24 strained cells are still computed from the input structure and
        # the result is flagged (``lattice_relaxed: false``). Set
        # ``require_lattice_relax: true`` in the protocol to abort with
        # ERROR_RELAX instead.
        inputs["require_lattice_relax"] = orm.Bool(
            elastic_proto.get("require_lattice_relax", False)
        )
        # Relaxation settings of the ``vasp.relax`` lattice-relaxation child.
        # The protocol block uses the SAME vocabulary as
        # ``parameters/relax.yml``'s vasp block (``relax_type`` /
        # ``relax_method`` / ``max_ionic_steps`` / ``force_cutoff`` / ...),
        # which is translated onto aiida-vasp's RelaxOptions here; a raw
        # ``relax_settings`` dict (RelaxOptions keys) overrides the
        # translation. Without either, the workchain sends the plugin
        # defaults (full cell + ions relaxation).
        relax_block = {
            key: value
            for key, value in elastic_proto.items()
            if key in _RELAX_PROTOCOL_KEYS
        }
        if _has_relax_spec(relax_block):
            from ..relax.vasp import relax_settings_from_protocol

            settings = relax_settings_from_protocol(
                relax_block, structure.get_cell_volume()
            )
            inputs["relax_settings"] = orm.Dict(dict=settings)
        elif relax_block.get("relax_settings"):
            # Only the raw RelaxOptions dict was given.
            inputs["relax_settings"] = orm.Dict(
                dict=relax_block["relax_settings"]
            )
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
