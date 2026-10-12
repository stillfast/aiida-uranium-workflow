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

        # Fixed-lattice run — the cell stays at the supercell volume.
        # ``calculation`` defaults to ``scf`` but an explicit value in the
        # preset is honoured (e.g. ``calculation: nscf`` to run a
        # non-self-consistent step on the stored density read via
        # ``init_chg file``).
        params["input"].setdefault("calculation", "scf")

        options = self.metadata.get("options", {})
        base: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(params),
                "metadata": {"options": options} if options else {},
            },
        }
        # Forward the SCF preset's ``setting`` (retrieve list) and
        # ``extra_files`` / ``extra_files_basepath`` (e.g. a pre-expanded
        # SPIN1_CHG.cube read via ``init_chg file``) to the per-cell
        # abacus.base children — ``SupercellScfWorkChain._scf_inputs_for``
        # already copies every non-``parameters`` ``abacus`` key through.
        entry = self.software_params
        setting = entry.get("setting") or entry.get("settings")
        if setting:
            base["abacus"]["settings"] = orm.Dict(dict=copy.deepcopy(dict(setting)))
        from ..base import resolve_extra_files

        extra_files = resolve_extra_files(entry.get("extra_files"))
        if extra_files:
            base["abacus"]["extra_files"] = extra_files
        if entry.get("extra_files_basepath"):
            base["abacus"]["extra_files_basepath"] = orm.Str(
                str(entry["extra_files_basepath"])
            )
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
