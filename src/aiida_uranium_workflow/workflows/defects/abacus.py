"""ABACUS defect-formation-energy WorkChain.

Runs the host (perfect) and defective cells through the plugin ABACUS
WorkChains:

* ``mode: scf``    → ``abacus.base`` (fixed-lattice SCF)
* ``mode: relax``  → ``abacus.relax`` with ``relax_type: positions`` —
  only the atomic positions are relaxed, the cell stays fixed. The
  plugin's meta-convergence loop (volume-based) is switched off
  (``meta_convergence=False``) because the cell volume never changes.

Then computes the neutral formation energy::

    E_f = E_defect − E_host − Σ_i n_i·μ_i

The SCF/relax base comes from ``parameters/abacus/scf.yml``; the
supercell / defect / chemical-potential settings come from the workflow
protocol (``parameters/defects.yml``). An optional ``relax_settings``
Dict overrides the default ``RelaxOptions`` (relax_method / force_cutoff
/ max_ionic_steps ...).

Layout::

    inputs
    ├── structure / supercell_matrix / defect / wf_parameters
    ├── chemical_potentials   (Dict, optional)
    ├── relax_settings        (Dict, optional) — abacus.relax RelaxOptions
    ├── base                  (namespace) — AbacusBaseWorkChain inputs
    outputs
    ├── output_parameters     (Dict) — formation energy + energies
    ├── host_structure        (StructureData)
    └── defect_structure      (StructureData)
"""

from __future__ import annotations

from aiida import orm
from aiida.plugins import WorkflowFactory
from typing import Any, Dict

from aiida_uranium_workflow.workflows.defects.base import DefectsWorkChainBase

# The plugin's own ABACUS entry points (SCF / relax children).
_PluginBaseWorkChain = WorkflowFactory("abacus.base")
_PluginRelaxWorkChain = WorkflowFactory("abacus.relax")

#: Default abacus.relax settings: relax atomic positions only (the cell
#: stays fixed — what a defect calculation needs). force_cutoff matches
#: the relax workflow protocol (parameters/relax.yml): 0.05 eV/Å.
DEFAULT_RELAX_SETTINGS: Dict[str, Any] = {
    "relax_type": "positions",  # RelaxType.POSITIONS: atoms only, cell fixed
    "perform": True,
    "relax_method": "cg",
    "max_ionic_steps": 50,
    "force_cutoff": 0.05,  # eV/Å (force_thr_ev)
    "stress_cutoff": 1.0,  # kBar (stress_thr; unused for positions relax)
}


class AbacusDefectsWorkChain(DefectsWorkChainBase):
    """Compute a defect formation energy with ABACUS."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace(
            "base", dynamic=True, help="AbacusBaseWorkChain inputs (SCF base)."
        )
        spec.input(
            "relax_settings",
            valid_type=orm.Dict,
            required=False,
            help="abacus.relax RelaxOptions (relax_type/relax_method/"
            "max_ionic_steps/force_cutoff/...). Defaults to "
            "relax_type='positions' (atomic positions only, cell fixed).",
        )

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _wc_cls(self):
        return (
            _PluginRelaxWorkChain if self._mode() == "relax" else _PluginBaseWorkChain
        )

    def _make_calc_inputs(
        self, structure, label: str, kpoints=None, scf_overrides=None
    ) -> Dict[str, Any]:
        base = dict(self.inputs.base) if "base" in self.inputs else {}
        params = (
            dict(base["abacus"]["parameters"].get_dict())
            if "parameters" in base["abacus"]
            else {}
        )
        params.setdefault("input", {})
        # Per-cell SCF overrides from the protocol (e.g. ``mixing_ndim: 24``
        # for a large supercell) win over the shared SCF base preset.
        if scf_overrides:
            params["input"].update(dict(scf_overrides))

        if self._mode() == "relax":
            # abacus.relax: structure at top level, SCF base in the
            # ``base`` namespace, relaxation settings in ``relax_settings``.
            # ``calculation`` must already be set — the plugin's input
            # validator requires it — and stays "relax" for positions relax.
            params["input"]["calculation"] = "relax"
            relax_settings = dict(DEFAULT_RELAX_SETTINGS)
            if "relax_settings" in self.inputs:
                relax_settings.update(self.inputs.relax_settings.get_dict())

            base_ns: Dict[str, Any] = {
                "abacus": {
                    "code": base["abacus"]["code"],
                    "parameters": orm.Dict(dict=params),
                },
            }
            if "metadata" in base["abacus"]:
                base_ns["abacus"]["metadata"] = base["abacus"]["metadata"]
            if kpoints is not None:  # per-cell k-mesh override
                base_ns["kpoints"] = kpoints
            elif "kpoints" in base:
                base_ns["kpoints"] = base["kpoints"]
            elif "kpoints_distance" in base:
                base_ns["kpoints_distance"] = base["kpoints_distance"]
            if "pseudo_family" in base:
                base_ns["pseudo_family"] = base["pseudo_family"]
            # ``abacus.relax`` exposes the base WorkChain namespace, so the
            # restart budget travels as ``base.max_iterations`` (default 0:
            # no automatic SCF retry — see the WorkChain input's help).
            base_ns["max_iterations"] = orm.Int(self._child_max_iterations())

            inputs: Dict[str, Any] = {
                "structure": structure,
                "base": base_ns,
                "relax_settings": orm.Dict(dict=relax_settings),
                # The plugin's meta-convergence loop compares the cell
                # volume between iterations — meaningless when the cell
                # is fixed. With it on, positions relax would run the
                # same relax twice; off, it converges after one run.
                "meta_convergence": orm.Bool(False),
            }
        else:
            params["input"]["calculation"] = "scf"
            inputs = {
                "abacus": {
                    "code": base["abacus"]["code"],
                    "parameters": orm.Dict(dict=params),
                    "structure": structure,
                },
            }
            if "metadata" in base["abacus"]:
                inputs["abacus"]["metadata"] = base["abacus"]["metadata"]
            if kpoints is not None:  # per-cell k-mesh override
                inputs["kpoints"] = kpoints
            elif "kpoints" in base:
                inputs["kpoints"] = base["kpoints"]
            elif "kpoints_distance" in base:
                inputs["kpoints_distance"] = base["kpoints_distance"]
            if "pseudo_family" in base:
                inputs["pseudo_family"] = base["pseudo_family"]
            # scf mode submits ``abacus.base`` directly → its own top-level
            # ``max_iterations`` port (default 0: no SCF retry ladder).
            inputs["max_iterations"] = orm.Int(self._child_max_iterations())

        inputs["metadata"] = {
            "label": f"defects_{label}",
            "description": f"{label.capitalize()} cell ({self._mode()})",
        }
        return inputs

    def _read_energy(self, workchain) -> float:
        """Total energy (eV) from the ABACUS calculation output.

        Both ``abacus.base`` and ``abacus.relax`` expose the calculation's
        ``misc`` output (``abacus.relax`` re-exposes the last base
        workchain outputs), so the same access works for both modes.
        """
        return float(workchain.outputs.misc.get_dict()["total_energy"])

    def _read_summary(self, workchain) -> dict:
        """Wall time + electronic steps of a finished child (for the report).

        ``fetch_summary`` reads the ABACUS ``running_scf.log`` (``Total
        Time`` and the iteration table) through the child's ``retrieved``
        folder; it returns ``None`` for the individual fields when the file
        is missing, so a failed-but-finished child cannot break the scan.
        """
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        try:
            return fetch_summary(workchain, "abacus")
        except Exception:  # noqa: BLE001 — timings are best-effort
            return {}
