"""FLEUR defect-formation-energy WorkChain.

Runs the host (perfect) and defective cells through the plugin FLEUR
WorkChains — ``fleur.scf`` (plain SCF) or ``fleur.relax`` (atomic
positions only; the cell stays fixed) — then computes the neutral
formation energy::

    E_f = E_defect − E_host − Σ_i n_i·μ_i

The SCF base comes from ``parameters/fleur/scf.yml``; the supercell /
defect / chemical-potential settings come from the workflow protocol
(``parameters/defects.yml``).

Layout::

    inputs
    ├── structure / supercell_matrix / defect / wf_parameters
    ├── chemical_potentials   (Dict, optional)
    ├── base                  (namespace) — FLEUR SCF base
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

#: eV per Hartree (FLEUR SCF energies are reported in Hartree).
HA_TO_EV = 27.211386245988

_FleurScfWorkChain = WorkflowFactory("fleur.scf")
_FleurRelaxWorkChain = WorkflowFactory("fleur.relax")


class FleurDefectsWorkChain(DefectsWorkChainBase):
    """Compute a defect formation energy with FLEUR."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace(
            "base", dynamic=True, help="FLEUR SCF base (fleur/inpgen/codes...)."
        )
        spec.input(
            "relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "fleur.relax wf_parameters overrides for the relax modes, "
                "e.g. {'force_criterion': 0.00097} (Ha/bohr ≈ 0.05 eV/Å). "
                "Merged over the plugin defaults."
            ),
        )

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _wc_cls(self):
        return _FleurRelaxWorkChain if self._mode() == "relax" else _FleurScfWorkChain

    def _make_calc_inputs(self, structure, label: str) -> Dict[str, Any]:
        base = dict(self.inputs.base) if "base" in self.inputs else {}

        if self._mode() == "relax":
            # fleur.relax: SCF base under the ``scf`` namespace, relax
            # settings in the top-level ``wf_parameters`` (the plugin
            # defaults to relaxation_type='atoms' — positions only).
            # The force criterion is taken from ``relax_settings``
            # (defects.yml), merged over the plugin defaults, so the
            # relax converges to the same physical force (0.05 eV/Å) as
            # the relax workflow instead of stopping with exit 354
            # ("Force is small, switch to BFGS") mid-optimization.
            scf: Dict[str, Any] = {
                "fleur": base["fleur"],
                "structure": structure,
                "wf_parameters": base.get("wf_parameters", orm.Dict(dict={})),
                "calc_parameters": base.get("calc_parameters", orm.Dict(dict={})),
            }
            if "inpgen" in base:
                scf["inpgen"] = base["inpgen"]
            if "options" in base:
                scf["options"] = base["options"]
            relax_wf: Dict[str, Any] = {}
            if "relax_settings" in self.inputs:
                relax_wf.update(self.inputs.relax_settings.get_dict())
            inputs: Dict[str, Any] = {
                "scf": scf,
                "wf_parameters": orm.Dict(dict=relax_wf),
            }
        else:
            inputs = {
                "fleur": base["fleur"],
                "structure": structure,
                "wf_parameters": base.get("wf_parameters", orm.Dict(dict={})),
                "calc_parameters": base.get("calc_parameters", orm.Dict(dict={})),
            }
            if "inpgen" in base:
                inputs["inpgen"] = base["inpgen"]
            if "options" in base:
                inputs["options"] = base["options"]
            if "options_inpgen" in base:
                inputs["options_inpgen"] = base["options_inpgen"]

        inputs["metadata"] = {
            "label": f"defects_{label}",
            "description": f"{label.capitalize()} cell ({self._mode()})",
        }
        return inputs

    def _read_energy(self, workchain) -> float:
        """Total energy (eV) of a finished FLEUR child."""
        if self._mode() == "relax":
            para = workchain.outputs.output_relax_wc_para.get_dict()
            # aiida-fleur already converts to eV.
            return float(para["last_energy"])
        para = workchain.outputs.output_scf_wc_para.get_dict()
        return float(para["total_energy"]) * HA_TO_EV
