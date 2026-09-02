"""FLEUR elastic-constants WorkChain (energy method).

FLEUR does not print a stress tensor (no ``stress`` support in any
schema version, see fleur.md), so elastic constants are extracted from
the total energy of deformed structures:

1. Generate deformed structures (Materials Project methodology:
   3 normal strains × ``norm_strains`` + 3 shear strains ×
   ``shear_strains``) plus, when ``combined_strains`` is given, the
   user-method combined modes (biaxial / triaxial, e.g. ε1+ε2) that
   probe the off-diagonal constants C12 / C13 / C23.
2. Run a ``FleurScfWorkChain`` for each (or a ``FleurRelaxWorkChain``
   with ``relaxation_type='atoms'`` when ``relax_internal`` — the cell
   stays fixed at the strained lattice and the internal coordinates
   relax, giving the physically relevant "relaxed" constants); read the
   total energy. **Units**: the relax workflow's ``last_energy`` is
   already in eV (bcc-U 2 atoms ≈ -1 528 354 eV, matching the EOS
   report); the plain SCF workflow's ``total_energy`` is in Hartree —
   see :meth:`gather_results`.
3. Fit the full 6×6 elastic tensor to the Voigt quadratic form
   ``E(ε) = E0 + (V/2)·εᵀ·C·ε`` by least squares (combined strains
   deliver the off-diagonal elements).

The SCF base comes from ``parameters/fleur/scf.yml``; the strain lists
come from the workflow protocol (``parameters/elastic.yml``).
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida.plugins import WorkflowFactory

#: eV per Hartree — used to convert ``output_scf_wc_para.total_energy``
#: (Hartree) to eV. Note the relax workflow's ``last_energy`` is already
#: in eV and must NOT be multiplied again (see gather_results).
HA_TO_EV = 27.211386245988

FleurScfWorkChain = WorkflowFactory("fleur.scf")
FleurRelaxWorkChain = WorkflowFactory("fleur.relax")

#: Default FLEUR relax wf parameters for the strained cells: relax the
#: atomic positions only (the cell is fixed by the strain), then stop —
#: no extra final SCF, matching the ABACUS elastic example.
DEFAULT_RELAX_WF_PARAMETERS = {
    "relaxation_type": "atoms",  # positions only, cell fixed
    "run_final_scf": False,
    "force_criterion": 0.001,  # Htr/bohr — the plugin default
    "relax_iter": 20,
}


class FleurElasticWorkChain(WorkChain):
    """Compute elastic constants of a structure with FLEUR (energy method)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("fleur", valid_type=orm.InstalledCode, required=True)
        spec.input("inpgen", valid_type=orm.InstalledCode, required=False)
        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input("wf_parameters", valid_type=orm.Dict, required=False)
        spec.input("calc_parameters", valid_type=orm.Dict, required=False)
        spec.input("options", valid_type=orm.Dict, required=False)
        spec.input("options_inpgen", valid_type=orm.Dict, required=False)
        spec.input(
            "norm_strains",
            valid_type=orm.List,
            required=False,
            help="Normal strain magnitudes, e.g. [-0.01, -0.005, 0.005, 0.01].",
        )
        spec.input(
            "shear_strains",
            valid_type=orm.List,
            required=False,
            help="Shear strain magnitudes, e.g. [-0.01, -0.005, 0.005, 0.01].",
        )
        spec.input(
            "combined_strains",
            valid_type=orm.List,
            required=False,
            help=(
                "User-method combined-strain Voigt modes (each a 6-component "
                "list, e.g. [1, 1, 0, 0, 0, 0] for the biaxial ε1+ε2), "
                "applied at every norm_strains magnitude. They probe the "
                "off-diagonal constants C12 / C13 / C23."
            ),
        )
        spec.input(
            "relax_internal",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Relax the internal atomic coordinates under each strain "
                "(FleurRelaxWorkChain, relaxation_type='atoms', cell fixed "
                "by the strain) so the fitted constants are the physically "
                "relevant 'relaxed' ones. For cells with no free internal "
                "coordinates (e.g. the 2-atom bcc U cell) it is a no-op — "
                "and the plain FleurScfWorkChain (density-converged SCF, "
                "the EOS-style energy quality) is then the better choice. "
                "Set to False for clamped-ion constants (fixed-lattice "
                "FleurScfWorkChain children)."
            ),
        )
        spec.input(
            "relax_wf_parameters",
            valid_type=orm.Dict,
            required=False,
            help=(
                "Optional overrides for the FleurRelaxWorkChain wf "
                "parameters used on the strained cells (defaults: "
                "relaxation_type='atoms', run_final_scf=False, "
                "force_criterion=0.001 Htr/bohr, relax_iter=20). For "
                "energy-method elastic constants on cells with free "
                "internal coordinates, run a final density-converged SCF "
                "after the relaxation — e.g. "
                "{\"run_final_scf\": true, \"force_criterion\": 0.0005} — "
                "since the force-mode relaxation SCF alone does not reach "
                "meV energy accuracy."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.generate_deformations,
            cls.submit_scfs,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "Elastic constants computed.")
        spec.exit_code(300, "ERROR_CHILD", "An SCF child failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to fit elastic tensor.")

    def generate_deformations(self):
        """Create the deformed structures and record their strains."""
        from aiida_uranium_workflow.utils.elastic import (
            apply_deformation_matrix,
            generate_deformations,
        )

        norm = list(self.inputs.norm_strains.get_list())
        shear = list(self.inputs.shear_strains.get_list())
        combined = (
            list(self.inputs.combined_strains.get_list())
            if "combined_strains" in self.inputs
            else []
        )
        deformations = generate_deformations(norm, shear, combined)

        self.ctx.strain_voigts = [d[0] for d in deformations]
        self.ctx.deformation_labels = [d[2] for d in deformations]
        self.ctx.deformed_structures = [
            apply_deformation_matrix(self.inputs.structure, d[1]) for d in deformations
        ]
        self.report(
            f"generated {len(deformations)} deformed structures "
            f"(norm={norm}, shear={shear}, combined={len(combined)} modes)"
        )

    def _child_inputs(self, structure, relax_internal: bool) -> dict:
        """Compose the child inputs for one deformed structure.

        With ``relax_internal`` the child is a ``FleurRelaxWorkChain``
        (``relaxation_type='atoms'`` — atoms relax, the cell stays fixed
        at the strained lattice; the plugin switches the SCF to force
        mode itself); otherwise a fixed-lattice ``FleurScfWorkChain``.
        """
        scf_inputs: dict = {
            "fleur": self.inputs.fleur,  # required by FleurScfWorkChain
            "wf_parameters": self.inputs.wf_parameters,
            "calc_parameters": self.inputs.calc_parameters,
            "structure": structure,
        }
        if "inpgen" in self.inputs:
            scf_inputs["inpgen"] = self.inputs.inpgen
        if "options" in self.inputs:
            scf_inputs["options"] = self.inputs.options
        if "options_inpgen" in self.inputs:
            scf_inputs["options_inpgen"] = self.inputs.options_inpgen

        if relax_internal:
            # FleurRelaxWorkChain: SCF base in the ``scf`` namespace, the
            # relax workflow's own wf parameters at the top level. The
            # defaults can be overridden per strain via
            # ``relax_wf_parameters`` (e.g. run a final density-converged
            # SCF for meV-accurate energies).
            wf_parameters = dict(DEFAULT_RELAX_WF_PARAMETERS)
            if "relax_wf_parameters" in self.inputs:
                wf_parameters.update(self.inputs.relax_wf_parameters.get_dict())
            return {
                "scf": scf_inputs,
                "wf_parameters": orm.Dict(dict=wf_parameters),
            }
        return scf_inputs

    def submit_scfs(self):
        """Submit one SCF / position-relax per deformed structure."""
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )
        child_cls = FleurRelaxWorkChain if relax_internal else FleurScfWorkChain

        for idx, structure in enumerate(self.ctx.deformed_structures):
            child_inputs = self._child_inputs(structure, relax_internal)
            child_inputs["metadata"] = {
                "label": f"elastic_{idx:03d}_{self.ctx.deformation_labels[idx]}",
                "description": (
                    "Position-relaxed SCF of a deformed structure"
                    if relax_internal
                    else "Fixed-lattice SCF of a deformed structure"
                ),
            }
            running = self.submit(child_cls, **child_inputs)
            self.report(f"submitted {running.process_label} pk={running.pk} for {idx}")
            self.to_context(**{f"scf_{idx}": running})

    def gather_results(self):
        """Collect total energies and fit the diagonal elastic constants."""
        from aiida_uranium_workflow.utils.elastic import crystal_system_of

        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )
        strain_voigts = list(self.ctx.strain_voigts)
        energies_ev = []
        all_ok = True

        for idx in range(len(self.ctx.deformed_structures)):
            child = getattr(self.ctx, f"scf_{idx}", None)
            if child is None or not child.is_finished_ok:
                all_ok = False
                continue
            try:
                if relax_internal:
                    # Relaxed structure's total energy. aiida-fleur 2.0.0's
                    # ``output_relax_wc_para['last_energy']`` is the final
                    # total energy in **eV** (the plugin's Htr→eV conversion
                    # already happened; e.g. bcc-U 2 atoms ≈ -1 528 354 eV,
                    # matching the EOS report). The parallel ``energy`` field
                    # is the raw value / 27.2114 and must NOT be used.
                    para = child.outputs.output_relax_wc_para.get_dict()
                    raw = para.get("last_energy")
                    energies_ev.append(float(raw) * 1.0)  # already eV
                else:
                    # Fixed-lattice SCF energy: output_scf_wc_para's
                    # ``total_energy`` is in **Hartree** (the plugin stores
                    # energy_hartree there) — convert to eV.
                    para = child.outputs.output_scf_wc_para.get_dict()
                    raw = para.get("total_energy")
                    energies_ev.append(float(raw) * HA_TO_EV)
                if raw is None:
                    all_ok = False
                    continue
            except (AttributeError, KeyError, TypeError):
                all_ok = False

        if not all_ok:
            return self.exit_codes.ERROR_CHILD
        if len(energies_ev) < 12:
            self.report(
                f"only {len(energies_ev)}/{len(self.ctx.deformed_structures)} "
                "energies available — need at least 12 for a fit"
            )
            return self.exit_codes.ERROR_PARSER

        volume = self.inputs.structure.get_cell_volume()
        self.out(
            "output_parameters",
            _combine_elastic_result(
                fit_data=orm.Dict(
                    dict={
                        "strain_voigts": [
                            sv.tolist() for sv in strain_voigts[: len(energies_ev)]
                        ],
                        "energies_ev": energies_ev,
                        "volume_ang3": volume,
                        "crystal_system": crystal_system_of(self.inputs.structure),
                    }
                ),
            ),
        )
        return None  # SUCCESS


@calcfunction
def _combine_elastic_result(fit_data):
    """Fit the elastic tensor inside a calcfunction (provenance-safe).

    All fit inputs travel in a single ``Dict`` (JSON-serialisable, no
    numpy arrays) to avoid ``orm.List`` equality pitfalls in AiiDA's
    link validation.
    """
    from aiida_uranium_workflow.utils.elastic import fit_elastic_from_energy

    data = fit_data.get_dict()
    result = fit_elastic_from_energy(
        data["strain_voigts"],
        data["energies_ev"],
        volume_ang3=float(data["volume_ang3"]),
        crystal_system=data.get("crystal_system"),
    )
    return orm.Dict(dict=result)
