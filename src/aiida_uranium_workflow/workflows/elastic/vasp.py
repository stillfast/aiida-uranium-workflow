"""VASP elastic-constants WorkChain (stress method).

Generates 24 deformed structures (Materials Project methodology:
3 normal strains × ``norm_strains`` + 3 shear strains ×
``shear_strains``), runs one ``VaspWorkChain`` per deformed cell, reads
the final ionic step's 3×3 stress tensor (kB) from each child's ``misc``
output, and fits the full 6×6 elastic tensor via pymatgen's
``ElasticTensor.from_independent_strains``. Aggregate moduli (K, G,
anisotropy, Poisson) are derived from it.

**Lattice relaxation** (``relax_lattice``, default True): before the
deformations are generated, the input structure is fully relaxed
(cell + ions, i.e. ISIF=3) by a ``vasp.relax`` child
(:class:`VaspRelaxWorkChain` — the workflow translates
``relax_settings`` into the correct IBRION / NSW / ISIF combination and
runs its own convergence checks). The deformation set is then built
from the *relaxed* output structure — this is the VASP counterpart of
the official ABACUS elastic example, whose ``prepare_elastic.sh`` first
optimises the cell (``../OPT/``, ``calculation cell-relax``) and only
then generates the strained cells. Set ``relax_lattice=False`` to skip
this step and deform the input structure as-is.

By default the internal atomic coordinates are relaxed under each
strain (``relax_internal`` — the child INCAR gets ``ISIF=2`` +
``IBRION=2``, atoms relax while the cell stays fixed at the strained
lattice). This gives the physically relevant "relaxed" elastic
constants and matches the VASP counterpart of the official ABACUS
elastic example (one ``calculation relax`` per deformed cell, i.e.
``ISIF=2`` as the example's README notes). For cells without free
internal coordinates (e.g. the 2-atom bcc U cell) the relaxation is a
no-op. Set ``relax_internal=False`` for the clamped-ion constants
(fixed-lattice SCF children, ``ISIF=4`` + ``IBRION=-1``).

**Stress sign convention**: VASP's "total stress" (OUTCAR / vasprun
``<varray name="stress">``) reports **positive = compression**, the same
convention as ABACUS — compressing the cell produces a positive
stress. The tensors are therefore fitted through the same
compression→tension flip as the ABACUS workflow (the default
``compression_positive=True``; pymatgen's own
``ElasticTensor.from_independent_strains`` applies the same negation
for VASP data via its ``vasp`` flag).

The SCF / relaxation parameters come from the per-backend preset
(``parameters/vasp/elastic.yml``); the strain lists, the lattice-relax
switch and its settings come from the workflow protocol
(``parameters/elastic.yml``).

Layout::

    inputs
    ├── code / structure / parameters / potential_family /
    │   potential_mapping / kpoints (or kpoints_spacing) / calc
    │       — VaspWorkChain inputs (``structure`` is overridden per
    │         deformed cell; ``parameters.incar`` gains the ISIF /
    │         IBRION / NSW of the chosen relax mode)
    ├── norm_strains    (List) — normal strain magnitudes
    ├── shear_strains   (List) — shear strain magnitudes
    ├── relax_internal  (Bool) — relax internal coords under strain (default True)
    ├── relax_lattice   (Bool) — full lattice relaxation (vasp.relax,
    │                            cell + ions) before the deformations (default True)
    ├── relax_settings  (Dict) — VaspRelaxWorkChain RelaxOptions, e.g.
    │                            {"force_cutoff": 0.03, "steps": 60}
    outputs
    └── output_parameters (Dict) — elastic tensor + moduli (GPa)
"""

from __future__ import annotations

from typing import Any, Dict

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida.plugins import WorkflowFactory

# The aiida-vasp v2 ``VaspWorkChain`` (``vasp.v2.vasp`` entry point) and
# the ``VaspRelaxWorkChain`` (``vasp.v2.relax``) used for the optional
# full lattice relaxation of the input structure.
ChildWorkChain = WorkflowFactory("vasp.v2.vasp")
RelaxWorkChain = WorkflowFactory("vasp.v2.relax")

#: INCAR keys injected for the chosen relax mode. ``relax_internal``
#: (default) relaxes atoms only — ISIF=2 + IBRION=2 + NSW, the VASP
#: equivalent of the ABACUS elastic example's ``calculation relax``
#: (the example's README itself notes ``ISIF = 2``). Otherwise the cell
#: is clamped (ISIF=4 + IBRION=-1, single point).
_RELAX_INCAR: Dict[str, Any] = {"isif": 2, "ibrion": 2, "nsw": 50}
_CLAMPED_INCAR: Dict[str, Any] = {"isif": 4, "ibrion": -1, "nsw": 0}
#: Force criterion (eV/Å) applied when the internal coordinates are
#: relaxed under the strain — same order as the abacus.relax default
#: force_cutoff (0.05 eV/Å) but tighter, since the stress (and hence
#: the fitted tensor) is sensitive to the residual forces.
_FORCE_CUTOFF_EV_ANG = -0.02


class VaspElasticWorkChain(WorkChain):
    """Compute elastic constants of a structure with VASP (stress method)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=[
                "code",
                "structure",
                "kpoints",
                "kpoints_spacing",
                "parameters",
                "potential_family",
                "potential_mapping",
                "calc",
            ],
        )
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
            "relax_internal",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Relax the internal atomic coordinates under each strain "
                "(child INCAR ``ISIF=2`` + ``IBRION=2``, cell fixed by the "
                "strain) so the fitted constants are the physically relevant "
                "'relaxed' ones — the VASP counterpart of the official ABACUS "
                "elastic example's ``calculation relax`` (the example's README "
                "notes ``ISIF = 2``). For cells with no free internal "
                "coordinates (e.g. the 2-atom bcc U cell) it is a no-op. Set "
                "to False for clamped-ion constants (``ISIF=4`` fixed-lattice "
                "SCF children)."
            ),
        )
        spec.input(
            "relax_lattice",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Fully relax the lattice (cell + ions, ISIF=3) of the input "
                "structure with a ``vasp.relax`` child before generating the "
                "deformations, and build the strained cells from the relaxed "
                "structure — the VASP counterpart of the official ABACUS "
                "elastic example's ``prepare_elastic.sh`` (``../OPT/``, "
                "``calculation cell-relax``). Default True. Set to False to "
                "deform the input structure as-is."
            ),
        )
        spec.input(
            "relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "VaspRelaxWorkChain RelaxOptions dict (e.g. "
                "``{\"force_cutoff\": 0.03, \"steps\": 60}``). Defaults to "
                "full cell + ions relaxation with the plugin's defaults."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)
        spec.output("deformed_structures", valid_type=orm.List, required=False)

        spec.outline(
            cls.initialize,
            cls.generate_deformations,
            cls.submit_scfs,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "Elastic constants computed.")
        spec.exit_code(300, "ERROR_CHILD", "An SCF child failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to fit elastic tensor.")
        spec.exit_code(310, "ERROR_RELAX", "The lattice relaxation child failed.")

    def initialize(self):
        """Optionally relax the lattice (cell + ions) before the deformations.

        With ``relax_lattice`` (default) a ``vasp.relax`` child is
        submitted with the plugin's RelaxOptions translated from
        ``relax_settings``; the engine waits for it to finish before the
        next outline step runs, so ``ctx.lattice_relax`` holds the
        finished child when ``generate_deformations`` executes.
        """
        relax_lattice = (
            self.inputs.relax_lattice.value
            if "relax_lattice" in self.inputs
            else True
        )
        if not relax_lattice:
            self.report("skipping lattice relaxation (relax_lattice=False)")
            return

        base = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        relax_inputs: Dict[str, Any] = {
            "vasp": {k: v for k, v in base.items() if k != "structure"},
            "structure": self.inputs.structure,
            # ``relax_settings`` is a *required* input of VaspRelaxWorkChain
            # and, importantly, the workchain uses the dict verbatim as its
            # runtime configuration (``ctx.relax_settings``) — a partial dict
            # would blow up on ``relax_settings.perform``. Always pass the
            # **complete** RelaxOptions defaults, with any user-supplied keys
            # merged on top.
            "relax_settings": self._relax_settings_node(),
        }
        relax_inputs["metadata"] = {
            "label": "lattice_relax",
            "description": (
                "Full lattice relaxation (cell + ions) of the input "
                "structure before the elastic deformations."
            ),
        }
        running = self.submit(RelaxWorkChain, **relax_inputs)
        self.report(
            f"submitted lattice relaxation {running.process_label} pk={running.pk}"
        )
        self.to_context(lattice_relax=running)

    def _relax_settings_node(self) -> orm.Dict:
        """Return the full VaspRelaxWorkChain RelaxOptions dict.

        VaspRelaxWorkChain consumes ``relax_settings`` verbatim as its
        runtime configuration (``ctx.relax_settings``), so a partial or
        empty dict breaks it (``relax_settings.perform`` KeyError). The
        complete RelaxOptions defaults (full cell + ions relaxation,
        force_cutoff 0.03 eV/Å, steps 60, convergence checks on) are
        therefore always sent, with the user-supplied ``relax_settings``
        input merged on top.
        """
        from aiida_vasp.utils.opthold import RelaxOptions

        settings: Dict[str, Any] = dict(RelaxOptions().model_dump())
        if "relax_settings" in self.inputs:
            settings.update(self.inputs.relax_settings.get_dict())
        return orm.Dict(dict=settings)

    def _equilibrium_structure(self):
        """Return the structure the deformations are applied to.

        With ``relax_lattice`` this is the relaxed output structure of
        the ``vasp.relax`` child (the equilibrium lattice); otherwise
        the input structure is used as-is.
        """
        relax = getattr(self.ctx, "lattice_relax", None)
        if relax is not None and relax.is_finished_ok:
            try:
                return relax.outputs["relax.structure"]
            except (KeyError, AttributeError):
                self.report(
                    "lattice relaxation finished without a relaxed structure "
                    "output — using the input structure"
                )
        return self.inputs.structure

    def generate_deformations(self):
        """Create the deformed structures and record their strains."""
        from aiida_uranium_workflow.utils.elastic import (
            apply_deformation_matrix,
            generate_deformations,
        )

        relax_lattice = (
            self.inputs.relax_lattice.value
            if "relax_lattice" in self.inputs
            else True
        )
        if relax_lattice:
            relax = getattr(self.ctx, "lattice_relax", None)
            if relax is None or not relax.is_finished_ok:
                self.report("lattice relaxation did not finish successfully")
                return self.exit_codes.ERROR_RELAX

        norm = list(self.inputs.norm_strains.get_list())
        shear = list(self.inputs.shear_strains.get_list())

        deformations = generate_deformations(norm, shear)
        self.ctx.strain_voigts = [d[0] for d in deformations]
        self.ctx.deformation_labels = [d[2] for d in deformations]

        structure = self._equilibrium_structure()
        deformed = [
            apply_deformation_matrix(structure, d[1]) for d in deformations
        ]
        self.ctx.deformed_structures = deformed
        self.report(
            f"generated {len(deformed)} deformed structures "
            f"(norm={norm}, shear={shear})"
        )

    def submit_scfs(self):
        """Submit one VaspWorkChain per deformed structure.

        The child INCAR is extended with the ISIF / IBRION / NSW of the
        chosen relax mode; everything else comes from the exposed
        VaspWorkChain inputs.
        """
        base = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )

        for idx, structure in enumerate(self.ctx.deformed_structures):
            child_inputs = dict(base)
            child_inputs["structure"] = structure
            child_inputs["parameters"] = self._relaxed_parameters(
                base.get("parameters", {}), relax_internal
            )
            child_inputs["metadata"] = {
                "label": f"elastic_{idx:03d}_{self.ctx.deformation_labels[idx]}",
                "description": (
                    "Position-relaxed SCF of a deformed structure"
                    if relax_internal
                    else "Fixed-lattice SCF of a deformed structure"
                ),
            }
            running = self.submit(ChildWorkChain, **child_inputs)
            self.report(f"submitted {running.process_label} pk={running.pk} for {idx}")
            self.to_context(**{f"scf_{idx}": running})

    @staticmethod
    def _relaxed_parameters(parameters: Any, relax_internal: bool) -> Dict[str, Any]:
        """Extend the child INCAR with the ISIF / IBRION / NSW of the relax mode."""
        if hasattr(parameters, "get_dict"):
            params = parameters.get_dict()
        else:
            params = dict(parameters)
        incar = dict(params.get("incar", {}))
        if relax_internal:
            incar.setdefault("ediffg", _FORCE_CUTOFF_EV_ANG)
            incar.update(_RELAX_INCAR)
        else:
            incar.update(_CLAMPED_INCAR)
        params["incar"] = incar
        return params

    def gather_results(self):
        """Collect stresses and fit the elastic tensor."""
        from aiida_uranium_workflow.utils.elastic import crystal_system_of

        strain_voigts = list(self.ctx.strain_voigts)
        stress_tensors = []
        all_ok = True

        for idx in range(len(self.ctx.deformed_structures)):
            child = getattr(self.ctx, f"scf_{idx}", None)
            if child is None or not child.is_finished_ok:
                all_ok = False
                continue
            try:
                misc = child.outputs.misc.get_dict()
                stress = misc.get("stress")
                if stress is None:
                    self.report(f"structure {idx}: no stress parsed")
                    all_ok = False
                    continue
                stress_tensors.append(stress)
            except (AttributeError, KeyError):
                all_ok = False

        if not all_ok:
            return self.exit_codes.ERROR_CHILD
        if len(stress_tensors) < 12:
            self.report(
                f"only {len(stress_tensors)}/{len(self.ctx.deformed_structures)} "
                "stress tensors available — need at least 12 for a fit"
            )
            return self.exit_codes.ERROR_PARSER

        self.out(
            "output_parameters",
            _combine_elastic_result(
                fit_data=orm.Dict(
                    dict={
                        "strain_voigts": [
                            sv.tolist() for sv in strain_voigts[: len(stress_tensors)]
                        ],
                        "stress_tensors": [list(s) for s in stress_tensors],
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
    link validation. VASP stresses are fitted with the default
    compression→tension flip — VASP reports positive = compression,
    exactly like ABACUS (verified against real OUTCAR/vasprun output:
    a compressed cell shows positive stress).
    """
    from aiida_uranium_workflow.utils.elastic import fit_elastic_from_stress

    data = fit_data.get_dict()
    result = fit_elastic_from_stress(
        data["strain_voigts"],
        data["stress_tensors"],
        crystal_system=data.get("crystal_system"),
    )
    return orm.Dict(dict=result)
