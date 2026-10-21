"""ABACUS elastic-constants WorkChain (stress method).

Generates 24 deformed structures (Materials Project methodology:
3 normal strains × ``norm_strains`` + 3 shear strains ×
``shear_strains``), runs a fixed-lattice SCF for each, reads the
``TOTAL-STRESS`` tensor from each calculation, and fits the full 6×6
elastic tensor via pymatgen's ``ElasticTensor.from_independent_strains``.
Aggregate moduli (K, G, anisotropy, Poisson) are derived from it.

**Lattice relaxation** (``relax_lattice``): before the deformations are
generated, the input structure can be fully relaxed (cell + ions) by an
``abacus.relax`` child with the protocol's ``lattice_relax`` RelaxOptions
— the ABACUS counterpart of the official example's
``prepare_elastic.sh``/``../OPT/`` cell-relax, and the same layout as the
VASP side. A failed pre-relaxation does not abort the workflow: the
strained cells are computed from the input structure and the result
records ``equilibrium_structure_source: input`` / ``lattice_relaxed:
false`` (``require_lattice_relax=True`` aborts with ``ERROR_RELAX``
instead).

By default the internal atomic coordinates are relaxed under each
strain (``relax_internal``, an ``abacus.relax`` child with
``relax_type='positions'`` — the cell stays fixed at the strained
lattice, RelaxOptions from the protocol's ``child_relax`` block). This
gives the physically relevant "relaxed" elastic constants and matches
the official ABACUS elastic example (one ``calculation relax`` per
deformed cell). The force criterion matters: a ±1 % shear of a soft-mode
metal (α-U) develops only ~0.03 eV/Å, so a loose ``force_cutoff`` leaves
the cell clamped and the shear constants ~2× too stiff. For cells
without free internal coordinates (e.g. the 2-atom bcc U cell) the
relaxation is a no-op. Set ``relax_internal=False`` for the clamped-ion
constants (fixed-lattice SCF children).

The SCF base comes from ``parameters/abacus/scf.yml``; the strain lists
come from the workflow protocol (``parameters/elastic.yml``).

Layout::

    inputs
    ├── code / structure
    ├── base            (namespace) — AbacusBaseWorkChain inputs (SCF base)
    ├── norm_strains    (List) — normal strain magnitudes
    ├── shear_strains   (List) — shear strain magnitudes
    ├── relax_internal  (Bool) — relax internal coords under strain (default True)
    ├── relax_lattice   (Bool) — full lattice pre-relaxation (default False)
    ├── require_lattice_relax (Bool) — abort when the pre-relaxation fails
    ├── relax_settings  (Dict) — RelaxOptions of the lattice pre-relaxation
    │                            (parameters/elastic.yml ``abacus.lattice_relax``)
    ├── child_relax_settings (Dict) — RelaxOptions of the strained cells
    │                            (``abacus.child_relax``)
    outputs
    └── output_parameters (Dict) — elastic tensor + moduli (GPa)
"""

from __future__ import annotations

from typing import Any, Dict

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida.plugins import WorkflowFactory

# The plugin's own ``abacus.base`` / ``abacus.relax`` entry points.
_PluginBaseWorkChain = WorkflowFactory("abacus.base")
_PluginRelaxWorkChain = WorkflowFactory("abacus.relax")

#: Default abacus.relax settings for the strained cells: relax atomic
#: positions only (the cell is fixed by the strain) with the same
#: force criterion as the relax protocol (0.05 eV/Å).
DEFAULT_RELAX_SETTINGS: Dict[str, Any] = {
    "relax_type": "positions",  # RelaxType.POSITIONS: atoms only, cell fixed
    "perform": True,
    "relax_method": "cg",
    "max_ionic_steps": 50,
    "force_cutoff": 0.05,  # eV/Å (force_thr_ev)
    "stress_cutoff": 1.0,  # kBar (stress_thr; unused for positions relax)
}


#: Fallback RelaxOptions of the lattice pre-relaxation child (used when the
#: protocol has no ``abacus.lattice_relax`` block): full cell + ions relax.
DEFAULT_LATTICE_RELAX_SETTINGS: Dict[str, Any] = {
    "relax_type": "positions_cell",
    "relax_method": "cg",
    "max_ionic_steps": 50,
    "force_cutoff": 0.03,
    "stress_cutoff": 1.0,
    "meta_convergence": True,
    "volume_convergence": 0.01,
    "max_meta_convergence_iterations": 5,
}


class AbacusElasticWorkChain(WorkChain):
    """Compute elastic constants of a structure with ABACUS (stress method)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input_namespace("base", dynamic=True, help="AbacusBaseWorkChain inputs.")
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
            "relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "``abacus.relax`` RelaxOptions of the *lattice* pre-relaxation "
                "(the ``abacus.lattice_relax`` block of "
                "``parameters/elastic.yml``: relax_type / relax_method / "
                "max_ionic_steps / force_cutoff / stress_cutoff / "
                "meta_convergence / volume_convergence / …)."
            ),
        )
        spec.input(
            "relax_lattice",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Fully relax the lattice (cell + ions) with an "
                "``abacus.relax`` child before generating the deformations, "
                "and build the strained cells from the relaxed structure — "
                "the ABACUS counterpart of the official elastic example's "
                "``prepare_elastic.sh``/``../OPT/`` cell-relax. Default False: "
                "deform the input structure as-is."
            ),
        )
        spec.input(
            "require_lattice_relax",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Abort with ``ERROR_RELAX`` when the lattice relaxation fails. "
                "Default False: the strained cells are still computed from the "
                "input structure and the result records "
                "``equilibrium_structure_source: input``."
            ),
        )
        spec.input(
            "child_relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "``abacus.relax`` RelaxOptions of the strained-cell children "
                "(the ``abacus.child_relax`` block of "
                "``parameters/elastic.yml``). Defaults to "
                "DEFAULT_RELAX_SETTINGS (positions only, cg, 50 steps, "
                "force_cutoff 0.05 eV/Å)."
            ),
        )
        spec.input(
            "relax_internal",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Relax the internal atomic coordinates under each strain "
                "(abacus.relax, relax_type='positions', cell fixed by the "
                "strain) so the fitted constants are the physically relevant "
                "'relaxed' ones — the method of the official ABACUS elastic "
                "example. For cells with no free internal coordinates "
                "(e.g. the 2-atom bcc U cell) it is a no-op. Set to False "
                "for clamped-ion constants (fixed-lattice SCF children)."
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

        With ``relax_lattice`` an ``abacus.relax`` child (the plugin relax
        WorkChain) is submitted with the protocol's ``lattice_relax``
        RelaxOptions; the engine waits for it, so ``ctx.lattice_relax`` holds
        the finished child in :meth:`generate_deformations`.
        """
        relax_lattice = (
            self.inputs.relax_lattice.value if "relax_lattice" in self.inputs else False
        )
        if not relax_lattice:
            self.report("skipping lattice relaxation (relax_lattice=False)")
            return

        base = dict(self.inputs.base) if "base" in self.inputs else {}
        settings = (
            dict(self.inputs.relax_settings.get_dict())
            if "relax_settings" in self.inputs
            else dict(DEFAULT_LATTICE_RELAX_SETTINGS)
        )
        relax_inputs = {
            "structure": self.inputs.structure,
            "base": base,
            "relax_settings": orm.Dict(dict=settings),
            "metadata": {
                "label": "lattice_relax",
                "description": (
                    "Full lattice relaxation (cell + ions) of the input "
                    "structure before the elastic deformations."
                ),
            },
        }
        running = self.submit(_PluginRelaxWorkChain, **relax_inputs)
        self.report(
            f"submitted lattice relaxation {running.process_label} pk={running.pk}"
        )
        self.to_context(lattice_relax=running)

    def _equilibrium_structure(self):
        """Structure the deformations are applied to (relaxed or input)."""
        relax = getattr(self.ctx, "lattice_relax", None)
        if relax is not None and relax.is_finished_ok:
            try:
                return relax.outputs.structure
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

        norm = list(self.inputs.norm_strains.get_list())
        shear = list(self.inputs.shear_strains.get_list())

        relax_lattice = (
            self.inputs.relax_lattice.value if "relax_lattice" in self.inputs else False
        )
        if relax_lattice:
            relax = getattr(self.ctx, "lattice_relax", None)
            if relax is None or not relax.is_finished_ok:
                if relax is None:
                    message = "lattice relaxation was never submitted"
                else:
                    message = (
                        f"lattice relaxation <{relax.pk}> finished with "
                        f"exit status {relax.exit_status}"
                    )
                require = (
                    self.inputs.require_lattice_relax.value
                    if "require_lattice_relax" in self.inputs
                    else False
                )
                if require:
                    self.report(f"{message} — aborting (require_lattice_relax=True)")
                    return self.exit_codes.ERROR_RELAX
                self.report(
                    f"{message} — CONTINUING with the input structure (NOT "
                    "relaxed); the reported tensor is not on the equilibrium "
                    "lattice. Set require_lattice_relax=true to abort instead."
                )

        deformations = generate_deformations(norm, shear)
        self.ctx.strain_voigts = [d[0] for d in deformations]
        self.ctx.deformation_labels = [d[2] for d in deformations]

        structure = self._equilibrium_structure()
        self.ctx.equilibrium_structure_source = (
            "lattice_relax" if structure is not self.inputs.structure else "input"
        )
        deformed = [apply_deformation_matrix(structure, d[1]) for d in deformations]
        self.ctx.deformed_structures = deformed
        self.report(
            f"generated {len(deformed)} deformed structures "
            f"from the {self.ctx.equilibrium_structure_source} structure "
            f"(norm={norm}, shear={shear})"
        )

    def submit_scfs(self):
        """Submit one SCF / position-relax per deformed structure.

        With ``relax_internal`` the child is an ``abacus.relax``
        (``relax_type='positions'`` — atoms relax, the cell stays fixed
        at the strained lattice); otherwise a fixed-lattice
        ``abacus.base`` SCF as before.
        """
        base = dict(self.inputs.base) if "base" in self.inputs else {}
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )

        for idx, structure in enumerate(self.ctx.deformed_structures):
            if relax_internal:
                child_inputs = self._relax_inputs_for(structure, base)
            else:
                child_inputs = self._scf_inputs_for(structure, base)
            child_inputs["metadata"] = {
                "label": f"elastic_{idx:03d}_{self.ctx.deformation_labels[idx]}",
                "description": (
                    "Position-relaxed SCF of a deformed structure"
                    if relax_internal
                    else "Fixed-lattice SCF of a deformed structure"
                ),
            }
            running = self.submit(
                _PluginRelaxWorkChain if relax_internal else _PluginBaseWorkChain,
                **child_inputs,
            )
            self.report(f"submitted {running.process_label} pk={running.pk} for {idx}")
            self.to_context(**{f"scf_{idx}": running})

    def _scf_inputs_for(self, structure, base):
        """Compose an ``abacus.base`` fixed-lattice SCF for a deformed cell."""
        child_inputs = {k: v for k, v in base.items() if k != "abacus"}
        abacus = dict(base["abacus"])
        abacus["structure"] = structure
        child_inputs["abacus"] = abacus
        return child_inputs

    def _relax_inputs_for(self, structure, base):
        """Compose an ``abacus.relax`` position-relax child for a deformed cell.

        Mirrors the defect workflow's relax children: ``structure`` at the
        top level, the SCF base in the ``base`` namespace (without a
        structure — the plugin sets it per iteration), the cell fixed by
        the strained lattice, and ``meta_convergence`` off (the volume
        comparison is meaningless for a fixed cell).
        """
        base_abacus = base.get("abacus", {}) if base else {}
        params = (
            dict(base_abacus["parameters"].get_dict())
            if "parameters" in base_abacus
            else {}
        )
        params.setdefault("input", {})
        params["input"]["calculation"] = "relax"
        params["input"]["cal_stress"] = 1  # stress output stays on

        base_ns: Dict[str, Any] = {k: v for k, v in base.items() if k != "abacus"}
        base_ns["abacus"] = {k: v for k, v in base_abacus.items() if k != "parameters"}
        base_ns["abacus"]["parameters"] = orm.Dict(dict=params)

        settings = (
            self.inputs.child_relax_settings
            if "child_relax_settings" in self.inputs
            else orm.Dict(dict=DEFAULT_RELAX_SETTINGS)
        )
        return {
            "structure": structure,
            "base": base_ns,
            "relax_settings": settings,
            "meta_convergence": orm.Bool(False),
            "clean_workdir": orm.Bool(False),
        }

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
                stress = misc.get("final_stress")
                if stress is None:
                    all_stress = misc.get("all_stress") or []
                    stress = all_stress[-1] if all_stress else None
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
                        # which lattice the deformations were applied to
                        "equilibrium_structure_source": getattr(
                            self.ctx, "equilibrium_structure_source", "input"
                        ),
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
    from aiida_uranium_workflow.utils.elastic import fit_elastic_from_stress

    data = fit_data.get_dict()
    result = fit_elastic_from_stress(
        data["strain_voigts"],
        data["stress_tensors"],
        crystal_system=data.get("crystal_system"),
    )
    source = data.get("equilibrium_structure_source", "input")
    result["equilibrium_structure_source"] = source
    result["lattice_relaxed"] = source == "lattice_relax"
    return orm.Dict(dict=result)
