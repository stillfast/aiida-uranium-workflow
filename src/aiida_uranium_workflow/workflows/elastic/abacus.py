"""ABACUS elastic-constants WorkChain (stress method).

Generates 24 deformed structures (Materials Project methodology:
3 normal strains × ``norm_strains`` + 3 shear strains ×
``shear_strains``), runs a fixed-lattice SCF for each, reads the
``TOTAL-STRESS`` tensor from each calculation, and fits the full 6×6
elastic tensor via pymatgen's ``ElasticTensor.from_independent_strains``.
Aggregate moduli (K, G, anisotropy, Poisson) are derived from it.

By default the internal atomic coordinates are relaxed under each
strain (``relax_internal``, an ``abacus.relax`` child with
``relax_type='positions'`` — the cell stays fixed at the strained
lattice). This gives the physically relevant "relaxed" elastic
constants and matches the official ABACUS elastic example (one
``calculation relax`` per deformed cell). For cells without free
internal coordinates (e.g. the 2-atom bcc U cell) the relaxation is a
no-op. Set ``relax_internal=False`` for the clamped-ion constants
(fixed-lattice SCF children, the previous behaviour).

The SCF base comes from ``parameters/abacus/scf.yml``; the strain lists
come from the workflow protocol (``parameters/elastic.yml``).

Layout::

    inputs
    ├── code / structure
    ├── base            (namespace) — AbacusBaseWorkChain inputs (SCF base)
    ├── norm_strains    (List) — normal strain magnitudes
    ├── shear_strains   (List) — shear strain magnitudes
    ├── relax_internal  (Bool) — relax internal coords under strain (default True)
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

        deformations = generate_deformations(norm, shear)
        self.ctx.strain_voigts = [d[0] for d in deformations]
        self.ctx.deformation_labels = [d[2] for d in deformations]

        deformed = [
            apply_deformation_matrix(self.inputs.structure, d[1]) for d in deformations
        ]
        self.ctx.deformed_structures = deformed
        self.report(
            f"generated {len(deformed)} deformed structures "
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

        base_ns: Dict[str, Any] = {
            k: v for k, v in base.items() if k != "abacus"
        }
        base_ns["abacus"] = {
            k: v for k, v in base_abacus.items() if k != "parameters"
        }
        base_ns["abacus"]["parameters"] = orm.Dict(dict=params)

        return {
            "structure": structure,
            "base": base_ns,
            "relax_settings": orm.Dict(dict=DEFAULT_RELAX_SETTINGS),
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
    return orm.Dict(dict=result)
