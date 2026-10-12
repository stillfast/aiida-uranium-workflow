"""QE (pw.x) elastic-constants WorkChain (stress method).

Generates 24 deformed structures (Materials Project methodology:
3 normal strains × ``norm_strains`` + 3 shear strains ×
``shear_strains``), runs one ``PwBaseWorkChain`` per deformed cell and
fits the full 6×6 elastic tensor via pymatgen's
``ElasticTensor.from_independent_strains`` from the stress tensors.

**Stress source and units**: QE prints the stress tensor for every
ionic step; aiida-qe parses it into the ``output_trajectory``'s
``stress`` array (shape ``(n_steps, 3, 3)``) in **GPa** (converted from
atomic units). The workflow reads the *last* frame — the stress of the
converged SCF / relaxed ionic step — and converts it back to **kbar**
(1 kbar = 0.1 GPa) for the shared fit helper.

**Stress sign convention**: like ABACUS and VASP, QE's "total stress"
uses positive = compression for the hydrostatic part (the fitted
tensor must be negated — the default ``compression_positive=True``).
If your QE build reports tension-positive stresses instead, the fitted
constants come out negative; check one deformed state's raw stress
sign against the applied strain before trusting the numbers.

By default the internal atomic coordinates are relaxed under each
strain (``relax_internal`` — the child pw parameters get
``CONTROL.calculation = 'relax'``; QE relaxes the ions with the cell
fixed by the strained lattice unless ``vc-relax`` is requested). For
cells without free internal coordinates (e.g. the 2-atom bcc U cell)
the relaxation is a no-op. Set ``relax_internal=False`` for the
clamped-ion constants (fixed-lattice ``'scf'`` children).

The pw (SCF / relax) parameters come from the per-backend preset
(``parameters/qe/elastic.yml``); the strain lists and the
``relax_internal`` switch come from the workflow protocol
(``parameters/elastic.yml``).

Layout::

    inputs
    ├── structure              — the equilibrium cell the strains act on
    ├── pw / kpoints (or kpoints_distance)
    │       — PwBaseWorkChain inputs (``pw.structure`` is overridden per
    │         deformed cell; ``pw.parameters`` gains the relax mode)
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

# The aiida-qe ``PwBaseWorkChain`` (``quantumespresso.pw.base``).
ChildWorkChain = WorkflowFactory("quantumespresso.pw.base")

#: pw CONTROL injected for the chosen relax mode. ``relax_internal``
#: (default) relaxes the ions under the strained cell
#: (``calculation = 'relax'`` — QE keeps the cell fixed unless
#: ``vc-relax`` is requested); otherwise a fixed-lattice SCF.
_RELAX_CALCULATION = "relax"
_CLAMPED_CALCULATION = "scf"


class QeElasticWorkChain(WorkChain):
    """Compute elastic constants of a structure with QE (stress method)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=[
                "pw",
                "kpoints",
                "kpoints_distance",
            ],
        )
        spec.input("structure", valid_type=orm.StructureData, required=True)
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
                "(child pw ``CONTROL.calculation = 'relax'``; QE moves the "
                "ions with the cell fixed by the strained lattice) so the "
                "fitted constants are the physically relevant 'relaxed' "
                "ones. For cells with no free internal coordinates (e.g. "
                "the 2-atom bcc U cell) it is a no-op. Set to False for "
                "clamped-ion constants (fixed-lattice SCF children)."
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
            apply_deformation_matrix(self.inputs.structure, d[1])
            for d in deformations
        ]
        self.ctx.deformed_structures = deformed
        self.report(
            f"generated {len(deformed)} deformed structures "
            f"(norm={norm}, shear={shear})"
        )

    def submit_scfs(self):
        """Submit one PwBaseWorkChain per deformed structure.

        The child's ``pw.structure`` is the deformed cell; the pw
        parameters get ``CONTROL.calculation`` = 'relax' (or 'scf').
        """
        base = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )
        calculation = _RELAX_CALCULATION if relax_internal else _CLAMPED_CALCULATION

        for idx, structure in enumerate(self.ctx.deformed_structures):
            child_inputs = dict(base)
            pw = dict(base.get("pw", {}))
            parameters = pw.get("parameters", {})
            if hasattr(parameters, "get_dict"):
                params = parameters.get_dict()
            else:
                params = dict(parameters)
            control = params.setdefault("CONTROL", {})
            control["calculation"] = calculation
            # The stress method needs the stress tensor of every deformed
            # cell; QE skips it for plain scf/relax unless tstress is set.
            control.setdefault("tstress", True)
            pw["parameters"] = orm.Dict(dict=params)
            pw["structure"] = structure
            child_inputs["pw"] = pw
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

    def gather_results(self):
        """Collect the last-frame stresses and fit the elastic tensor."""
        from aiida_uranium_workflow.utils.elastic import crystal_system_of

        strain_voigts = list(self.ctx.strain_voigts)
        stress_tensors_kbar = []
        all_ok = True

        for idx in range(len(self.ctx.deformed_structures)):
            child = getattr(self.ctx, f"scf_{idx}", None)
            if child is None or not child.is_finished_ok:
                all_ok = False
                continue
            try:
                trajectory = child.outputs.output_trajectory
                stress_gpa = trajectory.get_array("stress")
                if stress_gpa is None or len(stress_gpa) == 0:
                    self.report(f"structure {idx}: no stress parsed")
                    all_ok = False
                    continue
                # Last ionic step's 3×3 stress, GPa → kbar (1 kbar = 0.1 GPa).
                stress_tensors_kbar.append(
                    (stress_gpa[-1] * 10.0).tolist()
                )
            except (AttributeError, KeyError):
                all_ok = False

        if not all_ok:
            return self.exit_codes.ERROR_CHILD
        if len(stress_tensors_kbar) < 12:
            self.report(
                f"only {len(stress_tensors_kbar)}/{len(self.ctx.deformed_structures)} "
                "stress tensors available — need at least 12 for a fit"
            )
            return self.exit_codes.ERROR_PARSER

        self.out(
            "output_parameters",
            _combine_elastic_result(
                fit_data=orm.Dict(
                    dict={
                        "strain_voigts": [
                            sv.tolist()
                            for sv in strain_voigts[: len(stress_tensors_kbar)]
                        ],
                        "stress_tensors": stress_tensors_kbar,
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
    link validation. QE stresses are fitted with the default
    compression→tension flip (positive = compression, like ABACUS /
    VASP) — verify the raw stress sign before trusting the result.
    """
    from aiida_uranium_workflow.utils.elastic import fit_elastic_from_stress

    data = fit_data.get_dict()
    result = fit_elastic_from_stress(
        data["strain_voigts"],
        data["stress_tensors"],
        crystal_system=data.get("crystal_system"),
    )
    return orm.Dict(dict=result)
