"""ABACUS phonon WorkChain (frozen-phonon, phonopy post-processing).

Computes the phonon band structure and DOS of a structure with the
finite-displacement method:

1. ``prepare`` — build the phonopy ``PreProcessData`` (supercell matrix,
   primitive-cell reduction, displacement generation) through the
   aiida-phonopy calcfunction helpers (provenance-safe).
2. ``run_supercells`` — submit one fixed-lattice ``AbacusBaseWorkChain``
   (SCF + forces, ``cal_force=1``) per displaced supercell.
3. ``inspect_forces`` — fail fast if any SCF child failed.
4. ``run_phonopy`` — collect the per-supercell forces (eV/Å) into a
   ``PhonopyData`` and submit the ``phonopy.phonopy`` CalcJob that fits
   the force constants and computes bands / DOS.
5. ``gather_results`` — forward the phonopy outputs (``phonon_bands``,
   ``total_phonon_dos``, ``force_constants``, ``phonopy_data``) and a
   ``output_parameters`` summary.

The ABACUS SCF base (``base`` namespace) is assembled by
:class:`aiida_uranium_workflow.input_builders.phonopy.abacus.AbacusPhonopyAdapter`
from ``parameters/abacus/scf.yml``; the phonopy-specific settings come
from ``parameters/phonopy.yml``.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory

import numpy as np

PreProcessData = DataFactory("phonopy.preprocess")
PhonopyData = DataFactory("phonopy.phonopy")
# NOTE: use CalculationFactory — ``WorkflowFactory("phonopy.phonopy")``
# resolves to the aiida-phonopy **PhonopyWorkChain** (the same entry
# point name is registered in both the ``aiida.workflows`` and
# ``aiida.calculations`` groups), which has no ``metadata.options``
# ports and would fail validation at submission.
PhonopyCalculation = CalculationFactory("phonopy.phonopy")
# The plugin's own ``abacus.base`` entry point (SCF + forces child).
_PluginBaseWorkChain = WorkflowFactory("abacus.base")


class AbacusPhonopyWorkChain(WorkChain):
    """Compute phonon bands / DOS with ABACUS forces + phonopy post-processing.

    Outline: prepare → run_supercells → inspect_forces → run_phonopy →
    gather_results.

    Exit codes
    ----------
    * 0   ``SUCCESS``      — phonon calculation finished.
    * 300 ``ERROR_CHILD``  — an SCF child or the phonopy CalcJob failed.
    * 305 ``ERROR_PARSER`` — could not read the phonopy outputs.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- Top-level inputs --------------------------------------------
        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input_namespace(
            "base",
            dynamic=True,
            help="AbacusBaseWorkChain inputs (SCF base from scf.yml).",
        )
        # phonopy pre-processing
        spec.input(
            "supercell_matrix",
            valid_type=orm.List,
            required=False,
            help="Supercell transformation of the unit cell (3 or 3x3 list).",
        )
        spec.input(
            "primitive_matrix",
            valid_type=(orm.Str, orm.List),
            required=False,
            help=(
                "Phonopy primitive matrix: 'auto' (reduce input to primitive "
                "cell), an explicit 3x3 list, or omit to keep the input cell."
            ),
        )
        spec.input("symprec", valid_type=orm.Float, required=False)
        spec.input("is_symmetry", valid_type=orm.Bool, required=False)
        spec.input("distinguish_kinds", valid_type=orm.Bool, required=False)
        spec.input(
            "displacement_generator",
            valid_type=orm.Dict,
            required=False,
            help="Displacement settings (distance / is_plusminus / ...).",
        )
        # phonopy post-processing
        spec.input("phonopy_code", valid_type=orm.InstalledCode, required=True)
        spec.input(
            "phonopy_parameters",
            valid_type=orm.Dict,
            required=True,
            help=(
                "Phonopy setting tags for the band / DOS run "
                "(band | band_paths, band_points, dos, mesh, fmin, ...)."
            ),
        )
        spec.input(
            "phonopy_settings",
            valid_type=orm.Dict,
            required=False,
            help="Phonopy CalcJob settings (keep_phonopy_yaml, ...).",
        )
        spec.input(
            "phonopy_options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler options for the phonopy CalcJob.",
        )
        spec.input(
            "band_labels",
            valid_type=orm.List,
            required=False,
            help=(
                "High-symmetry labels of the band path (manual mode); used "
                "by the report / plot to annotate the figure."
            ),
        )

        # ---- Outputs ------------------------------------------------------
        spec.output(
            "phonon_bands",
            valid_type=orm.BandsData,
            required=True,
            help="Phonon band structure (THz) from the phonopy CalcJob.",
        )
        spec.output(
            "total_phonon_dos",
            valid_type=orm.XyData,
            required=False,
            help="Total phonon DOS from the phonopy CalcJob.",
        )
        spec.output(
            "force_constants",
            valid_type=orm.ArrayData,
            required=False,
            help="Force constants (ArrayData) from the phonopy CalcJob.",
        )
        spec.output(
            "phonopy_data",
            valid_type=PhonopyData,
            required=False,
            help="PhonopyData (preprocess + force sets) used for the run.",
        )
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.prepare,
            cls.run_supercells,
            cls.inspect_forces,
            cls.run_phonopy,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "Phonon calculation completed.")
        spec.exit_code(300, "ERROR_CHILD", "An SCF child or phonopy CalcJob failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to read the phonopy outputs.")

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def prepare(self):
        """Create the phonopy PreProcessData (calcfunction, stored).

        When ``primitive_matrix`` is requested (``"auto"`` or an explicit
        matrix), the **input structure is reduced to its minimal primitive
        cell first** and the supercell is built from that primitive — this
        keeps the SCF cells as small as possible (e.g. bcc-U 3×3×3 of the
        2-atom conventional cell = 54 atoms, but 2×2×2 of the 1-atom
        primitive = 8 atoms) and is the only way the supercell SCFs stay
        tractable. The band path / DOS then live in the primitive
        reciprocal space (the phonopy convention).
        """
        kwargs = {"structure": self.inputs.structure}
        if "primitive_matrix" in self.inputs:
            # Reduce to the primitive cell, then treat it as the unit cell
            # (primitive_matrix=None — it is already minimal).
            kwargs["structure"] = _to_primitive(
                structure=self.inputs.structure,
                primitive_matrix=self.inputs.primitive_matrix,
            )
        if "supercell_matrix" in self.inputs:
            kwargs["supercell_matrix"] = self.inputs.supercell_matrix
        if "symprec" in self.inputs:
            kwargs["symprec"] = self.inputs.symprec
        if "is_symmetry" in self.inputs:
            kwargs["is_symmetry"] = self.inputs.is_symmetry
        if "distinguish_kinds" in self.inputs:
            kwargs["distinguish_kinds"] = self.inputs.distinguish_kinds
        if "displacement_generator" in self.inputs:
            kwargs["displacement_generator"] = self.inputs.displacement_generator

        preprocess = PreProcessData.generate_preprocess_data(**kwargs)
        self.ctx.preprocess_data = preprocess
        self.report(f"stored PreProcessData pk={preprocess.pk}")

    def run_supercells(self):
        """Generate the displaced supercells and submit one SCF per cell."""
        supercells = (
            self.ctx.preprocess_data.calcfunctions.get_supercells_with_displacements()
        )
        self.ctx.supercells = dict(supercells)
        self.ctx.n_supercells = len(supercells)
        self.report(f"generated {len(supercells)} displaced supercell(s)")

        base = dict(self.inputs.base) if "base" in self.inputs else {}
        for label, supercell in supercells.items():
            child_inputs = dict(base)
            child_inputs["abacus"]["structure"] = supercell
            child_inputs["metadata"] = {
                "label": f"phonon_{label}",
                "description": "Fixed-lattice SCF with forces of a displaced supercell",
            }
            running = self.submit(_PluginBaseWorkChain, **child_inputs)
            self.report(f"submitted SCF pk={running.pk} for {label}")
            self.to_context(**{f"force_{label}": running})

    def inspect_forces(self):
        """Fail fast if any displaced-supercell SCF did not finish OK."""
        for label in self.ctx.supercells:
            child = getattr(self.ctx, f"force_{label}", None)
            if child is None or not child.is_finished_ok:
                self.report(
                    f"SCF child for {label} "
                    f"pk={getattr(child, 'pk', None)} failed "
                    f"(state={getattr(child, 'process_state', None)}, "
                    f"exit={getattr(child, 'exit_status', None)})"
                )
                return self.exit_codes.ERROR_CHILD
        self.report("all displaced-supercell SCFs finished OK")
        return None

    def run_phonopy(self):
        """Assemble the PhonopyData and submit the phonopy CalcJob."""
        forces = {}
        for label in self.ctx.supercells:
            suffix = label.rsplit("_", 1)[-1]
            child = getattr(self.ctx, f"force_{label}")
            forces[f"forces_{suffix}"] = _extract_forces(misc=child.outputs.misc)

        phonopy_data = self.ctx.preprocess_data.calcfunctions.generate_phonopy_data(
            **forces
        )
        self.ctx.phonopy_data = phonopy_data
        self.report(f"assembled PhonopyData pk={phonopy_data.pk}")

        options = {}
        if "phonopy_options" in self.inputs:
            options = dict(self.inputs.phonopy_options.get_dict())
        # The plugin defaults to no MPI; align with the registered code so
        # presubmit doesn't reject the job for inconsistent MPI usage.
        # (``with_mpi`` is stored as a code attribute in aiida-core 2.8.)
        options.setdefault(
            "withmpi",
            bool(self.inputs.phonopy_code.base.attributes.get("with_mpi", False)),
        )
        options.setdefault("resources", {"num_machines": 1, "tot_num_mpiprocs": 1})

        inputs = {
            "code": self.inputs.phonopy_code,
            "phonopy_data": phonopy_data,
            "parameters": self.inputs.phonopy_parameters,
            "metadata": {
                "label": "phonopy-postprocess",
                "description": "Phonopy band structure + DOS post-processing",
                "options": options,
            },
        }
        if "phonopy_settings" in self.inputs:
            inputs["settings"] = self.inputs.phonopy_settings

        running = self.submit(PhonopyCalculation, **inputs)
        self.report(f"submitted PhonopyCalculation pk={running.pk}")
        self.to_context(phonopy_run=running)

    def gather_results(self):
        """Forward the phonopy outputs and build the summary."""
        calc = getattr(self.ctx, "phonopy_run", None)
        if calc is None:
            return self.exit_codes.ERROR_PARSER
        if not calc.is_finished_ok:
            self.report(
                f"PhonopyCalculation pk={calc.pk} failed "
                f"(state={calc.process_state}, exit={calc.exit_status})"
            )
            return self.exit_codes.ERROR_CHILD

        outputs = {
            link.link_label: link.node
            for link in calc.base.links.get_outgoing().all()
            if link.link_type.value == "create"
        }
        if "phonon_bands" not in outputs:
            self.report(
                f"phonopy calc pk={calc.pk} has no phonon_bands output; "
                f"available: {sorted(outputs)}"
            )
            return self.exit_codes.ERROR_PARSER

        self.out("phonon_bands", outputs["phonon_bands"])
        if "total_phonon_dos" in outputs:
            self.out("total_phonon_dos", outputs["total_phonon_dos"])
        if "output_force_constants" in outputs:
            self.out("force_constants", outputs["output_force_constants"])
        self.out("phonopy_data", self.ctx.phonopy_data)

        self.out(
            "output_parameters",
            _summarize(
                preprocess_data=self.ctx.preprocess_data,
                bands=outputs["phonon_bands"],
                dos=outputs.get("total_phonon_dos"),
                phonopy_para=outputs.get("output_parameters"),
                phonopy_parameters=self.inputs.phonopy_parameters,
                n_supercells=self.ctx.n_supercells,
                structure_formula=self.inputs.structure.get_formula(),
                supercell_matrix=(
                    list(self.inputs.supercell_matrix.get_list())
                    if "supercell_matrix" in self.inputs
                    else None
                ),
                primitive_matrix=(
                    self.inputs.primitive_matrix.value
                    if "primitive_matrix" in self.inputs
                    and isinstance(self.inputs.primitive_matrix, orm.Str)
                    else (
                        list(self.inputs.primitive_matrix.get_list())
                        if "primitive_matrix" in self.inputs
                        else None
                    )
                ),
                symprec=(
                    float(self.inputs.symprec.value)
                    if "symprec" in self.inputs
                    else None
                ),
                band_labels=(
                    list(self.inputs.band_labels.get_list())
                    if "band_labels" in self.inputs
                    else None
                ),
                phonopy_pk=calc.pk,
                phonopy_uuid=str(calc.uuid),
            ),
        )
        return None  # SUCCESS


# ---------------------------------------------------------------------------
# calcfunctions
# ---------------------------------------------------------------------------


@calcfunction
def _to_primitive(structure, primitive_matrix):
    """Reduce a StructureData to its minimal primitive cell (provenance-safe).

    Uses phonopy's ``Phonopy(unitcell, primitive_matrix=...)`` machinery —
    ``"auto"`` picks the primitive cell via spglib, or an explicit 3×3
    matrix can be given. The band structure / DOS of the phonon run then
    live in the primitive reciprocal space.
    """
    from aiida_phonopy.calculations.functions.link_structures import (
        phonopy_atoms_from_structure,
        phonopy_atoms_to_structure,
    )
    from phonopy import Phonopy

    if isinstance(primitive_matrix, orm.Str):
        pm = primitive_matrix.value
    else:
        pm = list(primitive_matrix.get_list())

    unit, _mapping = phonopy_atoms_from_structure(structure)
    ph = Phonopy(unit, primitive_matrix=pm)
    primitive = ph.primitive
    return phonopy_atoms_to_structure(primitive)


@calcfunction
def _extract_forces(misc):
    """Extract ``final_forces`` (eV/Å) from an ABACUS misc Dict into an ArrayData."""
    forces = misc.get_dict().get("final_forces")
    if forces is None:
        raise ValueError(
            "ABACUS misc output has no 'final_forces' — make sure the SCF "
            "input enables force calculation (cal_force=1)."
        )
    array = orm.ArrayData()
    array.set_array("forces", np.asarray(forces, dtype=float))
    return array


@calcfunction
def _summarize(
    preprocess_data,
    bands,
    phonopy_parameters,
    dos=None,
    phonopy_para=None,
    n_supercells=None,
    structure_formula=None,
    supercell_matrix=None,
    primitive_matrix=None,
    symprec=None,
    band_labels=None,
    phonopy_pk=None,
    phonopy_uuid=None,
):
    """Build the ``output_parameters`` Dict of the phonon WorkChain.

    ``dos`` / ``phonopy_para`` are optional Data nodes and may be ``None``
    (plain keyword values are passed through without being stored).
    """
    freqs = np.asarray(bands.get_bands())
    phonopy_settings = phonopy_parameters.get_dict()

    result = {
        "workflow": "phonopy",
        "backend": "abacus",
        "structure_formula": structure_formula,
        "n_supercells": n_supercells,
        "supercell_matrix": supercell_matrix,
        "primitive_matrix": primitive_matrix,
        "symprec": symprec,
        "band_labels": band_labels,
        "phonopy_parameters": phonopy_settings,
        "frequency_min_thz": float(freqs.min()) if freqs.size else None,
        "frequency_max_thz": float(freqs.max()) if freqs.size else None,
        "n_imaginary_modes": int((freqs < 0).sum()) if freqs.size else None,
        "phonopy_pk": phonopy_pk,
        "phonopy_uuid": phonopy_uuid,
    }
    if phonopy_para is not None:
        result["phonopy_calc_parameters"] = dict(phonopy_para.get_dict())
    return orm.Dict(dict=result)
