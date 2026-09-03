"""FLEUR phonon WorkChain (frozen-phonon, phonopy post-processing).

Computes the phonon band structure and DOS of a structure with the
finite-displacement method, using FLEUR forces:

1. ``prepare`` — build the phonopy ``PreProcessData`` (supercell matrix,
   primitive-cell reduction, displacement generation).
2. ``run_supercells`` — submit one ``FleurScfWorkChain`` per displaced
   supercell. The SCF runs in **density** mode with the geometry
   optimisation switched on explicitly (``l_f="T"`` via an inpxml
   change): FLEUR computes atomic forces only when the geometry
   optimisation is active (fleur.md §4.5: "all force contributions are
   only calculated if a structural optimization of the atom positions is
   activated"), and aiida-fleur exposes them per SCF iteration in
   ``out.xml`` as ``output_parameters['force_atoms']`` (per atom
   ``[fx, fy, fz]`` in Htr/bohr).

   aiida-fleur's ``mode='force'`` is *not* used here: it declares
   convergence only once the underlying run produced a ``relax.xml``
   (relax_parameters), which a fixed-lattice run (``qfix``) never
   writes — it would spin until ``fleur_runmax`` and exit 362. Density
   mode converges the SCF and returns, and the forces are already in
   ``force_atoms``.

   Because FLEUR only reports forces for the *representative* atom of
   each symmetry group (``totalForcesOnRepresentativeAtoms``), every
   displaced supercell is first passed through ``_break_fleur_symmetry``:
   each atom receives its own species (fractional atomic number, e.g.
   ``92.1``, ``92.2``, ..., fleur.md §3.6), so the input generator builds
   one atom group per atom and only the identity symmetry operation
   survives (P1). Otherwise a symmetry-reduced supercell yields fewer
   force rows than atoms and phonopy fails with "Shape mismatch between
   displacements and forces".
3. ``inspect_forces`` — fail fast if any SCF child failed.
4. ``run_phonopy`` — collect the per-supercell forces (converted to
   eV/Å) into a ``PhonopyData`` and submit the ``phonopy.phonopy``
   CalcJob that fits the force constants and computes bands / DOS.
5. ``gather_results`` — forward the phonopy outputs (``phonon_bands``,
   ``total_phonon_dos``, ``force_constants``, ``phonopy_data``) and a
   ``output_parameters`` summary.

The FLEUR SCF base (``base`` namespace) is assembled by
:class:`aiida_uranium_workflow.input_builders.phonopy.fleur.FleurPhonopyAdapter`
from ``parameters/fleur/scf.yml`` (its ``wf_parameters`` keep density
mode and gain an ``l_f`` inpxml change so each displaced supercell
returns atomic forces); the phonopy-specific settings come from
``parameters/phonopy.yml``.
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
# point name is registered in both groups).
PhonopyCalculation = CalculationFactory("phonopy.phonopy")
FleurScfWorkChain = WorkflowFactory("fleur.scf")

#: eV per Hartree.
HA_TO_EV = 27.211386245988
#: Å per Bohr.
BOHR_TO_ANG = 0.529177210903
#: eV/Å per Htr/bohr (FLEUR forces in relax.xml are Htr/bohr).
HTR_PER_BOHR_TO_EV_PER_ANG = HA_TO_EV / BOHR_TO_ANG


class FleurPhonopyWorkChain(WorkChain):
    """Compute phonon bands / DOS with FLEUR forces + phonopy post-processing.

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
            help="FLEUR SCF base (fleur/inpgen/wf_parameters/calc_parameters/options).",
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

        When ``primitive_matrix`` is requested the input structure is
        reduced to its minimal primitive cell first and the supercell is
        built from that primitive — keeps the FLEUR SCF cells as small as
        possible (the band path / DOS live in the primitive reciprocal
        space, the phonopy convention).
        """
        kwargs = {"structure": self.inputs.structure}
        if "primitive_matrix" in self.inputs:
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
        """Generate the displaced supercells and submit one FLEUR SCF (with
        forces) per cell."""
        supercells = (
            self.ctx.preprocess_data.calcfunctions.get_supercells_with_displacements()
        )
        self.ctx.supercells = dict(supercells)
        self.ctx.n_supercells = len(supercells)
        self.report(f"generated {len(supercells)} displaced supercell(s)")

        base = dict(self.inputs.base) if "base" in self.inputs else {}
        for label, supercell in supercells.items():
            # FLEUR only writes forces for symmetry-representative atoms;
            # give every atom its own species so each displaced supercell
            # runs in P1 and all 3N forces come back (see module docstring).
            supercell = _break_fleur_symmetry(structure=supercell)
            inputs = {
                "fleur": base["fleur"],
                "structure": supercell,
                "wf_parameters": base["wf_parameters"],
                "calc_parameters": base["calc_parameters"],
            }
            if "inpgen" in base:
                inputs["inpgen"] = base["inpgen"]
            if "options" in base:
                inputs["options"] = base["options"]
            if "options_inpgen" in base:
                inputs["options_inpgen"] = base["options_inpgen"]
            inputs["metadata"] = {
                "label": f"phonon_{label}",
                "description": "Fixed-lattice FLEUR SCF with forces of a displaced supercell",
            }
            running = self.submit(FleurScfWorkChain, **inputs)
            self.report(f"submitted FLEUR SCF pk={running.pk} for {label}")
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
            forces[f"forces_{suffix}"] = _extract_fleur_forces(
                output_parameters=child.outputs.last_calc.output_parameters,
            )

        phonopy_data = self.ctx.preprocess_data.calcfunctions.generate_phonopy_data(
            **forces
        )
        self.ctx.phonopy_data = phonopy_data
        self.report(f"assembled PhonopyData pk={phonopy_data.pk}")

        options = {}
        if "phonopy_options" in self.inputs:
            options = dict(self.inputs.phonopy_options.get_dict())
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
def _break_fleur_symmetry(structure):
    """Return a copy of the structure with a distinct kind per atom (P1).

    FLEUR groups symmetry-equivalent atoms into atom groups and writes
    the forces only for the *representative* atom of each group (out.xml
    ``totalForcesOnRepresentativeAtoms``, see fleur.md §3.6). A
    symmetry-reduced supercell therefore yields fewer force rows than
    atoms and phonopy fails with "Shape mismatch between displacements
    and forces".

    Phonopy needs the forces of every atom of each displaced supercell,
    so the FLEUR symmetry has to be broken. The documented way to make
    all atoms of the same element inequivalent is to give them different
    species via fractional atomic numbers (e.g. ``92.1``, ``92.2``, ...):
    the input generator then creates one atom group per atom and only the
    identity symmetry operation survives (P1). The atom order is
    preserved, so the returned force rows still match the phonopy
    supercell order.
    """
    new = orm.StructureData(cell=structure.cell)
    new.pbc = structure.pbc
    for i, site in enumerate(structure.sites, start=1):
        symbol = structure.get_kind(site.kind_name).symbol
        new.append_atom(
            position=site.position,
            symbols=[symbol],
            name=f"{symbol}{i}",
        )
    return new


@calcfunction
def _to_primitive(structure, primitive_matrix):
    """Reduce a StructureData to its minimal primitive cell (provenance-safe)."""
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


def _parse_force_atoms(last_iter):
    """Adaptively parse one ``force_atoms`` iteration into an (n, 3) array.

    Accepts the common layouts seen from masci-tools / older FLEUR output:
    ``[(atomType, [fx, fy, fz]), ...]``, ``[[fx, fy, fz], ...]``,
    ``[atomType, [fx, fy, fz], atomType, ...]`` (flat alternating) and
    ``[atomType, fx, fy, fz]`` rows. Integer entries (atom-type markers)
    are skipped.
    """
    rows = []
    for entry in last_iter:
        if not isinstance(entry, (list, tuple)):
            continue  # flat atom-type marker
        if len(entry) == 2 and isinstance(entry[1], (list, tuple)) and len(entry[1]) >= 3:
            rows.append(entry[1][:3])  # (atomType, [fx, fy, fz])
        elif len(entry) == 3:
            rows.append(entry)  # [fx, fy, fz]
        elif len(entry) >= 4:
            rows.append(entry[1:4])  # [atomType, fx, fy, fz, ...]
        else:
            raise ValueError(
                f"unexpected force_atoms[-1] entry {entry!r} "
                f"(type {type(entry).__name__}, len {len(entry)})"
            )
    if not rows:
        raise ValueError("force_atoms[-1] contains no force entries")
    return np.asarray(rows, dtype=float)


@calcfunction
def _extract_fleur_forces(output_parameters):
    """Extract the forces (eV/Å) from a FLEUR calculation's ``output_parameters``.

    The FleurCalculation parser runs ``outxml_parser`` over ``out.xml`` and
    stores the result as ``output_parameters``; its ``force_atoms`` entry is
    per SCF iteration the per-atom forces (Htr/bohr — FLEUR's
    ``totalForcesOnRepresentativeAtoms``; with the symmetry broken these are
    all atoms). The last iteration's forces are converted to eV/Å and stored
    as an ArrayData ``forces`` array.

    .. note::
        ``relax.xml``'s ``posforces`` (a previous data source) only tracks
        the *representative* atoms of each symmetry group, so its atom count
        can be smaller than the supercell — phonopy then fails with a
        "Shape mismatch between displacements and forces".
    """
    para = output_parameters.get_dict()
    force_atoms = para.get("force_atoms")
    if not force_atoms:
        raise ValueError(
            "FLEUR output_parameters has no 'force_atoms' — make sure the "
            "SCF runs in 'force' mode (l_f=True, see fleur.md §4.5)."
        )
    forces = _parse_force_atoms(force_atoms[-1])
    forces = forces * HTR_PER_BOHR_TO_EV_PER_ANG  # Htr/bohr → eV/Å

    array = orm.ArrayData()
    array.set_array("forces", forces)
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
    """Build the ``output_parameters`` Dict of the phonon WorkChain."""
    freqs = np.asarray(bands.get_bands())
    phonopy_settings = phonopy_parameters.get_dict()

    result = {
        "workflow": "phonopy",
        "backend": "fleur",
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
