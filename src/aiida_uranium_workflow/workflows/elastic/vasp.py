"""VASP elastic-constants WorkChain (stress method).

Generates 24 deformed structures (Materials Project methodology:
3 normal strains × ``norm_strains`` + 3 shear strains ×
``shear_strains``), runs one ``vasp.relax`` child per deformed cell
(:class:`VaspRelaxWorkChain` — the plugin relax WorkChain, exactly like
the ABACUS side uses ``abacus.relax``), reads the final ionic step's 3×3
stress tensor (kB) from each child's ``misc`` output, and fits the full
6×6 elastic tensor via pymatgen's
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

The INCAR that goes into the ``vasp.relax`` child has the tags the
plugin derives from ``relax_settings`` (NSW / IBRION / ISIF / EDIFFG)
stripped: aiida-vasp applies the ``incar`` namespace *after* its relax
set-functions, so the SCF preset's ``nsw: 0`` would silently turn the
lattice relaxation into a static run — and a static VASP run omits the
``structure`` output, which made the relax child exit 300 and aborted
the whole workflow before a single strained cell was submitted.

If the lattice relaxation still fails (convergence, wall-time, ...), the
24 strained cells are **computed anyway** from the input structure and
the result is flagged (``equilibrium_structure_source: input`` /
``lattice_relaxed: false`` in ``output_parameters``) — set
``require_lattice_relax=True`` to abort with ``ERROR_RELAX`` instead.

By default the internal atomic coordinates are relaxed under each
strain: each strained cell is a ``vasp.relax`` child with
``relax_settings = {positions: True, shape: False, volume: False, …}``
(ISIF=2, cell held at the strained lattice) taken from the protocol's
``vasp.child_relax`` block — the same vocabulary as
``parameters/relax.yml``. ``force_cutoff`` (= INCAR ``EDIFFG``) is the
knob that decides how much of the internal-coordinate relaxation is
captured: a ±1 % shear of a soft-mode metal (α-U) develops only
~0.03 eV/Å, so a loose criterion leaves the cell clamped and the shear
constants come out ~2× too stiff. Set ``relax_internal=False`` for the
clamped-ion constants (one static SCF per cell, ``perform=False``).

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
    │       — VaspWorkChain inputs, forwarded to both the lattice-relax
    │         and the strained cells (the relax-induced INCAR tags are
    │         stripped from ``parameters``)
    ├── norm_strains    (List) — normal strain magnitudes
    ├── shear_strains   (List) — shear strain magnitudes
    ├── relax_internal  (Bool) — relax internal coords under strain (default True)
    ├── relax_lattice   (Bool) — full lattice relaxation (vasp.relax,
    │                            cell + ions) before the deformations (default True)
    ├── relax_settings  (Dict) — RelaxOptions of the *lattice* relax, e.g.
    │                            {"positions": True, "volume": True, "steps": 60}
    ├── child_relax_settings (Dict) — RelaxOptions of the strained cells,
    │                            e.g. {"positions": True, "shape": False,
    │                            "volume": False, "steps": 100,
    │                            "force_cutoff": 0.005}
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
#: INCAR tags ``VaspRelaxWorkChain`` derives from ``relax_settings``
#: (NSW / IBRION / ISIF / EDIFFG). They are stripped from the SCF preset
#: before it is handed to the ``vasp.relax`` child: aiida-vasp applies the
#: ``incar`` namespace *after* its relax set-functions, so a leftover
#: ``nsw: 0`` (the SCF presets carry it) silently turns the lattice
#: relaxation into a static calculation — and a static VASP run omits the
#: ``structure`` output (``VaspParser._compose_structure`` with
#: ``omit_structure``), so the relax workchain then exits 300
#: (``ERROR_MISSING_REQUIRED_OUTPUT``) and no strained cell is submitted.
#: Same set as ``input_builders.relax.vasp.RELAX_CONTROLLED_INCAR_TAGS``.
_RELAX_CONTROLLED_INCAR_TAGS = ("nsw", "ibrion", "isif", "ediffg")
#: Force criterion (eV/Å) applied when the internal coordinates are
#: relaxed under the strain — same order as the abacus.relax default
#: force_cutoff (0.05 eV/Å) but tighter, since the stress (and hence
#: the fitted tensor) is sensitive to the residual forces.
_FORCE_CUTOFF_EV_ANG = -0.02
#: Fallback RelaxOptions of the strained-cell children (used when the
#: protocol has no ``vasp.child_relax`` block): one positions-only
#: relaxation at the strained lattice with a *tight* force criterion — a
#: ±1 % shear of α-U develops only ~0.03 eV/Å of internal force, so a loose
#: criterion (0.02 eV/Å) stops after one ionic step and the shear constants
#: come out clamped-ion (~2× too stiff).
_DEFAULT_CHILD_RELAX_SETTINGS: Dict[str, Any] = {
    "positions": True,
    "shape": False,
    "volume": False,
    "algo": "cg",
    "steps": 100,
    "force_cutoff": 0.005,
    "convergence_on": False,  # one relax run is enough for a strained cell
    "residual_forces_check": False,
}


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
            "require_lattice_relax",
            valid_type=orm.Bool,
            required=False,
            help=(
                "Abort with ``ERROR_RELAX`` when the ``vasp.relax`` child fails. "
                "Default False: the 24 strained cells are still computed, with "
                "the deformations applied to the *input* structure (the tensor "
                "is then not on the equilibrium lattice — the output records "
                "``equilibrium_structure_source: input``)."
            ),
        )
        spec.input(
            "relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "VaspRelaxWorkChain RelaxOptions dict (e.g. "
                '``{"force_cutoff": 0.03, "steps": 60}``). Defaults to '
                "full cell + ions relaxation with the plugin's defaults."
            ),
        )
        spec.input(
            "child_relax_settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "``VaspRelaxWorkChain`` RelaxOptions of the strained-cell "
                "children (the ``vasp.child_relax`` block of "
                "``parameters/elastic.yml``), e.g. "
                "``{'positions': True, 'shape': False, 'volume': False, "
                "'steps': 100, 'force_cutoff': 0.005}``. The force criterion "
                "decides how much of the internal-coordinate relaxation is "
                "captured — the shear constants of a soft-mode metal (α-U: "
                "C44/C55/C66) differ by ~2x between a clamped and a relaxed "
                "internal coordinate. Defaults to a single positions-only "
                "relaxation (cell fixed, 100 steps, 0.005 eV/Å); "
                "``relax_internal=false`` overrides it with ``perform=False`` "
                "(one static SCF)."
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
            self.inputs.relax_lattice.value if "relax_lattice" in self.inputs else True
        )
        if not relax_lattice:
            self.report("skipping lattice relaxation (relax_lattice=False)")
            return

        base = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        vasp_ns: Dict[str, Any] = {k: v for k, v in base.items() if k != "structure"}
        vasp_ns["parameters"] = self._relax_child_parameters(base.get("parameters"))
        relax_inputs: Dict[str, Any] = {
            "vasp": vasp_ns,
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

    @staticmethod
    def _relax_child_parameters(parameters: Any) -> orm.Dict:
        """SCF parameters for the ``vasp.relax`` child, minus the relax tags.

        The ``incar`` namespace is applied last by aiida-vasp's
        ``ParametersMassage``, so an INCAR tag the plugin derives from
        ``relax_settings`` would silently win: the SCF presets carry
        ``nsw: 0`` (they are meant for single-point runs), which turns the
        "lattice relaxation" into a static calculation — and a static VASP
        run produces no ``structure`` output, so ``VaspRelaxWorkChain``
        exits 300 and the strained cells are never submitted. Drop
        :data:`_RELAX_CONTROLLED_INCAR_TAGS` here and let
        ``relax_settings`` own them.
        """
        if hasattr(parameters, "get_dict"):
            params = parameters.get_dict()
        else:
            params = dict(parameters or {})
        incar = dict(params.get("incar", {}) or {})
        for tag in _RELAX_CONTROLLED_INCAR_TAGS:
            incar.pop(tag, None)
        params["incar"] = incar
        return orm.Dict(dict=params)

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
            self.inputs.relax_lattice.value if "relax_lattice" in self.inputs else True
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
                # Still compute the 24 strained cells: the elastic tensor is
                # a strain response, so an unrelaxed equilibrium lattice
                # yields a usable tensor. It is *not* the equilibrium-lattice
                # result though — flag it in the report and in the outputs.
                self.report(
                    f"{message} — CONTINUING with the input structure (NOT "
                    "relaxed); the reported tensor is not on the equilibrium "
                    "lattice. Set require_lattice_relax=true to abort instead."
                )

        norm = list(self.inputs.norm_strains.get_list())
        shear = list(self.inputs.shear_strains.get_list())

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
        """Submit one ``vasp.relax`` child per deformed structure.

        Each strained cell is a ``VaspRelaxWorkChain`` — the plugin's relax
        WorkChain, exactly like the ABACUS side uses ``abacus.relax`` — so
        the ionic settings are expressed with **RelaxOptions**
        (``positions`` / ``steps`` / ``force_cutoff`` / …) instead of raw
        INCAR tags and aiida-vasp derives the INCAR itself. The cell is held
        at the strained lattice (``shape`` / ``volume`` off), so the relax
        WorkChain performs no final single-point calculation.

        ``relax_internal=false`` (clamped-ion constants) runs the same child
        with ``perform=False`` — a single static SCF.
        """
        base = self.exposed_inputs(ChildWorkChain, agglomerate=True)
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )

        if relax_internal:
            settings = (
                dict(self.inputs.child_relax_settings.get_dict())
                if "child_relax_settings" in self.inputs
                else dict(_DEFAULT_CHILD_RELAX_SETTINGS)
            )
        else:
            # No relaxation: one static SCF at the strained lattice.
            settings = {"perform": False}

        # SCF parameters without the relax-controlled INCAR tags (a preset
        # ``nsw: 0`` would otherwise win over ``relax_settings``).
        vasp_ns = {k: v for k, v in base.items() if k != "structure"}
        vasp_ns["parameters"] = self._relax_child_parameters(base.get("parameters"))

        for idx, structure in enumerate(self.ctx.deformed_structures):
            child_inputs = {
                "structure": structure,
                "vasp": dict(vasp_ns),
                "relax_settings": orm.Dict(dict=settings),
                "metadata": {
                    "label": f"elastic_{idx:03d}_{self.ctx.deformation_labels[idx]}",
                    "description": (
                        "Positions-relaxed strained cell (vasp.relax, cell "
                        "fixed by the strain)"
                        if relax_internal
                        else "Fixed-lattice SCF of a deformed structure"
                    ),
                },
            }
            running = self.submit(RelaxWorkChain, **child_inputs)
            self.report(f"submitted {running.process_label} pk={running.pk} for {idx}")
            self.to_context(**{f"scf_{idx}": running})

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
                        # Which lattice the deformations were applied to:
                        # "lattice_relax" (equilibrium cell from vasp.relax) or
                        # "input" (the input cell — either relax_lattice=False
                        # or the lattice relaxation failed / degraded).
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
    # Carry the equilibrium-lattice provenance into the result so a report
    # can tell an equilibrium-lattice tensor from one fitted on the
    # unrelaxed input cell.
    source = data.get("equilibrium_structure_source", "input")
    result["equilibrium_structure_source"] = source
    result["lattice_relaxed"] = source == "lattice_relax"
    return orm.Dict(dict=result)
