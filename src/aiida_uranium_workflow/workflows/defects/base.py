"""Defect WorkChain base — structure generation + formation energy.

Shared, code-agnostic logic for the defect workflows:

1. ``generate_structures`` — build the host supercell (pymatgen
   ``make_supercell``) and the defect structure (vacancy = removed atom,
   interstitial = inserted atom at a user-given fractional position).
2. ``submit_calculations`` — run the backend calculation (relax — atomic
   positions only — or SCF) for the perfect *and* the defective cell.
3. ``gather_results`` — read the two total energies and compute the
   defect formation energy (no chemical potentials)::

       E_f = E_defect − E_host × (N_defect / N_host)

   the perfect-cell energy is scaled to the defect cell's atom count.

Backend subclasses implement :meth:`_make_calc_inputs` and
:meth:`_read_energy`; everything else lives here.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from typing import Any, Dict

from aiida_uranium_workflow.utils.defects import (
    create_interstitial,
    create_vacancy,
    formation_energy,
    make_supercell,
)

#: wf_parameters key choosing relax vs plain SCF.
MODE_KEY = "mode"

#: Supported defect types.
DEFECT_TYPES = ("vacancy", "interstitial")


@calcfunction
def _make_supercell_cf(structure, supercell_matrix):
    """Calcfunction wrapper around :func:`make_supercell`."""
    return make_supercell(structure, supercell_matrix.get_list())


@calcfunction
def _create_defect_cf(host, defect_dict):
    """Calcfunction wrapper: build the defective structure from a Dict.

    ``defect_dict`` carries:

    * ``type``: ``"vacancy"`` | ``"interstitial"``
    * ``site_index``: int (vacancy) — 0-based site of the removed atom
    * ``element``: str — species removed (vacancy) / inserted (interstitial)
    * ``position``: [x, y, z] fractional (interstitial)
    * ``label``: str — human-readable defect label (stored in the Dict)
    """
    d = defect_dict.get_dict()
    dtype = str(d.get("type", "")).lower()
    if dtype == "vacancy":
        structure, _removed = create_vacancy(host, int(d["site_index"]))
    elif dtype == "interstitial":
        structure = create_interstitial(host, d["element"], list(d["position"]))
    else:
        raise ValueError(
            f"Unknown defect type {dtype!r}; expected one of {DEFECT_TYPES}"
        )
    return structure


@calcfunction
def _compute_formation_energy_cf(
    defect_energy,
    host_energy,
    defect_dict,
    host_natoms,
    defect_natoms,
    mode,
):
    """Calcfunction wrapper around :func:`formation_energy`.

    Builds the **complete** ``output_parameters`` Dict (formation energy +
    defect / mode / atom-count summary) inside the calcfunction so the
    returned node is stored — a WorkChain may not create new ``Data``
    nodes itself and return them as outputs (AiiDA rejects unstored
    outputs in ``update_outputs``).

    Formation energy (no chemical potentials)::

        E_f = E_defect − E_host × (N_defect / N_host)
    """
    out = formation_energy(
        defect_energy.value,
        host_energy.value,
        host_natoms=host_natoms.value,
        defect_natoms=defect_natoms.value,
    )
    out["defect"] = defect_dict.get_dict()
    out["mode"] = mode.value
    return orm.Dict(dict=out)


class DefectsWorkChainBase(WorkChain):
    """Abstract defect-formation-energy WorkChain."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input(
            "supercell_matrix",
            valid_type=orm.List,
            required=False,
            help="Diagonal supercell repetition, e.g. [2, 2, 2]. "
            "Omit to use the input structure as the host cell.",
        )
        spec.input(
            "defect",
            valid_type=orm.Dict,
            required=True,
            help=(
                "Defect definition: {'type': 'vacancy'|'interstitial', "
                "'site_index': int (vacancy), 'element': str, "
                "'position': [x,y,z] fractional (interstitial), 'label': str}."
            ),
        )
        spec.input(
            "wf_parameters",
            valid_type=orm.Dict,
            required=False,
            help=f"Workflow settings: {{'{MODE_KEY}': 'relax'|'scf'}} — "
            "relax optimizes atomic positions only (cell fixed).",
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)
        spec.output("host_structure", valid_type=orm.StructureData, required=True)
        spec.output("defect_structure", valid_type=orm.StructureData, required=True)

        spec.outline(
            cls.generate_structures,
            cls.submit_calculations,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "Defect formation energy computed.")
        spec.exit_code(300, "ERROR_CHILD", "A host/defect calculation failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to read the total energies.")

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def generate_structures(self):
        """Build the host supercell and the defective structure."""
        if "supercell_matrix" in self.inputs:
            host = _make_supercell_cf(
                self.inputs.structure, self.inputs.supercell_matrix
            )
        else:
            host = self.inputs.structure
        defect = _create_defect_cf(host, self.inputs.defect)

        self.ctx.host = host
        self.ctx.defect = defect
        self.ctx.defect_dict = dict(self.inputs.defect.get_dict())
        self.report(
            f"host: {len(host.sites)} sites, defect: {len(defect.sites)} sites "
            f"({self.ctx.defect_dict.get('type')})"
        )

    def submit_calculations(self):
        """Submit the host and defect calculations in parallel."""
        from aiida.engine import ToContext

        wc_cls = self._wc_cls()
        host_inputs = self._make_calc_inputs(self.ctx.host, "host")
        defect_inputs = self._make_calc_inputs(self.ctx.defect, "defect")
        return ToContext(
            host_wc=self.submit(wc_cls, **host_inputs),
            defect_wc=self.submit(wc_cls, **defect_inputs),
        )

    def gather_results(self):
        """Read energies, compute the formation energy, store outputs."""
        host_wc = self.ctx.host_wc
        defect_wc = self.ctx.defect_wc
        if not host_wc.is_finished_ok or not defect_wc.is_finished_ok:
            self.report(
                f"host finished_ok={host_wc.is_finished_ok}, "
                f"defect finished_ok={defect_wc.is_finished_ok}"
            )
            return self.exit_codes.ERROR_CHILD

        try:
            e_host = self._read_energy(host_wc)
            e_defect = self._read_energy(defect_wc)
        except Exception as exc:  # noqa: BLE001 — surface as ERROR_PARSER
            self.report(f"could not read total energies: {exc!r}")
            return self.exit_codes.ERROR_PARSER

        try:
            host_natoms = len(self.ctx.host.sites)
            defect_natoms = len(self.ctx.defect.sites)

            result = _compute_formation_energy_cf(
                defect_energy=orm.Float(e_defect),
                host_energy=orm.Float(e_host),
                defect_dict=self.inputs.defect,
                host_natoms=orm.Int(host_natoms),
                defect_natoms=orm.Int(defect_natoms),
                mode=orm.Str(self._mode()),
            )

            # ``result`` is a calcfunction output — a stored node — so it
            # can be returned as a WorkChain output (creating an unstored
            # ``orm.Dict`` here would make AiiDA's update_outputs reject it).
            self.out("output_parameters", result)
            self.out("host_structure", self.ctx.host)
            self.out("defect_structure", self.ctx.defect)
        except Exception as exc:  # noqa: BLE001 — surface as ERROR_PARSER
            self.report(f"gather_results failed: {exc!r}")
            return self.exit_codes.ERROR_PARSER
        return None  # SUCCESS

    # ------------------------------------------------------------------
    # Backend-specific hooks
    # ------------------------------------------------------------------

    def _wc_cls(self):
        """Return the backend WorkChain class to submit."""
        raise NotImplementedError

    def _make_calc_inputs(self, structure, label: str) -> Dict[str, Any]:
        """Return the backend calculation inputs for ``structure``.

        ``label`` is ``"host"`` or ``"defect"`` (used for metadata).
        """
        raise NotImplementedError

    def _read_energy(self, workchain) -> float:
        """Return the total energy (eV) of a finished child workchain."""
        raise NotImplementedError

    def _mode(self) -> str:
        """Resolve the calculation mode: ``'relax'`` (atomic positions
        only) or ``'scf'``."""
        if "wf_parameters" in self.inputs:
            mode = str(
                self.inputs.wf_parameters.get_dict().get(MODE_KEY, "scf")
            ).lower()
        else:
            mode = "scf"
        return mode if mode in ("relax", "scf") else "scf"
