"""ABACUS cubic elastic-constant WorkChain (energy–strain method).

Determines the three cubic elastic constants (C11, C12, C44) of a cubic
structure from **total energies of distorted cells**, following the
textbook scheme of "Computational Quantum Mechanics for Materials
Engineers" (Section 6.2, Cubic Lattice):

1. ``prepare`` — validate that only the ``cubic`` crystal system is
   requested and that the input cell actually looks cubic.
2. ``run_eos`` — when the equilibrium bulk modulus ``B``, volume ``V0``
   and energy ``E0`` are **not** all given, run the ABACUS EOS workflow
   (``uranium.eos.abacus``) on 7 scan points to obtain them.
3. ``run_distortions`` — for each δ = 0.00, 0.01, …, 0.05 compute the
   fixed-lattice SCF total energy of the volume-conserving distortions

   * orthorhombic  D_o (6.38):  diag(1+δ, 1−δ, 1/(1−δ²))  →
     ΔE = 2·V·c′·δ²,   c′ = (C11−C12)/2
   * monoclinic   D_m (6.40):  [[1,δ],[δ,1]] ⊕ 1/(1−δ²) →
     ΔE = 2·V·C44·δ²

   (only positive δ — both distortions are symmetric in δ).

   The asymmetric tetragonal distortion D_t (6.36),
   ``diag(1+δ, 1+δ, 1/(1+δ)²)``, is deliberately **not** used: unlike
   D_o/D_m it is not even in δ (δ → −δ gives the elongated instead of
   the compressed tetragonal cell, which no cubic symmetry operation
   maps onto each other), so its energy carries an odd-power term,
   ΔE = 6·V·c′·δ² + O(δ³) (6.37), and would require *both* ±δ plus an
   a₃δ³ term.  D_o/D_m are even in δ, so their first correction is
   O(δ⁴) (6.39)/(6.41) and half the SCF points suffice.
4. ``gather`` — fit ΔE(δ) = a₂δ² + a₄δ⁴ on the even powers, read the
   δ² coefficient, and with

   B  = (C11 + 2·C12)/3      (6.34)
   c′ = (C11 − C12)/2        (6.35)

   solve  C11 = B + (4/3)c′,  C12 = B − (2/3)c′;  C44 = a₂/(2V0).

The ABACUS base (``base`` namespace) is assembled by the adapter from
``parameters/abacus/scf.yml``.  With ``relax_internal = True`` (default)
the distorted cells are relaxed through ``abacus.relax`` with the lattice
frozen (``DEFAULT_RELAX_SETTINGS``), i.e. only the internal coordinates
are allowed to respond — the relaxed-ion elastic constants.  With
``relax_internal = False`` the plain ``abacus.base`` SCF is run and the
fractional coordinates stay clamped (clamped-ion constants).

Exit codes
----------
* 0   ``SUCCESS``        — elastic constants computed.
* 300 ``ERROR_CHILD``    — an EOS or distorted-cell SCF child failed.
* 305 ``ERROR_PARSER``   — energies / equilibrium data missing.
* 400 ``ERROR_UNSUPPORTED_CRYSTAL_SYSTEM`` — crystal system ≠ cubic.
"""

from __future__ import annotations

from typing import Any, Dict

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

import numpy as np

#: eV per Å³ → GPa.
EV_PER_A3_TO_GPA = 160.217733

#: The six textbook distortions: δ = 0.00, 0.01, …, 0.05.
DEFAULT_DELTAS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

#: Minimum number of converged distortion points needed per distortion
#: for the E₀ + a₂δ² + a₄δ⁴ least-squares fit (3 parameters).
MIN_DISTORTION_POINTS = 4

_PluginBaseWorkChain = WorkflowFactory("abacus.base")
_PluginRelaxWorkChain = WorkflowFactory("abacus.relax")
_EosWorkChain = WorkflowFactory("uranium.eos.abacus")

#: Default abacus.relax settings for the distorted cells: relax atomic
#: positions only (the cell is fixed by the strain) with the same force
#: criterion as the relax protocol (0.05 eV/Å).
DEFAULT_RELAX_SETTINGS = {
    "relax_type": "positions",  # RelaxType.POSITIONS: atoms only, cell fixed
    "perform": True,
    "relax_method": "cg",
    "max_ionic_steps": 50,
    "force_cutoff": 0.05,  # eV/Å (force_thr_ev)
    "stress_cutoff": 1.0,  # kBar (stress_thr; unused for positions relax)
}


def _orthorhombic_matrix(delta: float) -> np.ndarray:
    """D_o (6.38): volume-conserving orthorhombic distortion."""
    return np.diag([1.0 + delta, 1.0 - delta, 1.0 / (1.0 - delta**2)])


def _monoclinic_matrix(delta: float) -> np.ndarray:
    """D_m (6.40): volume-conserving monoclinic distortion."""
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = delta
    matrix[2, 2] = 1.0 / (1.0 - delta**2)
    return matrix


def distort_structure(structure, matrix: np.ndarray):
    """Return ``structure`` deformed by the affine matrix ``F = I + D``.

    The cell vectors (rows) are right-multiplied by the (symmetric)
    distortion and the fractional coordinates are kept — the standard
    elastic energy method, which needs no internal relaxation.
    """
    from ase import Atoms

    atoms = structure.get_ase()
    new_cell = np.dot(np.asarray(atoms.cell, dtype=float), matrix)
    deformed = Atoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=new_cell,
        pbc=True,
    )
    return orm.StructureData(ase=deformed)


def scale_to_volume(structure, target_volume: float):
    """Isotropically rescale ``structure`` to ``target_volume`` (Å³).

    Only the lattice is scaled; the **fractional** coordinates of every
    atom are kept unchanged (their absolute Cartesian positions scale
    with the cell — no internal relaxation). Returns ``(structure,
    factor)`` where ``factor`` is the applied uniform scale.
    """
    from ase import Atoms

    atoms = structure.get_ase()
    current = float(atoms.get_volume())
    if current <= 0.0:
        raise ValueError(f"structure has non-positive cell volume {current}")
    factor = (float(target_volume) / current) ** (1.0 / 3.0)
    if abs(factor - 1.0) < 1.0e-6:
        return structure, 1.0
    scaled = Atoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=np.asarray(atoms.cell, dtype=float) * factor,
        pbc=True,
    )
    return orm.StructureData(ase=scaled), float(factor)


def is_cubic(structure, tolerance: float = 1.0e-2) -> bool:
    """Rough cubic check: equal cell lengths (within ``tolerance`` in Å)
    and 90° angles (within 0.5°)."""
    cell = np.asarray(structure.cell, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if np.ptp(lengths) > tolerance:
        return False
    angles = np.degrees(
        np.arccos(
            np.clip(
                np.einsum("ij,ij->i", cell, np.roll(cell, -1, axis=0))
                / (lengths * np.roll(lengths, -1)),
                -1.0,
                1.0,
            )
        )
    )
    return bool(np.all(np.abs(angles - 90.0) < 0.5))


class AbacusCubicElasticWorkChain(WorkChain):
    """Cubic elastic constants (C11/C12/C44) from distorted-cell energies.

    Outline: prepare → run_eos → run_distortions → gather.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input_namespace(
            "base",
            dynamic=True,
            help="AbacusBaseWorkChain inputs (SCF base from scf.yml).",
        )
        spec.input(
            "crystal_system",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str("cubic"),
            help="Lattice family; only 'cubic' is currently supported.",
        )
        # ---- equilibrium input (optional; else 7-point EOS fallback) ----
        spec.input(
            "bulk_modulus",
            valid_type=orm.Float,
            required=False,
            help="Equilibrium bulk modulus B (GPa).",
        )
        spec.input(
            "volume_gs",
            valid_type=orm.Float,
            required=False,
            help="Equilibrium cell volume V0 (Å³).",
        )
        spec.input(
            "energy_gs_ev",
            valid_type=orm.Float,
            required=False,
            help="Equilibrium total energy E0 (eV).",
        )
        spec.input(
            "eos_parameters",
            valid_type=orm.Dict,
            required=False,
            help="EOS scan settings (points=7 / step / guess).",
        )
        spec.input(
            "deltas",
            valid_type=orm.List,
            required=False,
            help="Distortion amplitudes (default 0.00 … 0.05 step 0.01).",
        )
        spec.input(
            "relax_internal",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help=(
                "Relax the internal atomic coordinates under each "
                "distortion (abacus.relax, relax_type='positions', cell "
                "fixed by the strain) so the fitted constants are the "
                "physically relevant 'relaxed-ion' ones. Cubic lattices "
                "with a multi-atom basis (e.g. diamond Si) have free "
                "internal coordinates and MUST relax — the affine "
                "(clamped-ion) D_m energy then overestimates C44. For "
                "cells without free internal coordinates (e.g. the "
                "1-atom bcc primitive) it is a no-op. Set to False for "
                "the clamped-ion / textbook affine method."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.prepare,
            cls.run_eos,
            cls.run_distortions,
            cls.gather,
        )

        spec.exit_code(0, "SUCCESS", "Cubic elastic constants computed.")
        spec.exit_code(300, "ERROR_CHILD", "An EOS / distorted-cell SCF child failed.")
        spec.exit_code(305, "ERROR_PARSER", "Energies or equilibrium data missing.")
        spec.exit_code(
            400,
            "ERROR_UNSUPPORTED_CRYSTAL_SYSTEM",
            "Only the cubic crystal system is supported.",
        )

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def prepare(self):
        """Validate the cubic-only requirement and the input cell."""
        system = (
            self.inputs.crystal_system.value
            if isinstance(self.inputs.crystal_system, orm.Str)
            else str(self.inputs.crystal_system)
        )
        if system != "cubic":
            self.report(
                f"crystal_system={system!r} — only 'cubic' is supported"
            )
            return self.exit_codes.ERROR_UNSUPPORTED_CRYSTAL_SYSTEM
        if not is_cubic(self.inputs.structure):
            self.report(
                "input cell does not look cubic (unequal lengths or "
                "non-90° angles)"
            )
            return self.exit_codes.ERROR_UNSUPPORTED_CRYSTAL_SYSTEM

        self.ctx.deltas = (
            [float(d) for d in self.inputs.deltas.get_list()]
            if "deltas" in self.inputs
            else list(DEFAULT_DELTAS)
        )
        self.report(
            f"cubic structure ok; distortions δ = "
            f"{[round(d, 3) for d in self.ctx.deltas]}"
        )
        return None

    def run_eos(self):
        """Resolve B / V0 / E0 — from the inputs, or a 7-point EOS run."""
        have = {
            key: getattr(self.inputs, key, None) is not None
            for key in ("bulk_modulus", "volume_gs", "energy_gs_ev")
        }
        if all(have.values()):
            self.ctx.equilibrium = {
                "bulk_modulus": float(self.inputs.bulk_modulus.value),
                "volume_gs": float(self.inputs.volume_gs.value),
                "energy_gs_ev": float(self.inputs.energy_gs_ev.value),
            }
            self.ctx.eos_used = False
            self.ctx.work_structure, self.ctx.volume_scale = scale_to_volume(
                self.inputs.structure, self.ctx.equilibrium["volume_gs"]
            )
            self.report(
                f"using supplied B / V0 / E0 (no EOS run); "
                f"input cell rescaled to V0={self.ctx.equilibrium['volume_gs']} Å³ "
                f"(isotropic factor "
                f"{self.ctx.volume_scale:.6f}, fractional coords unchanged)"
            )
            return None

        missing = [k for k, v in have.items() if not v]
        self.report(
            f"equilibrium {missing} not supplied — running the 7-point EOS "
            f"workflow"
        )
        eos_parameters = dict(
            self.inputs.eos_parameters.get_dict()
            if "eos_parameters" in self.inputs
            else {}
        )
        # The fallback needs a wide-enough volume window: with a narrow
        # one (±1.5%, i.e. the EOS default 0.5% volume step) the energy
        # differences are comparable to the SCF noise and the
        # three-parameter Birch-Murnaghan fit becomes ill-conditioned
        # (e.g. B = 209 GPa with a non-physical B' = 0.95 for bcc-U).
        # Default: 9 points, 1% relative volume steps → ±4% volume.
        eos_parameters.setdefault("points", 9)
        eos_parameters.setdefault("step", 0.01)
        eos_parameters.setdefault("guess", 1.00)
        child_inputs = {
            "structure": self.inputs.structure,
            "base": dict(self.inputs.base) if "base" in self.inputs else {},
            "eos_parameters": orm.Dict(dict=eos_parameters),
            "metadata": {
                "label": "elastic_cubic_eos",
                "description": "7-point EOS fallback for B / V0 / E0",
            },
        }
        running = self.submit(_EosWorkChain, **child_inputs)
        self.report(f"submitted EOS workflow pk={running.pk}")
        self.to_context(eos=running)

    def run_distortions(self):
        """Submit one fixed-lattice SCF per distorted cell (D_o and D_m)."""
        if hasattr(self.ctx, "eos"):
            eos = self.ctx.eos
            if eos is None or not eos.is_finished_ok:
                return self.exit_codes.ERROR_CHILD
            para = eos.outputs.output_parameters.get_dict()
            self.ctx.equilibrium = {
                "bulk_modulus": para.get("bulk_modulus"),
                "volume_gs": para.get("volume_gs"),
                "energy_gs_ev": para.get("energy_gs_ev"),
            }
            self.ctx.eos_used = True
            base_structure = (
                eos.outputs.optimized_structure
                if "optimized_structure" in {
                    l.link_label for l in eos.base.links.get_outgoing().all()
                }
                else self.inputs.structure
            )
            # Pin the cell to the fitted equilibrium volume V0 (isotropic
            # scaling, fractional coordinates unchanged) so the
            # distortions are centred exactly on the equilibrium.
            self.ctx.work_structure, self.ctx.volume_scale = scale_to_volume(
                base_structure, float(self.ctx.equilibrium["volume_gs"])
            )
            self.report(
                f"EOS: B={self.ctx.equilibrium['bulk_modulus']} GPa, "
                f"V0={self.ctx.equilibrium['volume_gs']} Å³, "
                f"E0={self.ctx.equilibrium['energy_gs_ev']} eV "
                f"(cell pinned to V0, factor {self.ctx.volume_scale:.6f})"
            )

        equilibrium = getattr(self.ctx, "equilibrium", None)
        if (
            equilibrium is None
            or equilibrium["bulk_modulus"] is None
            or equilibrium["volume_gs"] is None
        ):
            return self.exit_codes.ERROR_PARSER

        base = dict(self.inputs.base) if "base" in self.inputs else {}
        structure = getattr(self.ctx, "work_structure", self.inputs.structure)
        relax_internal = (
            self.inputs.relax_internal.value
            if "relax_internal" in self.inputs
            else True
        )
        self.ctx.energies: dict = {"o": {}, "m": {}}
        submissions = []
        for kind, matrix_fn in (("o", _orthorhombic_matrix), ("m", _monoclinic_matrix)):
            for delta in self.ctx.deltas:
                if delta == 0.0:
                    deformed = structure  # δ=0: undistorted cell
                else:
                    deformed = distort_structure(structure, matrix_fn(delta))
                label = f"elastic_{kind}_{delta:.2f}".replace(".", "_")
                if relax_internal:
                    child_inputs = self._relax_inputs_for(deformed, base)
                    child_cls = _PluginRelaxWorkChain
                else:
                    child_inputs = self._scf_inputs_for(deformed, base)
                    child_cls = _PluginBaseWorkChain
                child_inputs["metadata"] = {
                    "label": label,
                    "description": (
                        f"Position-relaxed SCF under {kind} distortion "
                        f"δ={delta:.2f}"
                        if relax_internal
                        else f"Fixed-lattice SCF under {kind} distortion δ={delta:.2f}"
                    ),
                }
                running = self.submit(child_cls, **child_inputs)
                submissions.append((kind, delta, running))
                self.report(f"submitted {label} pk={running.pk}")
        for kind, delta, running in submissions:
            self.to_context(**{f"scf_{kind}_{delta:.2f}".replace(".", "_"): running})

    def _scf_inputs_for(self, structure, base):
        """A fixed-lattice ``abacus.base`` SCF of a distorted cell."""
        child_inputs = {k: v for k, v in base.items() if k != "abacus"}
        abacus = dict(base["abacus"])
        abacus["structure"] = structure
        child_inputs["abacus"] = abacus
        return child_inputs

    def _relax_inputs_for(self, structure, base):
        """An ``abacus.relax`` positions-only relax of a distorted cell.

        Mirrors the elastic (stress) workflow: ``structure`` at the top
        level, the SCF base in the ``base`` namespace, ``calculation =
        relax``, the cell fixed by the strain and ``meta_convergence``
        off (the volume comparison is meaningless for a fixed cell).
        """
        base_abacus = base.get("abacus", {}) if base else {}
        params = (
            dict(base_abacus["parameters"].get_dict())
            if "parameters" in base_abacus
            else {}
        )
        params.setdefault("input", {})
        params["input"]["calculation"] = "relax"

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
            "relax_settings": orm.Dict(dict=dict(DEFAULT_RELAX_SETTINGS)),
            "meta_convergence": orm.Bool(False),
            "clean_workdir": orm.Bool(False),
        }

    def gather(self):
        """Fit the δ² coefficients and solve for C11 / C12 / C44.

        Failed distorted-cell runs are **skipped** (like the EOS
        workflow) — the fit uses the surviving δ points of each
        distortion; fewer than :data:`MIN_DISTORTION_POINTS` survivors
        for either D_o or D_m abort the run.
        """
        kept: dict = {"o": {"deltas": [], "energies": []}, "m": {"deltas": [], "energies": []}}
        for kind in ("o", "m"):
            for delta in self.ctx.deltas:
                key = f"scf_{kind}_{delta:.2f}".replace(".", "_")
                child = getattr(self.ctx, key, None)
                if child is None or not child.is_finished_ok:
                    self.report(
                        f"distortion {kind} δ={delta:.2f} failed — skipping "
                        f"its point"
                    )
                    continue
                try:
                    total = child.outputs.misc.get_dict().get("total_energy")
                except (AttributeError, KeyError):
                    total = None
                if total is None:
                    self.report(
                        f"distortion {kind} δ={delta:.2f} has no total_energy "
                        f"— skipping its point"
                    )
                    continue
                kept[kind]["deltas"].append(float(delta))
                kept[kind]["energies"].append(float(total))

        if any(len(kept[kind]["deltas"]) < MIN_DISTORTION_POINTS for kind in ("o", "m")):
            self.report(
                f"too few converged distortion points: "
                f"D_o {len(kept['o']['deltas'])}, D_m {len(kept['m']['deltas'])} "
                f"(need ≥ {MIN_DISTORTION_POINTS} each)"
            )
            return self.exit_codes.ERROR_CHILD

        eq = self.ctx.equilibrium
        result = _solve_cubic(
            deltas_o=orm.List(list=kept["o"]["deltas"]),
            energies_o=orm.List(list=kept["o"]["energies"]),
            deltas_m=orm.List(list=kept["m"]["deltas"]),
            energies_m=orm.List(list=kept["m"]["energies"]),
            bulk_modulus=orm.Float(float(eq["bulk_modulus"])),
            volume_gs=orm.Float(float(eq["volume_gs"])),
        )
        self.out("output_parameters", result)
        return None  # SUCCESS


# ---------------------------------------------------------------------------
# calcfunctions
# ---------------------------------------------------------------------------


@calcfunction
def _solve_cubic(
    deltas_o,
    energies_o,
    deltas_m,
    energies_m,
    bulk_modulus,
    volume_gs,
):
    """Fit ΔE(δ) = a₂δ² + a₄δ⁴ for D_o / D_m and solve the constants.

    * c′  = a₂⁽ᵒ⁾ / (2·V0)          (D_o, 6.39)
    * C44 = a₂⁽ᵐ⁾ / (2·V0)          (D_m, 6.41)
    * C11 = B + (4/3)·c′, C12 = B − (2/3)·c′   (6.34 + 6.35)

    The fit uses the even polynomial E(δ) = E₀ + a₂δ² + a₄δ⁴ (the
    textbook prescription (6.31)–(6.32) only requires *taking the δ²
    coefficient* of a polynomial fit; D_o / D_m are even in δ so no odd
    powers appear). The δ⁴ term is necessary — over δ ≤ 0.05 a pure
    a₂δ² fit is biased low (e.g. C44 by ~4 GPa for Si) — while δ⁶ and
    higher are deliberately **not** fitted: with six points that would
    leave only two degrees of freedom and start absorbing SCF noise.
    The free intercept makes the δ = 0 point non-special, so failed
    distortion points can simply be dropped.

    The returned Dict is deliberately minimal: the equilibrium data
    (B / V0 / E0) and the flags (crystal system, relax_internal,
    eos_used) live on the WorkChain **inputs**, c′ and the 6×6 tensor are
    derived from the three constants — only the constants and the raw
    fit data are stored here.
    """
    volume = float(volume_gs.value)
    bulk_gpa = float(bulk_modulus.value)

    def fit_a2(deltas_raw, energies_raw):
        """E(δ) = E₀ + a₂δ² + a₄δ⁴ least squares → (a₂, data)."""
        pairs = sorted(
            (float(d), float(e))
            for d, e in zip(deltas_raw.get_list(), energies_raw.get_list())
        )
        x = np.asarray([p[0] for p in pairs])
        y = np.asarray([p[1] for p in pairs])
        design = np.vstack([np.ones_like(x), x**2, x**4]).T
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        a2 = float(coefficients[1])
        rms = float(np.sqrt(np.mean((design @ coefficients - y) ** 2)))
        return a2, x.tolist(), y.tolist(), rms

    a2_o, deltas_o_list, energies_o_list, rms_o = fit_a2(deltas_o, energies_o)
    a2_m, deltas_m_list, energies_m_list, rms_m = fit_a2(deltas_m, energies_m)
    # eV/Å³ → GPa
    c_prime = EV_PER_A3_TO_GPA * a2_o / (2.0 * volume)
    c44 = EV_PER_A3_TO_GPA * a2_m / (2.0 * volume)
    c11 = bulk_gpa + (4.0 / 3.0) * c_prime
    c12 = bulk_gpa - (2.0 / 3.0) * c_prime

    return orm.Dict(
        dict={
            "workflow": "elastic_cubic",
            "backend": "abacus",
            "method": "energy",
            "constants_gpa": {"c11": c11, "c12": c12, "c44": c44},
            "fits": {
                "orthorhombic": {
                    "a2_ev": a2_o,
                    "deltas": deltas_o_list,
                    "energies_ev": energies_o_list,
                    "n_points": len(energies_o_list),
                    "rms_ev": rms_o,
                },
                "monoclinic": {
                    "a2_ev": a2_m,
                    "deltas": deltas_m_list,
                    "energies_ev": energies_m_list,
                    "n_points": len(energies_m_list),
                    "rms_ev": rms_m,
                },
            },
        }
    )
