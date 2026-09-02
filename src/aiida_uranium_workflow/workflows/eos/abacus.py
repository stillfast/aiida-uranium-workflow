"""ABACUS equation-of-state (EOS) WorkChain.

Determines the ground-state (equilibrium) volume / lattice constant of a
structure by the standard EOS scan:

1. ``generate_structures`` — scale the input cell uniformly by the
   factors ``guess ± k·step`` (same convention as the aiida-fleur
   ``FleurEosWorkChain``).
2. ``submit_scfs`` — one fixed-lattice ``AbacusBaseWorkChain`` (SCF +
   total energy) per scaled cell.
3. ``inspect_scfs`` — report failed scaled-cell SCFs; the run is not
   aborted, the failed points are simply excluded from the fit.
4. ``collect_fit`` — fit the energy–volume curve (Birch-Murnaghan by
   default, via :mod:`ase.eos`) on the **per-atom** energies / volumes
   of the successful points and expose ``output_parameters``
   (``volume_gs`` / ``bulk_modulus`` / ``bulk_deriv`` / ``energy_gs_ev``
   / per-point table) plus ``optimized_structure`` scaled to the
   equilibrium volume.

The ABACUS SCF base (``base`` namespace) is assembled by
:class:`aiida_uranium_workflow.input_builders.eos.abacus.AbacusEosAdapter`
from ``parameters/abacus/scf.yml``; the EOS scan settings come from
``parameters/eos.yml``.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

import numpy as np

#: eV per Å³ → GPa (ase reports the bulk modulus in eV/Å³).
EV_PER_A3_TO_GPA = 160.217733

# The plugin's own ``abacus.base`` entry point (SCF child).
_PluginBaseWorkChain = WorkflowFactory("abacus.base")

_DEFAULT_EOS = {"points": 9, "step": 0.005, "guess": 1.00}

#: Minimum number of successful scan points needed for a reliable
#: Birch-Murnaghan fit (ase's cubic polynomial / curve_fit needs ≥ 4).
MIN_EOS_POINTS = 4


def scale_structure(structure, scale: float):
    """Return a copy of ``structure`` with every cell vector scaled by ``scale``."""
    cell = np.asarray(structure.cell, dtype=float) * scale
    sd = orm.StructureData(cell=cell, pbc=True)
    for site in structure.sites:
        sd.append_atom(position=site.position, symbols=[site.kind_name])
    return sd


class AbacusEosWorkChain(WorkChain):
    """Compute the equilibrium volume via an ABACUS energy–volume scan.

    Outline: generate_structures → submit_scfs → inspect_scfs →
    collect_fit.

    Exit codes
    ----------
    * 0   ``SUCCESS``      — EOS fit completed.
    * 300 ``ERROR_CHILD``  — fewer than :data:`MIN_EOS_POINTS` scaled-cell
      SCFs succeeded (failed points are otherwise skipped, not fatal).
    * 305 ``ERROR_PARSER`` — energies missing / EOS fit failed.
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
            "eos_parameters",
            valid_type=orm.Dict,
            required=False,
            help="EOS scan settings: points / step / guess.",
        )
        spec.input(
            "fit_type",
            valid_type=orm.Str,
            required=False,
            help=(
                "ase EOS name: 'birchmurnaghan' (default), 'vinet', "
                "'murnaghan', 'sjeos', ..."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)
        spec.output(
            "optimized_structure",
            valid_type=orm.StructureData,
            required=False,
            help="Structure scaled to the equilibrium volume.",
        )

        spec.outline(
            cls.generate_structures,
            cls.submit_scfs,
            cls.inspect_scfs,
            cls.collect_fit,
        )

        spec.exit_code(0, "SUCCESS", "EOS fit completed.")
        spec.exit_code(300, "ERROR_CHILD", "An SCF child failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to fit the equation of state.")

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def generate_structures(self):
        """Build the scaled cells from ``eos_parameters``."""
        eos = (
            dict(self.inputs.eos_parameters.get_dict())
            if "eos_parameters" in self.inputs
            else {}
        )
        points = int(eos.get("points", _DEFAULT_EOS["points"]))
        step = float(eos.get("step", _DEFAULT_EOS["step"]))
        guess = float(eos.get("guess", _DEFAULT_EOS["guess"]))
        start = guess - (points - 1) / 2 * step
        scales = [start + i * step for i in range(points)]

        self.ctx.scales = scales
        self.ctx.scaled_structures = [
            scale_structure(self.inputs.structure, s) for s in scales
        ]
        self.report(
            f"generated {len(scales)} scaled cells, scale factors "
            f"{[round(s, 4) for s in scales]}"
        )

    def submit_scfs(self):
        """Submit one fixed-lattice SCF per scaled cell."""
        base = dict(self.inputs.base) if "base" in self.inputs else {}
        for idx, structure in enumerate(self.ctx.scaled_structures):
            child_inputs = dict(base)
            child_inputs["abacus"]["structure"] = structure
            child_inputs["metadata"] = {
                "label": f"eos_scale_{self.ctx.scales[idx]:.4f}".replace(".", "_"),
                "description": "Fixed-lattice SCF of a scaled cell",
            }
            running = self.submit(_PluginBaseWorkChain, **child_inputs)
            self.report(
                f"submitted SCF pk={running.pk} for scale {self.ctx.scales[idx]:.4f}"
            )
            self.to_context(**{f"scf_{idx}": running})

    def inspect_scfs(self):
        """Report scaled-cell SCF failures without aborting.

        The EOS fit in :meth:`collect_fit` keeps the successful points
        and skips the failed ones (same behaviour as the aiida-fleur
        ``FleurEosWorkChain``); a run with too few survivors is caught
        there.
        """
        n_failed = 0
        for idx in range(len(self.ctx.scaled_structures)):
            child = getattr(self.ctx, f"scf_{idx}", None)
            if child is None or not child.is_finished_ok:
                n_failed += 1
                self.report(
                    f"SCF child {idx} pk={getattr(child, 'pk', None)} failed "
                    f"(state={getattr(child, 'process_state', None)}, "
                    f"exit={getattr(child, 'exit_status', None)}); "
                    f"its EOS point will be skipped"
                )
        if n_failed:
            self.report(
                f"{n_failed}/{len(self.ctx.scaled_structures)} scaled-cell SCFs "
                f"failed; the EOS fit will use the remaining points"
            )
        else:
            self.report("all scaled-cell SCFs finished OK")
        return None

    def collect_fit(self):
        """Collect energies / volumes of the successful scaled-cell SCFs,
        fit the EOS, and store the outputs.

        Failed SCF points are **skipped** — the fit uses the remaining
        points (like the aiida-fleur ``FleurEosWorkChain``). With fewer
        than :data:`MIN_EOS_POINTS` survivors the fit is not possible
        and the workchain exits with ``ERROR_CHILD``.
        """
        energies: list[float] = []
        volumes: list[float] = []
        scales: list[float] = []
        for idx in range(len(self.ctx.scaled_structures)):
            child = getattr(self.ctx, f"scf_{idx}", None)
            if child is None or not child.is_finished_ok:
                self.report(f"SCF child {idx} failed; skipping its EOS point")
                continue
            try:
                misc = child.outputs.misc.get_dict()
                e = misc.get("total_energy")
            except (AttributeError, KeyError):
                e = None
            if e is None:
                self.report(
                    f"SCF child {idx} has no total_energy; skipping its EOS point"
                )
                continue
            energies.append(float(e))
            # The child's input structure is the scaled cell (fixed-lattice).
            # ``AbacusBaseWorkChain`` exposes the AbacusCalculation inputs
            # under the ``abacus`` namespace, so the structure lives at
            # ``inputs.abacus.structure`` (a bare ``inputs.structure`` would
            # raise AttributeError and crash the workchain in collect_fit).
            volumes.append(float(child.inputs.abacus.structure.get_cell_volume()))
            scales.append(self.ctx.scales[idx])

        if len(energies) < MIN_EOS_POINTS:
            self.report(
                f"only {len(energies)}/{len(self.ctx.scaled_structures)} SCF points "
                f"succeeded; at least {MIN_EOS_POINTS} are required for the EOS fit"
            )
            return self.exit_codes.ERROR_CHILD

        eos_para = _fit_eos(
            energies=orm.List(list=energies),
            volumes=orm.List(list=volumes),
            scales=orm.List(list=scales),
            structure=self.inputs.structure,
            fit_type=self.inputs.fit_type if "fit_type" in self.inputs else None,
        )
        self.out("output_parameters", eos_para)

        data = eos_para.get_dict()
        if data.get("volume_gs") is not None:
            self.out(
                "optimized_structure",
                _equilibrium_structure(
                    structure=self.inputs.structure, eos_para=eos_para
                ),
            )
        return None  # SUCCESS


# ---------------------------------------------------------------------------
# calcfunctions
# ---------------------------------------------------------------------------


@calcfunction
def _fit_eos(energies, volumes, scales, structure, fit_type=None):
    """Fit the energy–volume curve and return the EOS summary Dict.

    The Birch-Murnaghan fit is performed on **per-atom** energy and
    volume (E/N vs V/N) — the standard DFT convention, identical to the
    aiida-fleur ``FleurEosWorkChain``. The reported ``volume_gs`` /
    ``energy_gs_ev`` are cell totals (per-atom values × natoms) so both
    backends stay consistent; per-atom values are stored as well.
    ``fit_type`` may be ``None`` (defaults to Birch-Murnaghan). The bulk
    modulus is reported in GPa (ase returns eV/Å³; B is intensive, so
    the per-atom fit gives the same value as a total-volume fit).
    """
    from ase.eos import EquationOfState

    e_ev = [float(x) for x in energies.get_list()]
    v = [float(x) for x in volumes.get_list()]
    scale_list = [float(x) for x in scales.get_list()]
    eos_name = fit_type.value if fit_type is not None else "birchmurnaghan"

    natoms = len(structure.sites)
    v_per_atom = [x / natoms for x in v]
    e_per_atom = [x / natoms for x in e_ev]

    result: dict = {
        "workflow": "eos",
        "backend": "abacus",
        "fit": eos_name,
        "natoms": natoms,
        "scales": scale_list,
        "volumes": v,
        "volumes_per_atom": v_per_atom,
        "energies_ev": e_ev,
        "energies_per_atom_ev": e_per_atom,
        "n_points": len(v),
    }

    try:
        eos = EquationOfState(v_per_atom, e_per_atom, eos=eos_name)
        v0_per_atom, e0_per_atom, bulk = eos.fit()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return orm.Dict(dict=result)

    if v0_per_atom is None or np.isnan(v0_per_atom):
        result["error"] = "EOS fit returned no ground-state volume"
        return orm.Dict(dict=result)

    v0_per_atom = float(v0_per_atom)
    volume_gs = v0_per_atom * natoms
    energy_gs = float(e0_per_atom) * natoms
    # ase 3.26: fit() returns (v0, e0, B); the pressure derivative B'
    # lives in the fitted parameters (index 2 of the BM parametrisation).
    bulk_deriv = None
    try:
        bulk_deriv = float(eos.eos_parameters[2])
    except (AttributeError, IndexError, TypeError):
        pass

    result.update(
        {
            "volume_gs": volume_gs,
            "volume_units": "A^3",
            "volume_gs_per_atom": v0_per_atom,
            "volume_gs_per_atom_units": "A^3/atom",
            "energy_gs_ev": energy_gs,
            "energy_gs_per_atom_ev": float(e0_per_atom),
            "bulk_modulus": float(bulk) * EV_PER_A3_TO_GPA,
            "bulk_modulus_units": "GPa",
            "bulk_deriv": bulk_deriv,
            "residuals": None,
            "scaling_gs": (volume_gs / float(structure.get_cell_volume())) ** (1.0 / 3.0),
        }
    )

    # Lattice constant only for cubic cells (a = V^(1/3)).
    cell = np.asarray(structure.cell, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    angles = np.array(
        [
            np.degrees(
                np.arccos(
                    np.clip(np.dot(cell[i], cell[j]) / (lengths[i] * lengths[j]), -1, 1)
                )
            )
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
    )
    if np.allclose(lengths, lengths[0], atol=1e-3) and np.allclose(
        angles, 90.0, atol=1e-3
    ):
        result["lattice_constant_gs"] = volume_gs ** (1.0 / 3.0)
        result["lattice_constant_units"] = "A"

    return orm.Dict(dict=result)


@calcfunction
def _equilibrium_structure(structure, eos_para):
    """Return the input structure scaled to the equilibrium volume."""
    v0 = eos_para.get_dict().get("volume_gs")
    if v0 is None:
        raise ValueError("no equilibrium volume in the EOS result")
    scale = (float(v0) / float(structure.get_cell_volume())) ** (1.0 / 3.0)
    return scale_structure(structure, scale)
