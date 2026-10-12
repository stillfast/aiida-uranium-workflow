"""ABACUS supercell SCF WorkChain.

Runs one fixed-lattice calculation per supercell matrix from the protocol
(``parameters/supercell.yml``), each with its own SCF parameters
(k-point mesh, scf_thr, mixing settings, …) applied on top of the
shared ``parameters/abacus/scf.yml`` preset.

The child mode follows the preset's ``calculation``: ``scf`` by default,
or e.g. ``nscf`` to diagonalise once on a charge density supplied via
``init_chg file`` (+ ``read_file_dir``).

Outline: generate_supercells → submit_scfs → inspect_scfs →
collect_results.

Exit codes
----------
* 0   ``SUCCESS``      — all supercell SCFs finished OK.
* 300 ``ERROR_CHILD``  — a supercell SCF child failed.
* 305 ``ERROR_PARSER`` — could not parse a total energy.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

# The plugin's own ``abacus.base`` entry point (SCF child).
_PluginBaseWorkChain = WorkflowFactory("abacus.base")

#: ABACUS ``input`` keys a supercell entry may override on top of the
#: SCF-base preset (k-points are handled separately).
_SCF_OVERRIDE_KEYS = (
    "scf_thr",
    "scf_nmax",
    "ecutwfc",
    "mixing_beta",
    "mixing_beta_mag",
    "mixing_angle",
    "mixing_ndim",
    "mixing_type",
    # Kerker preconditioning of the charge-density residual (ABACUS
    # default 1.0; ``0`` switches Kerker off, which the convergence
    # manual recommends for localized d/f systems such as actinides).
    # gg0 = (mixing_gg0 × 0.529177 / tpiba)², so the same value damps
    # harder in a larger cell — only compare mixing_gg0 across cells of
    # equal size. ABACUS silently skips Kerker when mixing_beta <= 0.1
    # (charge_mixing_preconditioner.cpp), so a gg0 scan needs
    # mixing_beta > 0.1 (lcao_1q: 0.5). Per-cell overridable so the
    # Kerker strength can be scanned (test_mixing_gg0).
    "mixing_gg0",
    "mixing_gg0_min",

    # Mixing-restart ladder (ABACUS >= v3.6.0; the ``abacus_lts`` driver
    # is v3.10.0): once drho drops to ``mixing_restart`` the SCF clears
    # its mixing history and continues from the current density.
    # ``mixing_dmr`` additionally mixes the real-space density matrix —
    # it only feeds back into the potential for DFT+U / EXX / DeePKS, so
    # in a plain SCF it costs the DM-mixing buffers without changing the
    # physics. Per-cell overridable so the restart threshold can be
    # scanned (test_mixing_restart).
    "mixing_restart",
    "mixing_dmr",

    # Oscillation guard (ABACUS >= v3.5.0; present in the v3.10.0 driver).
    # ``scf_os_stop 1`` aborts the SCF once the least-squares slope of
    # log10(drho) over the last ``scf_os_ndim`` iterations exceeds
    # ``scf_os_thr`` (default -0.01, i.e. drho no longer falling by >2.3%
    # per step) — see Charge_Mixing::if_scf_oscillate. It only fires on
    # runs that are already failing, so it is pure wallclock savings:
    # the 2662-atom run 384005 burned 38 steps x 2250 s and hit the
    # 24 h wallclock without ever converging; this would have stopped it
    # around ELEC 6. Default off (ABACUS scf_os_stop = false).
    # Verified accepted by abacus_lts@yeesuan (v3.10.0) 2026-09-21:
    # a 2-atom smoke run echoed scf_os_stop/scf_os_thr/scf_os_ndim=24 in
    # OUT.smoke/INPUT and exited 0.
    "scf_os_stop",
    "scf_os_thr",
    "scf_os_ndim",
    "smearing_method",
    "smearing_sigma",
    "nelec",
    "nbands",  # cap the number of bands (diagonaliser nev) — auto is
    # 1.2× occupied; test whether 1.1×/1.15× still converges the energy
    # (check the top-band occupations ≈ 0 in the log)
    "symmetry",  # 0 disables rhog_symmetry — a known ABACUS bottleneck
    # on large supercells (density symmetrization can exceed 80% of the
    # wall time; with a 1×1×1 k-mesh it is pure overhead).
    "gamma_only",  # 1 → real-matrix fast path for a single Gamma point
    "ks_solver",  # genelpa/elpa/scalapack_gvx/cg/dav/pexsi
    "calculation",  # scf (default) / nscf — nscf runs on the density
    # read via ``init_chg file`` and needs ``nbands``
    # ------------------------------------------------------------------
    # PEXSI (LCAO + Γ only). All of these are per-cell overridable so a
    # verification step (one knob per step, para_test style) can put every
    # value of that knob into a single protocol preset:
    #   1_method     pexsi_method       1 contour / 2 Moussa / 3 AAA
    #   2_npole      pexsi_npole        hard table caps: m2 <= 61, m3 <= 40,
    #                                   m1 <= 120 (getPole error otherwise)
    #   3_delta_e    pexsi_delta_e      spectral radius of S^-1 H, in Ry
    #   4_mu_bounds  pexsi_mu_lower / pexsi_mu_upper
    #                ★ the mu search only resolves mu to
    #                  (mu_upper - mu_lower)/81 Ry. ABACUS's default ∓10 Ry
    #                  gives 0.25 Ry = 3.4 eV, which is why the returned
    #                  density matrix carries a ~13 % charge error and the
    #                  SCF never closes. Tighten it around the Fermi level
    #                  (estimate mu from a converged genelpa run: where the
    #                  occupations cross 2 -> 0 in OUT.<suffix>/eig_occ.txt).
    #   5_temp       pexsi_temp         must equal smearing_sigma (Ry)
    #   6_mixing     mixing_beta        PEXSI limit-cycles at 0.5
    #   7_nproc_pole pexsi_nproc_pole   must divide the MPI rank count
    # ------------------------------------------------------------------
    "pexsi_npole",
    "pexsi_temp",
    "pexsi_nproc_pole",
    "pexsi_elec_thr",
    "pexsi_mu_thr",
    "pexsi_method",
    "pexsi_delta_e",
    "pexsi_mu_lower",
    "pexsi_mu_upper",
    "pexsi_mu",
    "pexsi_gap",
    "pexsi_inertia",
    "pexsi_nmax",
    "pexsi_zero_thr",
    "pexsi_ordering",
    "pexsi_row_ordering",
    "pexsi_nproc",
    "pexsi_symm",
    "pexsi_trans",
    "pexsi_storage",
    "pexsi_comm",
)


def make_supercell_structure(structure, matrix):
    """Return a ``StructureData`` of the supercell defined by ``matrix``.

    ``matrix`` is a 3×3 integer transformation (rows = supercell lattice
    vectors in terms of the primitive cell). Uses pymatgen's
    ``Structure.make_supercell`` so non-diagonal matrices work too.
    """
    from aiida_uranium_workflow.utils.elastic import (
        pymatgen_to_structure,
        structure_to_pymatgen,
    )

    pmg = structure_to_pymatgen(structure)
    supercell = pmg.make_supercell([list(map(int, row)) for row in matrix])
    return pymatgen_to_structure(supercell)


@calcfunction
def _collect_supercell_results(results: orm.Dict) -> orm.Dict:
    """Package the per-supercell SCF results into a JSON-safe Dict."""
    return orm.Dict(dict=dict(results))


class SupercellScfWorkChain(WorkChain):
    """Compute SCF total energies for a list of supercell matrices."""

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
            "supercell_parameters",
            valid_type=orm.Dict,
            required=True,
            help='Supercell protocol: {"supercells": [{"matrix": [[...]], '
            '"label": ..., "kpoints_mesh": [...], "scf_thr": ...}, ...]}',
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.generate_supercells,
            cls.submit_scfs,
            cls.inspect_scfs,
            cls.collect_results,
        )

        spec.exit_code(0, "SUCCESS", "All supercell SCFs finished OK.")
        spec.exit_code(300, "ERROR_CHILD", "A supercell SCF child failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse a supercell total energy.")

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def generate_supercells(self):
        """Build one StructureData per supercell matrix."""
        para = self.inputs.supercell_parameters.get_dict()
        supercells = para.get("supercells") or []
        if not supercells:
            return self.exit_codes.ERROR_PARSER

        cells = []
        for idx, entry in enumerate(supercells):
            matrix = entry.get("matrix")
            if matrix is None or len(matrix) != 3:
                self.report(f"supercell entry {idx} has no valid 3×3 matrix; skipping")
                continue
            label = str(entry.get("label") or f"cell{idx}")
            structure = make_supercell_structure(self.inputs.structure, matrix)
            cells.append(
                {
                    "idx": idx,
                    "label": label,
                    "matrix": matrix,
                    "entry": entry,
                    "structure": structure,
                }
            )
            self.report(
                f"supercell {label}: {len(structure.sites)} atoms, " f"matrix {matrix}"
            )

        if not cells:
            return self.exit_codes.ERROR_PARSER
        self.ctx.cells = cells
        return None

    def submit_scfs(self):
        """Submit one fixed-lattice SCF per supercell."""
        base = dict(self.inputs.base) if "base" in self.inputs else {}
        for cell in self.ctx.cells:
            child_inputs = self._scf_inputs_for(cell, base)
            running = self.submit(_PluginBaseWorkChain, **child_inputs)
            self.report(f"submitted SCF pk={running.pk} for supercell {cell['label']}")
            self.to_context(**{f"scf_{cell['idx']}": running})

    def _scf_inputs_for(self, cell, base):
        """Merge the per-supercell SCF overrides on top of the base."""
        import json

        entry = cell["entry"]
        # Base top-level keys (kpoints / kpoints_distance / pseudo_family
        # / …) carry over as-is; the per-supercell entry may override the
        # k-points below.
        child_inputs = {k: v for k, v in base.items() if k != "abacus"}

        abacus = {}
        base_abacus = base.get("abacus", {}) if base else {}
        for key, value in base_abacus.items():
            if key == "parameters":
                continue
            abacus[key] = value  # code / metadata — reuse the orm nodes

        # SCF parameters: start from the base preset, apply overrides.
        params = {}
        pnode = base_abacus.get("parameters")
        if pnode is not None:
            params = pnode.get_dict() if hasattr(pnode, "get_dict") else dict(pnode)
        params = json.loads(json.dumps(params))  # JSON-safe deep copy
        pinput = params.setdefault("input", {})
        for key in _SCF_OVERRIDE_KEYS:
            if key in entry:
                pinput[key] = entry[key]
        abacus["parameters"] = orm.Dict(dict=params)
        abacus["structure"] = cell["structure"]
        child_inputs["abacus"] = abacus

        # Do not let AbacusBaseWorkChain auto-restart a failed supercell
        # SCF: a restart re-runs the whole expensive 1024-atom
        # calculation, and the electronic-convergence retry ladder
        # (scf_nmax → mixing_beta) is counter-productive when the
        # per-cell SCF settings are already tuned. Force
        # ``max_iterations = 0`` (overriding anything the base carried).
        child_inputs["max_iterations"] = orm.Int(0)

        # K-points: per-supercell mesh / distance wins over the base.
        if "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            child_inputs.pop("kpoints_distance", None)
            KpointsData = DataFactory("core.array.kpoints")
            kpoints = KpointsData()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            child_inputs["kpoints"] = kpoints
        elif "kpoints_distance" in entry:
            child_inputs.pop("kpoints", None)
            child_inputs["kpoints_distance"] = orm.Float(
                float(entry["kpoints_distance"])
            )

        child_inputs["metadata"] = {"label": f"supercell_{cell['label']}"}
        return child_inputs

    def inspect_scfs(self):
        """Record which supercell SCFs failed, then keep going.

        A parameter scan (see the ``pexsi_*`` verification steps) routinely
        contains cells that are *expected* to fail — a pole count outside
        PEXSI's table, a solve that cannot converge. Aborting here would
        throw away the successful cells and leave the whole WorkChain
        without ``output_parameters``, so the report/collector could not
        show anything. Instead the failures are noted, ``collect_results``
        still writes every cell into the output Dict, and the WorkChain
        returns ``ERROR_CHILD`` at the very end so the batch is still
        flagged as failed.
        """
        failed = []
        for cell in self.ctx.cells:
            child = getattr(self.ctx, f"scf_{cell['idx']}", None)
            if child is None or not child.is_finished_ok:
                failed.append(cell["label"])
                self.report(
                    f"supercell {cell['label']} SCF failed "
                    f"(pk={getattr(child, 'pk', None)}, "
                    f"exit={getattr(child, 'exit_status', None)})"
                )
        self.ctx.failed_cells = failed
        if failed:
            self.report(f"{len(failed)}/{len(self.ctx.cells)} supercell SCF(s) failed: {failed}")
        else:
            self.report("all supercell SCFs finished OK")
        return None

    def collect_results(self):
        """Collect energies / volumes / SCF stats and package results.

        Cells whose SCF failed are recorded too (``status``/``exit_status``,
        ``energy`` = None) so a partially failed scan is still tabulated;
        the missing total energy is no longer fatal.
        """
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        results = {"workflow": "supercell", "backend": "abacus", "cells": []}
        for cell in self.ctx.cells:
            child = getattr(self.ctx, f"scf_{cell['idx']}")
            summary = fetch_summary(child, "abacus")
            energy = summary.get("energy_ev")
            volume = float(cell["structure"].get_cell_volume())
            natoms = len(cell["structure"].sites)
            entry = {
                "label": cell["label"],
                "matrix": cell["matrix"],
                "natoms": natoms,
                "volume": round(volume, 6),
                "volume_units": "A^3",
                "energy": float(energy) if energy is not None else None,
                "energy_units": "eV",
                "energy_per_atom": (float(energy) / natoms) if energy is not None and natoms else None,
                "energy_per_atom_units": "eV/atom",
                "time_s": summary.get("time_s"),
                "scf_steps": summary.get("scf_steps"),
                "scf_pk": child.pk,
                "status": "ok" if energy is not None else "failed",
                "exit_status": getattr(child, "exit_status", None),
            }
            if energy is None:
                self.report(f"supercell {cell['label']} has no total_energy (recorded as failed)")
            results["cells"].append(entry)
        failed = [c["label"] for c in results["cells"] if c["status"] != "ok"]
        results["failed_cells"] = failed
        results["n_cells"] = len(results["cells"])
        results["n_failed"] = len(failed)
        self.out(
            "output_parameters",
            _collect_supercell_results(results=orm.Dict(dict=results)),
        )
        if failed:
            self.report(f"finished with {len(failed)} failed cell(s): {failed}")
            return self.exit_codes.ERROR_CHILD
        return None
