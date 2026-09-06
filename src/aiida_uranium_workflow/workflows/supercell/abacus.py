"""ABACUS supercell SCF WorkChain.

Runs one fixed-lattice SCF per supercell matrix from the protocol
(``parameters/supercell.yml``), each with its own SCF parameters
(k-point mesh, scf_thr, mixing settings, …) applied on top of the
shared ``parameters/abacus/scf.yml`` preset.

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
    "mixing_beta",
    "mixing_beta_mag",
    "mixing_angle",
    "mixing_ndim",
    "mixing_type",
    "smearing_method",
    "smearing_sigma",
    "nelec",
    "symmetry",  # 0 disables rhog_symmetry — a known ABACUS bottleneck
    # on large supercells (density symmetrization can exceed 80% of the
    # wall time; with a 1×1×1 k-mesh it is pure overhead).
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
        # / max_iterations / …) carry over as-is; the per-supercell entry
        # may override the k-points / restart count below.
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

        # AbacusBaseWorkChain restart budget (top-level input, not part
        # of the SCF ``input`` block): how many times a failed SCF is
        # retried with adjusted settings (scf_nmax / mixing_beta). A
        # per-supercell ``max_iterations`` wins over the shared base
        # (base-level value already carried over in ``child_inputs``).
        if "max_iterations" in entry:
            child_inputs["max_iterations"] = orm.Int(int(entry["max_iterations"]))

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
        """Fail fast if any supercell SCF did not finish OK."""
        for cell in self.ctx.cells:
            child = getattr(self.ctx, f"scf_{cell['idx']}", None)
            if child is None or not child.is_finished_ok:
                self.report(
                    f"supercell {cell['label']} SCF failed "
                    f"(pk={getattr(child, 'pk', None)}, "
                    f"exit={getattr(child, 'exit_status', None)})"
                )
                return self.exit_codes.ERROR_CHILD
        self.report("all supercell SCFs finished OK")
        return None

    def collect_results(self):
        """Collect energies / volumes / SCF stats and package results."""
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        results = {"workflow": "supercell", "backend": "abacus", "cells": []}
        for cell in self.ctx.cells:
            child = getattr(self.ctx, f"scf_{cell['idx']}")
            summary = fetch_summary(child, "abacus")
            energy = summary.get("energy_ev")
            if energy is None:
                self.report(f"supercell {cell['label']} has no total_energy")
                return self.exit_codes.ERROR_PARSER
            volume = float(cell["structure"].get_cell_volume())
            natoms = len(cell["structure"].sites)
            results["cells"].append(
                {
                    "label": cell["label"],
                    "matrix": cell["matrix"],
                    "natoms": natoms,
                    "volume": round(volume, 6),
                    "volume_units": "A^3",
                    "energy": float(energy),
                    "energy_units": "eV",
                    "time_s": summary.get("time_s"),
                    "scf_steps": summary.get("scf_steps"),
                    "scf_pk": child.pk,
                }
            )
        self.out(
            "output_parameters",
            _collect_supercell_results(results=orm.Dict(dict=results)),
        )
        return None
