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

A cell whose host or defect child did not finish OK (or whose energy
cannot be read) is **recorded as a failure and skipped** — the remaining
cells still get their formation energies and the WorkChain stores
``output_parameters`` with ``cells`` (successes), ``failures`` (one entry
per failed cell, with the child pks / exit status / message) and a
``status`` field (``complete`` / ``partial``). It then exits with
``PARTIAL`` (310) so automation sees the problem while the report still
renders the data that could be computed. Only when *no* cell succeeded
does it exit with ``ERROR_CHILD`` (300) and store nothing.

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
    normalize_supercell_cells,
    normalize_supercell_matrices,
    supercell_label,
)

#: wf_parameters key choosing relax vs plain SCF.
MODE_KEY = "mode"

#: Supported defect types.
DEFECT_TYPES = ("vacancy", "interstitial")


@calcfunction
def _build_defect_cells_cf(structure, defect_dict, matrices_dict):
    """Build host + defect structures for every requested supercell.

    ``matrices_dict`` is ``{"matrices": [[...], ...]}`` — an empty list
    means "use the input structure as the host cell". Returns a dict of
    stored ``StructureData`` nodes keyed ``host_<i>`` / ``defect_<i>``
    (``i`` = index in the matrix list), which the WorkChain links to its
    context.
    """
    matrices = list(matrices_dict.get_dict().get("matrices") or [])
    d = defect_dict.get_dict()
    out = {}
    if not matrices:
        # Clone the host: returning the *input* node as an output of the
        # same calcfunction would create a cycle in the provenance graph.
        out["host_0"] = structure.clone()
        out["defect_0"] = _create_defect_structure(out["host_0"], d)
        return out
    for i, matrix in enumerate(matrices):
        host = make_supercell(structure, list(matrix))
        out[f"host_{i}"] = host
        out[f"defect_{i}"] = _create_defect_structure(host, d)
    return out


@calcfunction
def _make_kpoints_cf(spec, structure):
    """Build the per-cell ``KpointsData`` from a cell spec.

    ``spec`` is ``{"mesh": [n1, n2, n3]}`` or ``{"distance": float}``
    (the latter is converted to a uniform mesh from the cell's reciprocal
    lattice, mirroring aiida-abacus's ``kpoints_distance`` handling).
    """
    from aiida.plugins import DataFactory

    d = spec.get_dict()
    KpointsData = DataFactory("core.array.kpoints")
    kpoints = KpointsData()
    if d.get("mesh"):
        kpoints.set_kpoints_mesh([int(v) for v in d["mesh"]])
    elif d.get("distance"):
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints_mesh_from_density(float(d["distance"]))
    else:
        raise ValueError(
            f"cell k-point override needs 'kpoints_mesh' or "
            f"'kpoints_distance', got {d!r}"
        )
    return kpoints


def _create_defect_structure(host, defect: dict):
    """Build the defective structure (shared by the calcfunction wrappers)."""
    dtype = str(defect.get("type", "")).lower()
    if dtype == "vacancy":
        structure, _removed = create_vacancy(host, int(defect["site_index"]))
    elif dtype == "interstitial":
        structure = create_interstitial(host, defect["element"], list(defect["position"]))
    else:
        raise ValueError(
            f"Unknown defect type {dtype!r}; expected one of {DEFECT_TYPES}"
        )
    return structure


@calcfunction
def _summarize_defect_cells_cf(summary):
    """Compute the per-cell formation energies and build output_parameters.

    ``summary`` is ``{"mode": str, "defect": {...}, "cells": [
    {"label", "matrix", "host_natoms", "defect_natoms",
     "energy_host_ev", "energy_defect_ev"}, ...], "failures": [...],
    "n_cells_requested": int}``.

    ``cells`` holds only the cells that could be computed; ``failures``
    (cells whose children failed or whose energy was unreadable) is copied
    through so a partial scan still produces a report. The returned Dict
    always carries ``cells`` (one entry per *successful* supercell, with
    its own formation energy), plus ``status`` ('complete'/'partial'),
    ``n_cells_requested`` and ``n_cells_failed``; for a single cell the
    legacy flat keys (``formation_energy_ev``, ``host_natoms``, …) are
    also present for backwards compatibility.
    """
    data = summary.get_dict()
    defect = data.get("defect") or {}
    mode = data.get("mode")
    failures = list(data.get("failures") or [])
    n_requested = int(data.get("n_cells_requested") or 0) or (
        len(data.get("cells", [])) + len(failures)
    )

    cells_out = []
    for cell in data.get("cells", []):
        res = formation_energy(
            cell["energy_defect_ev"],
            cell["energy_host_ev"],
            host_natoms=cell["host_natoms"],
            defect_natoms=cell["defect_natoms"],
        )
        res.update(
            {
                "label": cell.get("label"),
                "matrix": cell.get("matrix"),
                "kpoints_mesh": cell.get("kpoints_mesh"),
                "kpoints_distance": cell.get("kpoints_distance"),
                # per-child runtime for the report (wall time / SCF steps)
                "time_host_s": cell.get("time_host_s"),
                "time_defect_s": cell.get("time_defect_s"),
                "time_total_s": cell.get("time_total_s"),
                "scf_steps_host": cell.get("scf_steps_host"),
                "scf_steps_defect": cell.get("scf_steps_defect"),
                "host_pk": cell.get("host_pk"),
                "defect_pk": cell.get("defect_pk"),
                "scf": cell.get("scf") or {},
                "mode": mode,
                "defect": defect,
                "status": "ok",
            }
        )
        cells_out.append(res)

    out: Dict[str, Any] = {
        "workflow": "defects",
        "mode": mode,
        "defect": defect,
        "status": "partial" if failures else "complete",
        "n_cells": len(cells_out),
        "n_cells_requested": n_requested,
        "n_cells_failed": len(failures),
        "cells": cells_out,
        "failures": failures,
        "failed_cells": [f.get("label") for f in failures],
        "supercell_matrices": [c.get("matrix") for c in cells_out],
    }
    if len(cells_out) == 1:
        # legacy flat layout (single supercell) — unchanged output keys.
        # ``status`` must not be copied: the top-level value describes the
        # whole scan ('partial' when other cells failed) while the cell's
        # own value is always 'ok'.
        flat = dict(cells_out[0])
        flat.pop("label", None)
        flat.pop("matrix", None)
        flat.pop("status", None)
        out.update(flat)
    else:
        out["formation_energies_ev"] = [c["formation_energy_ev"] for c in cells_out]
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
            "supercell_matrices",
            valid_type=orm.List,
            required=False,
            help=(
                "List of supercell matrices for a supercell-size scan, e.g. "
                "[[2, 2, 2], [3, 3, 3], [4, 4, 4]] (each entry a diagonal "
                "shorthand or a full 3x3 matrix). One host + one defect "
                "calculation is run per matrix and every cell gets its own "
                "formation energy in 'output_parameters.cells'."
            ),
        )
        spec.input(
            "supercell_cells",
            valid_type=orm.List,
            required=False,
            help=(
                "Per-cell spec for a supercell scan with individual settings, "
                "e.g. [{'matrix': [2, 2, 2], 'kpoints_mesh': [7, 7, 7]}, "
                "{'matrix': [3, 3, 3], 'kpoints_mesh': [5, 5, 5]}]. Each "
                "entry takes 'matrix' plus optional 'kpoints_mesh' "
                "([n1, n2, n3]), 'kpoints_distance' (1/A) and 'label', plus "
                "per-cell SCF overrides such as {'matrix': [6, 6, 6], "
                "'mixing_ndim': 24, 'scf_nmax': 250}. "
                "Takes precedence over 'supercell_matrices'."
            ),
        )
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            required=False,
            help=(
                "Restart budget handed to each child AbacusBaseWorkChain "
                "(ABACUS only; FLEUR manages its own restarts). Defaults to "
                "0 — a failed host/defect SCF is NOT auto-restarted, because "
                "the plugin's retry ladder (scf_nmax → mixing_beta) is "
                "counter-productive once the per-cell SCF settings are tuned "
                "(lowering mixing_beta below 0.1 even disables ABACUS's "
                "Kerker preconditioner) and every retry re-runs the whole "
                "expensive supercell."
            ),
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
        spec.exit_code(
            300,
            "ERROR_CHILD",
            "No cell could be computed (every host/defect calculation failed).",
        )
        spec.exit_code(305, "ERROR_PARSER", "Failed to read the total energies.")
        spec.exit_code(
            310,
            "PARTIAL",
            "Some cells failed; the remaining formation energies were computed "
            "and stored (see 'failures' in output_parameters).",
        )

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def generate_structures(self):
        """Build the host supercell(s) and the defective structure(s).

        One (host, defect) pair per requested supercell matrix; without any
        matrix the input structure itself is the host cell.
        """
        specs = self._resolve_cells()
        matrices = [spec["matrix"] for spec in specs]
        if any(m is None for m in matrices):
            # a cell without a matrix (= the input cell) is only meaningful
            # on its own
            if len(matrices) != 1:
                raise ValueError(
                    "a cell without 'matrix' can only be the only cell of the "
                    f"run; got matrices {matrices!r}"
                )
            matrices = []  # the calcfunction clones the input structure
        built = _build_defect_cells_cf(
            structure=self.inputs.structure,
            defect_dict=self.inputs.defect,
            matrices_dict=orm.Dict(dict={"matrices": matrices}),
        )
        # ``built`` is a dict of stored StructureData nodes keyed
        # ``host_<i>`` / ``defect_<i>`` (a calcfunction may return a dict).
        self.ctx.cells = []
        for i, spec in enumerate(specs):
            matrix = spec["matrix"]
            cell = {
                "index": i,
                "matrix": matrix,
                "label": spec.get("label")
                or (supercell_label(matrix) if matrix is not None else "host_cell"),
                "host": built[f"host_{i}"],
                "defect": built[f"defect_{i}"],
                "kpoints_mesh": spec.get("kpoints_mesh"),
                "kpoints_distance": spec.get("kpoints_distance"),
                # per-cell SCF overrides (ABACUS INPUT keys), e.g.
                # {'mixing_ndim': 24} for a large supercell
                "scf": dict(spec.get("scf") or {}),
            }
            # per-cell k-points: a KpointsData node built by a calcfunction
            if cell["kpoints_mesh"] or cell["kpoints_distance"]:
                cell["kpoints"] = _make_kpoints_cf(
                    spec=orm.Dict(
                        dict={
                            "mesh": cell["kpoints_mesh"],
                            "distance": cell["kpoints_distance"],
                        }
                    ),
                    structure=cell["host"],
                )
            self.ctx.cells.append(cell)
            self.report(
                f"cell {cell['label']}: host {len(cell['host'].sites)} sites, "
                f"defect {len(cell['defect'].sites)} sites, "
                f"k-mesh {cell['kpoints_mesh'] or cell['kpoints_distance'] or 'preset'}"
            )
        # legacy single-cell handles
        self.ctx.host = self.ctx.cells[0]["host"]
        self.ctx.defect = self.ctx.cells[0]["defect"]
        self.ctx.defect_dict = dict(self.inputs.defect.get_dict())

    def submit_calculations(self):
        """Submit the host and defect calculations of every cell in parallel."""
        from aiida.engine import ToContext

        wc_cls = self._wc_cls()
        submissions = {}
        for cell in self.ctx.cells:
            i = cell["index"]
            for kind in ("host", "defect"):
                inputs = self._make_calc_inputs(
                    cell[kind],
                    f"{kind}_{cell['label']}",
                    kpoints=cell.get("kpoints"),
                    scf_overrides=cell.get("scf") or None,
                )
                submissions[f"{kind}_{i}_wc"] = self.submit(wc_cls, **inputs)
        self.report(f"submitted {len(submissions)} calculations")
        return ToContext(**submissions)

    def gather_results(self):
        """Read energies, compute one formation energy per cell, store outputs.

        A cell whose host/defect child failed (or whose energy cannot be
        read) is recorded in ``failures`` and skipped, so a partially
        failed scan still yields the formation energies of the cells that
        did finish — see the module docstring.
        """
        cells_data = []
        ok_cells = []
        failures = []
        for cell in self.ctx.cells:
            i = cell["index"]
            host_wc = getattr(self.ctx, f"host_{i}_wc", None)
            defect_wc = getattr(self.ctx, f"defect_{i}_wc", None)

            host_ok = bool(getattr(host_wc, "is_finished_ok", False))
            defect_ok = bool(getattr(defect_wc, "is_finished_ok", False))
            if host_wc is None or defect_wc is None or not host_ok or not defect_ok:
                failures.append(
                    {
                        "label": cell["label"],
                        "matrix": cell["matrix"],
                        "reason": "child_failed",
                        "host_ok": host_ok,
                        "defect_ok": defect_ok,
                        "host_pk": getattr(host_wc, "pk", None),
                        "host_exit_status": getattr(host_wc, "exit_status", None),
                        "host_exit_message": getattr(host_wc, "exit_message", None),
                        "defect_pk": getattr(defect_wc, "pk", None),
                        "defect_exit_status": getattr(defect_wc, "exit_status", None),
                        "defect_exit_message": getattr(defect_wc, "exit_message", None),
                    }
                )
                self.report(
                    f"cell {cell['label']}: SKIPPED — host finished_ok={host_ok} "
                    f"(pk={getattr(host_wc, 'pk', None)}, "
                    f"exit={getattr(host_wc, 'exit_status', None)}), defect "
                    f"finished_ok={defect_ok} "
                    f"(pk={getattr(defect_wc, 'pk', None)}, "
                    f"exit={getattr(defect_wc, 'exit_status', None)})"
                )
                continue

            try:
                e_host = self._read_energy(host_wc)
                e_defect = self._read_energy(defect_wc)
            except Exception as exc:  # noqa: BLE001 — one cell, not the whole scan
                failures.append(
                    {
                        "label": cell["label"],
                        "matrix": cell["matrix"],
                        "reason": "energy_read_failed",
                        "host_pk": getattr(host_wc, "pk", None),
                        "defect_pk": getattr(defect_wc, "pk", None),
                        "error": repr(exc),
                    }
                )
                self.report(
                    f"cell {cell['label']}: SKIPPED — could not read total "
                    f"energies: {exc!r}"
                )
                continue

            ok_cells.append(cell)
            # per-child runtime (wall time + electronic steps) for the report
            host_sum = self._read_summary(host_wc) or {}
            defect_sum = self._read_summary(defect_wc) or {}
            t_host, t_defect = host_sum.get("time_s"), defect_sum.get("time_s")
            cells_data.append(
                {
                    "label": cell["label"],
                    "matrix": cell["matrix"],
                    "host_natoms": len(cell["host"].sites),
                    "defect_natoms": len(cell["defect"].sites),
                    "energy_host_ev": e_host,
                    "energy_defect_ev": e_defect,
                    "kpoints_mesh": cell.get("kpoints_mesh"),
                    "kpoints_distance": cell.get("kpoints_distance"),
                    "scf": cell.get("scf") or {},
                    "time_host_s": t_host,
                    "time_defect_s": t_defect,
                    "time_total_s": (
                        t_host + t_defect
                        if t_host is not None and t_defect is not None
                        else None
                    ),
                    "scf_steps_host": host_sum.get("scf_steps"),
                    "scf_steps_defect": defect_sum.get("scf_steps"),
                    "host_pk": getattr(host_wc, "pk", None),
                    "defect_pk": getattr(defect_wc, "pk", None),
                }
            )

        if not cells_data:
            self.report(
                f"all {len(self.ctx.cells)} cell(s) failed — no formation "
                f"energy computed; failures: "
                f"{[f['label'] for f in failures]}"
            )
            return self.exit_codes.ERROR_CHILD

        try:
            result = _summarize_defect_cells_cf(
                summary=orm.Dict(
                    dict={
                        "mode": self._mode(),
                        "defect": self.ctx.defect_dict,
                        "cells": cells_data,
                        "failures": failures,
                        "n_cells_requested": len(self.ctx.cells),
                    }
                )
            )
            # ``result`` is a calcfunction output — a stored node — so it
            # can be returned as a WorkChain output (creating an unstored
            # ``orm.Dict`` here would make AiiDA's update_outputs reject it).
            self.out("output_parameters", result)
            # legacy structure outputs point at the first computed cell
            self.out("host_structure", ok_cells[0]["host"])
            self.out("defect_structure", ok_cells[0]["defect"])
        except Exception as exc:  # noqa: BLE001 — surface as ERROR_PARSER
            self.report(f"gather_results failed: {exc!r}")
            return self.exit_codes.ERROR_PARSER

        if failures:
            self.report(
                f"computed {len(cells_data)}/{len(self.ctx.cells)} cells; "
                f"{len(failures)} failed ({[f['label'] for f in failures]}) — "
                f"partial results stored"
            )
            return self.exit_codes.PARTIAL
        return None  # SUCCESS

    def _child_max_iterations(self) -> int:
        """Restart budget for the child host/defect WorkChains (default 0).

        ``0`` disables the plugin's automatic SCF retry ladder: the per-cell
        settings are tuned already and every retry re-runs the whole
        supercell. Callers that want the old behaviour pass
        ``max_iterations=<n>`` to the WorkChain.
        """
        if "max_iterations" in self.inputs:
            return int(self.inputs.max_iterations.value)
        return 0

    def _resolve_cells(self) -> list:
        """Return the per-cell specs for this WorkChain.

        ``supercell_cells`` (per-cell settings, e.g. k-meshes) wins over
        ``supercell_matrices`` (plain scan), which wins over the legacy
        single ``supercell_matrix``. With nothing given there is exactly
        one cell that uses the input structure as the host.
        """
        if "supercell_cells" in self.inputs:
            return normalize_supercell_cells(
                list(self.inputs.supercell_cells.get_list())
            )
        if "supercell_matrices" in self.inputs:
            return [
                {"matrix": m}
                for m in normalize_supercell_matrices(
                    list(self.inputs.supercell_matrices.get_list())
                )
            ]
        if "supercell_matrix" in self.inputs:
            return [
                {"matrix": m}
                for m in normalize_supercell_matrices(
                    list(self.inputs.supercell_matrix.get_list())
                )
            ]
        return [{"matrix": None}]

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

    def _read_summary(self, workchain) -> dict:
        """Optional per-child summary used for the runtime columns.

        Returns ``{"time_s": float | None, "scf_steps": int | None}`` (or
        ``{}`` when the backend has no unified summary, e.g. FLEUR — the
        report then prints ``—``). The ABACUS subclass delegates to
        :func:`aiida_uranium_workflow.utils.parsers.fetch_summary`.
        """
        return {}

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
