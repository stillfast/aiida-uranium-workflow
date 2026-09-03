"""Child parser classes: one per backend, producing report-schema records.

A *child* is the process node a parent WorkChain submitted (e.g. an
``abacus.base`` WorkChain inside ``AbacusMagmomWorkChain``). Every
parent's ``gather`` step currently re-implements the same "read the
child's outputs and normalise them" logic by hand. These parser classes
centralise that per-backend knowledge so the parent only has to call
``<Backend>ChildParser().parse(child)`` and gets back one
:class:`~aiida_uranium_workflow.utils.report.schema.ChildRecord` with
canonical, backend-agnostic fields (energy in eV, wall time in seconds,
SCF step count, atom count, exit status, plus backend-specific
magnetism under ``data``).

The data class itself is defined by the *report* side — see
:mod:`aiida_uranium_workflow.utils.report.schema` (report-first: the
report declares what it needs, parsers produce records that satisfy
it). Parsers never define the output layout themselves.

Design notes
------------
* Backends that share a plugin output shape subclass a common parser
  (e.g. both ABACUS and VASP read ``outputs.misc`` for energy / time).
* Parsing is *defensive*: a child that did not finish OK, or whose
  outputs are missing, yields a record with ``None`` fields instead of
  raising — the parent decides what to do with gaps.
"""

from __future__ import annotations

from typing import Any, Dict

from aiida_uranium_workflow.utils.report.schema import ChildRecord


# Backwards-compatible alias: the parser layer used to define its own
# ``ChildResult``; it is now the report-side ``ChildRecord`` so the
# serialised keys (``data``) match what the report reads back.
ChildResult = ChildRecord


# ---------------------------------------------------------------------------
# Parser base
# ---------------------------------------------------------------------------


class ChildParser:
    """Base class for backend-specific child parsers.

    Subclasses implement :meth:`_read` (extract the fields from the
    child's outputs). :meth:`parse` wraps it with the common exit-status
    bookkeeping and converts exceptions into an *empty* record so a
    single malformed child never aborts a parent's gather step.
    """

    backend: str = ""

    def parse(self, child: Any) -> ChildRecord:
        """Parse one child node into a :class:`ChildRecord`.

        ``child`` is the AiiDA process node (WorkChain or CalcJob).
        Unfinished children yield ``finished_ok=False`` and are not
        parsed further (their output namespace may not exist yet).
        """
        result = ChildRecord(
            pk=int(child.pk),
            status=(
                int(child.exit_status)
                if child.exit_status is not None
                else None
            ),
            finished_ok=bool(child.is_finished_ok),
        )
        if not result.finished_ok:
            return result
        try:
            self._read(child, result)
        except Exception:  # noqa: BLE001 — parser must never raise
            return result
        return result

    # -- subclasses implement -------------------------------------------

    def _read(self, child: Any, result: ChildRecord) -> None:
        """Fill ``result`` from ``child.outputs`` / ``child.inputs``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ABACUS child parser
# ---------------------------------------------------------------------------


class AbacusChildParser(ChildParser):
    """Parse an ``abacus.base`` child (ABACUS SCF / relaxation WorkChain).

    Energy / time / steps / natoms come from the unified summary parser
    (``fetch_summary``); magnetism from the ``misc`` output's
    ``magnetism`` (per-iteration cell values) and ``final_magnetism``
    (converged cell total).
    """

    backend = "abacus"

    def _read(self, child: Any, result: ChildRecord) -> None:
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        summary = fetch_summary(child, "abacus")
        result.energy_ev = summary["energy_ev"]
        result.time_s = summary["time_s"]
        result.scf_steps = summary["scf_steps"]
        result.natoms = summary["natoms"]

        misc = child.outputs.misc.get_dict()
        result.data = {
            "magnetism": misc.get("magnetism"),
            "final_magnetism": misc.get("final_magnetism"),
        }

        # nspin (collinear / SOC mode) lives in the ABACUS input.
        try:
            parameters = child.inputs.abacus.parameters.get_dict()
            nspin = parameters.get("input", {}).get("nspin")
            if nspin is not None:
                result.data["nspin"] = nspin
        except Exception:  # noqa: BLE001 — inputs missing
            pass


# ---------------------------------------------------------------------------
# VASP child parser
# ---------------------------------------------------------------------------


class VaspChildParser(ChildParser):
    """Parse a VASP child (``vasp.v2.vasp`` WorkChain / CalcJob).

    Magnetism comes from the ``misc`` output: ``magnetization`` (cell
    total) and ``site_magnetization`` (per-site when parsed).
    """

    backend = "vasp"

    def _read(self, child: Any, result: ChildRecord) -> None:
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        summary = fetch_summary(child, "vasp")
        result.energy_ev = summary["energy_ev"]
        result.time_s = summary["time_s"]
        result.scf_steps = summary["scf_steps"]
        result.natoms = summary["natoms"]

        misc = child.outputs.misc.get_dict()
        result.data = {
            "magnetization": misc.get("magnetization"),
            "site_magnetization": misc.get("site_magnetization"),
        }


# ---------------------------------------------------------------------------
# QE (pw.x) child parser
# ---------------------------------------------------------------------------


class QePwChildParser(ChildParser):
    """Parse a QE ``PwBaseWorkChain`` child.

    Reads the pw.x ``output_parameters``: ``energy`` (already in eV —
    aiida-qe converts from Ry inside its parser), wall time, SCF
    iteration count, and the magnetism fields parsed by aiida-qe.
    """

    backend = "qe"

    def _read(self, child: Any, result: ChildRecord) -> None:
        para = child.outputs.output_parameters.get_dict()

        energy = para.get("energy")
        if energy is not None:
            try:
                result.energy_ev = float(energy)
            except (TypeError, ValueError):
                result.energy_ev = None

        try:
            result.time_s = float(para.get("wall_time_seconds"))
        except (TypeError, ValueError):
            result.time_s = None

        # pw.x prints one ``iteration #`` line per electronic SCF step;
        # aiida-qe exposes the total count at the top level.
        result.scf_steps = para.get("total_number_of_scf_iterations")

        try:
            result.natoms = len(child.inputs.pw.structure.sites)
        except Exception:  # noqa: BLE001 — inputs missing
            result.natoms = None

        # Non-collinear / SOC runs carry the cell magnetisation as a
        # vector (``total_vec`` in the XML); aiida-qe leaves
        # ``total_magnetization`` at 0.0 there. The vector magnitude is
        # ``absolute_magnetization`` — use that for the cell total.
        is_noncolinear = bool(
            para.get("non_colinear_calculation")
            or para.get("spin_orbit_calculation")
        )
        cell_total = (
            para.get("absolute_magnetization")
            if is_noncolinear
            else para.get("total_magnetization")
        )
        result.data = {
            # Key names mirror the legacy gather layout so the report's
            # per-backend tables keep reading the same top-level keys.
            "magnetization": cell_total,
            "absolute_magnetization": para.get("absolute_magnetization"),
            "atomic_magnetic_moments": para.get("atomic_magnetic_moments"),
        }


# ---------------------------------------------------------------------------
# FLEUR child parser
# ---------------------------------------------------------------------------


class FleurScfChildParser(ChildParser):
    """Parse a FLEUR SCF WorkChain child.

    Energy / wall time come from ``output_scf_wc_para`` (Hartree — the
    parent converts to eV when building its report); per-atom magnetic
    3-vectors from the last calc's ``output_parameters``.
    """

    backend = "fleur"

    #: Hartree → eV (CODATA).
    HA_TO_EV = 27.211386245988

    def _read(self, child: Any, result: ChildRecord) -> None:
        scf_para = child.outputs.output_scf_wc_para.get_dict()

        energy_ha = scf_para.get("total_energy")
        if energy_ha is not None:
            try:
                result.energy_ev = float(energy_ha) * self.HA_TO_EV
            except (TypeError, ValueError):
                result.energy_ev = None
        try:
            result.time_s = float(scf_para.get("total_wall_time"))
        except (TypeError, ValueError):
            result.time_s = None

        try:
            last_para = child.outputs.last_calc.output_parameters.get_dict()
        except Exception:  # noqa: BLE001
            last_para = {}
        result.data = {
            # Key names mirror the legacy gather layout (the report
            # reads ``magnetization`` = per-atom 3-vectors).
            "magnetization": last_para.get("magnetic_vec_moments"),
            "total_energy_hartree": energy_ha,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CHILD_PARSERS: Dict[str, ChildParser] = {
    cls.backend: cls()
    for cls in (
        AbacusChildParser,
        VaspChildParser,
        QePwChildParser,
        FleurScfChildParser,
    )
}


def get_child_parser(backend: str) -> ChildParser:
    """Return the parser registered for ``backend``."""
    try:
        return CHILD_PARSERS[backend]
    except KeyError:
        raise ValueError(
            f"No child parser for backend '{backend}'; "
            f"available: {sorted(CHILD_PARSERS)}"
        ) from None
