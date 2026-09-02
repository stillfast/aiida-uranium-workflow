"""Shared energy + wall-clock parser for AiiDA CalcJob nodes.

Used by every WorkChain's ``parse_and_gather_*`` calcfunction to fill
the ``total_energy`` and ``wall_time_seconds`` columns of the parent
WorkChain's ``output_parameters``.

The implementation was originally in ``temporary/parser_energy_time.py``
(used as a one-off debugging script). It is reproduced here so the
log-parsing logic can be:

* imported without going through ``aiida.orm.load_profile``,
* exercised by unit tests with synthetic log strings,
* shared across the smear / convergence / magmom workflows.

Backend semantics
-----------------

* **ABACUS** — total energy is read from ``node.outputs.misc`` (the
  ``aiida-abacus`` parser stores ``total_energy``). Wall-clock time is
  parsed from the last ``Total  Time : H h M mins S secs`` line of
  ``OUT.aiida/running_scf.log`` (or ``OUT.aiida/running_cell-relax.log``
  when the former is missing).

* **VASP** — total energy is read from
  ``node.outputs.misc``'s nested
  ``total_energies["energy_extrapolated"]``. Wall-clock CPU time is read
  from ``misc.run_stats.total_cpu_time_used``.

Both backends return ``(energy, time_seconds)``. Either value can be
``None`` when the corresponding field is missing or the log file can't
be opened.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Pure helpers (no AiiDA imports) — easy to unit-test
# ---------------------------------------------------------------------------


#: Regex for the ABACUS wall-time line.
#: ``Total  Time : 0 h 5 mins 12 secs`` (whitespace is loose).
_ABACUS_TOTAL_TIME_RE = re.compile(
    r"Total\s+Time\s*:\s*(\d+)\s*h\s*(\d+)\s*mins\s*(\d+)\s*secs"
)


def parse_total_time(content: str) -> Optional[float]:
    """Parse ``Total  Time : H h M mins S secs`` from an ABACUS log.

    Returns total seconds (float) or ``None`` if the line is not present.
    When the line appears more than once (e.g. multi-stage runs that
    append), the *last* occurrence wins — usually the run-level summary
    rather than a per-stage one.
    """
    total_time: Optional[float] = None
    for line in content.splitlines():
        match = _ABACUS_TOTAL_TIME_RE.search(line)
        if match:
            hours, minutes, seconds = (int(match.group(i)) for i in (1, 2, 3))
            total_time = float(hours * 3600 + minutes * 60 + seconds)
    return total_time


def parse_abacus_last_energy(content: str) -> Optional[float]:
    """Read the last ``E_tot`` / ``total energy`` line from an ABACUS log.

    The ``aiida-abacus`` parser typically already extracts the total
    energy into ``node.outputs.misc``; this helper is a fallback for
    cases where ``misc`` is unavailable (e.g. mid-WorkChain migration
    scripts, or unit tests). The first numeric column after the label
    is taken as the energy in eV.

    Returns ``None`` when no recognisable line is found.
    """
    last: Optional[float] = None
    for line in content.splitlines():
        # ``!FINAL_ETOT_IS  <energy> eV`` — common modern footer.
        if "!FINAL_ETOT_IS" in line:
            parts = line.split()
            for idx, token in enumerate(parts):
                if token == "!FINAL_ETOT_IS" and idx + 1 < len(parts):
                    try:
                        last = float(parts[idx + 1])
                    except ValueError:
                        pass
        # ``final etot is <energy>`` — older single-precision footer.
        if "final etot is" in line.lower():
            parts = line.split()
            for idx, token in enumerate(parts):
                if token.lower() == "is" and idx + 1 < len(parts):
                    try:
                        last = float(parts[idx + 1])
                    except ValueError:
                        pass
    return last


# ---------------------------------------------------------------------------
# AiiDA-aware helpers — operate on CalcJob nodes
# ---------------------------------------------------------------------------


def _read_abacus_log(retrieved, log_name: str) -> Optional[str]:
    """Read an ABACUS log file from ``retrieved``; ``None`` on miss."""
    try:
        with retrieved.open(f"OUT.aiida/{log_name}") as handle:
            return handle.read()
    except (OSError, KeyError, FileNotFoundError):
        return None


def fetch_abacus(node) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(energy, wall_time_seconds)`` for an ABACUS CalcJob.

    * Energy: ``node.outputs.misc["total_energy"]`` (fallback to the
      log's last line via :func:`parse_abacus_last_energy`).
    * Time:  ``Total  Time`` line of ``OUT.aiida/running_scf.log`` (or
      ``running_cell-relax.log`` if the SCF log is missing).
    """
    misc = node.outputs.misc.get_dict()
    energy = misc.get("total_energy")
    if energy is None:
        log_text = _read_abacus_log(node.outputs.retrieved, "running_scf.log")
        if log_text is not None:
            energy = parse_abacus_last_energy(log_text)

    wall_time: Optional[float] = None
    for log_name in ("running_scf.log", "running_cell-relax.log"):
        log_text = _read_abacus_log(node.outputs.retrieved, log_name)
        if log_text is None:
            continue
        wall_time = parse_total_time(log_text)
        if wall_time is not None:
            break

    return energy, wall_time


def fetch_vasp(node) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(energy, cpu_time_seconds)`` for a VASP CalcJob.

    * Energy: ``node.outputs.misc["total_energies"]["energy_extrapolated"]``.
    * Time:   ``node.outputs.misc["run_stats"]["total_cpu_time_used"]``.

    Both values are taken straight from the VASP parser's ``misc`` Dict;
    no log file IO is required.
    """
    misc = node.outputs.misc.get_dict()
    energy = None
    energies = misc.get("total_energies")
    if isinstance(energies, dict):
        energy = energies.get("energy_extrapolated")
    cpu_time = misc.get("run_stats", {}).get("total_cpu_time_used")
    return energy, cpu_time


def fetch_energy_time(node, backend: str) -> Tuple[Optional[float], Optional[float]]:
    """Dispatch to the right parser based on ``backend``.

    ``backend`` is the canonical name (``"abacus"`` / ``"vasp"``).
    Unknown backends raise :class:`ValueError`.
    """
    if backend == "abacus":
        return fetch_abacus(node)
    if backend == "vasp":
        return fetch_vasp(node)
    raise ValueError(f"Unknown backend '{backend}'; expected 'abacus' or 'vasp'")


# ---------------------------------------------------------------------------
# Unified summary schema (improve.md Phase A)
# ---------------------------------------------------------------------------


def _count_abacus_elec_steps(retrieved) -> Optional[int]:
    """Count electronic SCF steps from ``OUT.aiida/running_scf.log``.

    ABACUS prints one ``ION=   1  ELEC=   n`` line per electronic step in
    the fixed-lattice SCF log; the count of such lines is the number of
    electronic iterations. Returns ``None`` when the log can't be read.
    """
    log_text = _read_abacus_log(retrieved, "running_scf.log")
    if log_text is None:
        return None
    steps = len(re.findall(r"ION=\s*\d+\s+ELEC=\s*\d+", log_text))
    return steps if steps else None


def fetch_summary(node, backend: str) -> dict:
    """Unified per-calculation summary used by every WorkChain report.

    Returns a single dict with the canonical field names (all optional
    values are ``None`` when unavailable)::

        {
            "energy_ev": float | None,   # backend-native energy in eV
            "time_s":    float | None,   # wall-clock (ABACUS log / VASP cpu)
            "natoms":    int | None,
            "scf_steps": int | None,     # electronic iterations
        }

    ``backend`` is ``"abacus"`` / ``"vasp"``. The energy / time fields
    reuse :func:`fetch_abacus` / :func:`fetch_vasp`; ``natoms`` is the
    site count of the calc's input structure; ``scf_steps`` counts the
    electronic iterations from the log (ABACUS) or is ``None`` when the
    backend parser does not expose them (VASP).
    """
    if backend == "abacus":
        energy, time_s = fetch_abacus(node)
        try:
            natoms = len(node.inputs.abacus.structure.sites)
        except Exception:  # noqa: BLE001 — inputs missing
            natoms = None
        scf_steps = _count_abacus_elec_steps(node.outputs.retrieved)
    elif backend == "vasp":
        energy, time_s = fetch_vasp(node)
        try:
            natoms = len(node.inputs.structure.sites)
        except Exception:  # noqa: BLE001
            natoms = None
        scf_steps = None
    else:
        raise ValueError(f"Unknown backend '{backend}'; expected 'abacus' or 'vasp'")

    return {
        "energy_ev": energy,
        "time_s": time_s,
        "natoms": natoms,
        "scf_steps": scf_steps,
    }
