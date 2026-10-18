"""Sanity checks for per-atom magnetisation seeds.

A magmom sweep pairs a seed list from the protocol
(``parameters/magmom.yml``) with the structure named in ``static``.  Those
seeds are **per atom in site order**, so they are tied to the cell they
were written for: the shipped ``test_mag`` preset seeds the 2-atom bcc
cell (``[[0.0], [0.0]]`` / ``[[4.0], [4.0]]`` / ``[[4.0], [-4.0]]``).

Pointing that preset at a 4-atom ``alpha-uranium`` cell crashed only at
upload time, inside the backend's STRU writer::

    IndexError: list index out of range      # aiida_abacus/calculations.py
    PreSubmitException: exception occurred in presubmit call

Every child WorkChain was already submitted by then, so a config typo
cost a batch of failed processes plus an unhelpful traceback.  Validate
the seeds against the structure in the adapter instead, and say exactly
how many moments the cell needs.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _moment_shape_ok(moment: Any) -> bool:
    """One per-atom seed: a bare number or a 1-/3-component numeric sequence."""
    if _is_number(moment):
        return True
    if isinstance(moment, (str, bytes)):
        return False
    if isinstance(moment, Sequence):
        return len(moment) in (1, 3) and all(_is_number(v) for v in moment)
    return False


def _suggest(config: Sequence[Any], natoms: int) -> str:
    """Render a copy-pasteable configuration of the right length for this cell.

    The given seeds are repeated cyclically: a single ``[[4.0]]`` becomes an
    all-ferromagnetic cell, while ``[[4.0], [-4.0]]`` keeps alternating up /
    down.  Only a starting point — the user edits the pattern.
    """
    if natoms <= 0:
        return "[]"
    if not config:
        config = [[4.0]]
    repeated = [config[i % len(config)] for i in range(natoms)]
    return "[" + ", ".join(_render_moment(moment) for moment in repeated) + "]"


def _seed_number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _render_moment(moment: Any) -> str:
    if _is_number(moment):
        return str(_seed_number(moment))
    return "[" + ", ".join(str(_seed_number(v)) for v in moment) + "]"


def validate_per_atom_seeds(
    seeds: Iterable[Any] | None,
    natoms: int,
    *,
    field: str,
    backend: str,
    structure_label: str = "",
    document: str = "parameters/magmom.yml",
) -> None:
    """Raise ``ValueError`` unless every configuration seeds all ``natoms`` sites.

    ``seeds`` is the protocol list of initial magnetic configurations, each
    one a per-atom list.  A configuration that lists fewer (or more) moments
    than the structure has sites can only fail later, inside the backend —
    stop it here while the message can still point at the YAML.
    """
    if not seeds:
        return

    cell = f"the structure has {natoms} sites"
    if structure_label:
        cell += f" ({structure_label})"

    for index, config in enumerate(seeds):
        if isinstance(config, (str, bytes)) or not isinstance(config, Sequence):
            raise ValueError(
                f"magmom sweep: configuration #{index + 1} of {backend}.{field} "
                f"({document}) must be a per-atom list, got {config!r}."
            )
        if len(config) != natoms:
            unit = "bcc cell" if natoms == 2 else "cell"
            raise ValueError(
                f"magmom sweep: configuration #{index + 1} of {backend}.{field} "
                f"({document}) seeds {len(config)} atoms but {cell}.\n"
                f"  Seeds are per atom in site order, so each configuration must "
                f"list exactly {natoms} moments for this {unit}.\n"
                f"  Got:      {config}\n"
                f"  Expected: {_suggest(list(config), natoms)}\n"
                f"  Fix: add a preset written for this cell to {document} "
                f"(or point 'parameters.magmom' at one), then re-run."
            )
        for atom, moment in enumerate(config):
            if not _moment_shape_ok(moment):
                raise ValueError(
                    f"magmom sweep: configuration #{index + 1} of {backend}.{field} "
                    f"({document}), atom {atom}: {moment!r} is not a magnetic "
                    "moment — use one number per site for collinear spins "
                    "(e.g. [[4.0], [4.0]]) or three for SOC (e.g. [[0.0, 0.0, 4.0]])."
                )


def validate_structure_seeds(
    seeds: Iterable[Any] | None,
    structure,
    *,
    field: str,
    backend: str,
    document: str = "parameters/magmom.yml",
) -> None:
    """:func:`validate_per_atom_seeds` bound to an AiiDA structure node/ASE atoms."""
    try:
        natoms = len(structure.sites)
    except AttributeError:  # plain ASE Atoms
        natoms = len(structure)
    try:
        label = structure.get_formula()
    except AttributeError:
        label = getattr(structure, "get_chemical_formula", lambda: "")()
    validate_per_atom_seeds(
        seeds,
        natoms,
        field=field,
        backend=backend,
        structure_label=label,
        document=document,
    )
