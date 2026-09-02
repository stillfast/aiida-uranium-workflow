"""Extract normalized band / DOS data from an AiiDA WorkChain pk.

This module is the *normalization* layer: regardless of which DFT
backend produced the data (ABACUS, FLEUR, …), the rest of the package
can rely on a single, backend-agnostic :class:`BandData` /
:class:`DosData` schema.

Adding a new backend means adding one extractor function and
registering it in :data:`EXTRACTORS`. The downstream plotting code in
:mod:`.plot` does not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Normalized data containers
# ---------------------------------------------------------------------------


@dataclass
class BandData:
    """Backend-agnostic band structure.

    Attributes
    ----------
    energies : np.ndarray, shape (nbands, nk)
        Band energies in eV.
    kpoints : np.ndarray, shape (nk, 3)
        k-point coordinates in reciprocal Å⁻¹.
    labels : list[str]
        High-symmetry labels for the seekpath (if any).
    label_numbers : list[int]
        Indices into ``kpoints`` that correspond to each label.
    fermi_energy : float
        Fermi energy in eV (used to shift the y-axis so E_F = 0).
    workchain_pk : Optional[int]
        The AiiDA pk of the source WorkChain (for plotting annotations).
    weights : Optional[np.ndarray], shape (nk,)
        k-point weights (degeneracy / path segment lengths) when the
        backend stores them — used by the band-comparison metrics to
        weight the (band, k) sum.
    """

    energies: np.ndarray
    kpoints: np.ndarray
    labels: List[str] = field(default_factory=list)
    label_numbers: List[int] = field(default_factory=list)
    fermi_energy: float = 0.0
    workchain_pk: Optional[int] = None
    weights: Optional[np.ndarray] = None


@dataclass
class DosData:
    """Backend-agnostic density-of-states.

    Attributes
    ----------
    energy : np.ndarray, shape (nenergy,)
        Energy axis in eV.
    total : np.ndarray, shape (nenergy,)
        Total DOS in 1/eV.
    spin : dict[str, np.ndarray]
        Spin-projected channels (e.g. ``{"dos_mx": …, "dos_my": …}``).
        Empty for the spin-summed total DOS.
    fermi_energy : float
        Fermi energy in eV.
    workchain_pk : Optional[int]
        The AiiDA pk of the source WorkChain (for plotting annotations).
    """

    energy: np.ndarray
    total: np.ndarray
    spin: Dict[str, np.ndarray] = field(default_factory=dict)
    fermi_energy: float = 0.0
    workchain_pk: Optional[int] = None


@dataclass
class PdosData:
    """Backend-agnostic projected density of states.

    Attributes
    ----------
    energy : np.ndarray, shape (nenergy,)
        Energy axis in eV (absolute scale; ``fermi_energy`` is given
        separately so the renderer can shift to E_F).
    orbitals : dict[str, np.ndarray]
        Orbital-projected DOS in 1/eV, grouped by angular momentum
        (``"s"`` / ``"p"`` / ``"d"`` / ``"f"``) and summed over all
        atoms and spin channels.
    total : np.ndarray, shape (nenergy,)
        Sum of all orbitals (= total DOS).
    fermi_energy : float
        Fermi energy in eV.
    workchain_pk : Optional[int]
        The AiiDA pk of the source WorkChain (for plotting annotations).
    """

    energy: np.ndarray
    orbitals: Dict[str, np.ndarray]
    total: np.ndarray
    fermi_energy: float = 0.0
    workchain_pk: Optional[int] = None


# ---------------------------------------------------------------------------
# Backend-specific extractors
# ---------------------------------------------------------------------------


def _ensure_profile_loaded():
    """Ensure an AiiDA profile is loaded.

    Calling ``load_profile()`` is idempotent, so this is safe to call
    once at the top of every extractor.
    """
    from aiida import load_profile
    load_profile()


def _resolve_pk(pk_or_uuid):
    """Load a node by pk (int) or UUID (str)."""
    from aiida.orm import load_node
    return load_node(pk_or_uuid)


def _read_fermi_eV(node, prefer=("fermi_energy_band", "fermi_energy_scf")):
    """Return the Fermi energy in eV, picking the first available key.

    The node may be a ``BandsData`` / ``XyData`` / ``WorkChain``. We
    look in (in order):

    1. ``base.attributes.get("fermi_level")`` — preferred (ABACUS).
    2. The output_parameters Dict's ``fermi_energy_band`` / ``..._scf``
       (FLEUR stores Hartree, so we convert).
    """
    raw = node.base.attributes.get("fermi_level") if hasattr(node, "base") else None
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _read_optional_array(node, name: str):
    """Read a repository array, returning ``None`` when absent.

    Wraps :meth:`get_array` so extractors can carry optional extra
    arrays (e.g. k-point weights) without failing on backends that do
    not store them.
    """
    try:
        return np.asarray(node.get_array(name))
    except (KeyError, AttributeError, OSError):
        return None


def extract_abacus(wc) -> Dict[str, Any]:
    """Extract band + DOS from an ``AbacusBandWorkChain`` (pk=339125 family)."""
    bands = wc.outputs.band_structure
    try:
        dos = wc.outputs.dos
    except Exception:
        # ``dos`` is a ``required=False`` port of AbacusBandWorkChain —
        # band-only runs (run_dos=False) carry no DOS output; the band
        # data must still be plottable on its own.
        dos = None

    # Bands — ABACUS stores ``array|bands`` as (1, nk, nbands) in eV.
    energies = np.asarray(bands.get_array("bands"))
    kpoints = np.asarray(bands.get_array("kpoints"))
    if energies.ndim == 3 and energies.shape[0] == 1:
        energies = energies.squeeze(0).T  # (nbands, nk)
    elif energies.ndim == 2 and energies.shape[0] == len(kpoints):
        energies = energies.T

    band_data = BandData(
        energies=energies,
        kpoints=kpoints,
        labels=list(bands.base.attributes.get("labels") or []),
        label_numbers=list(bands.base.attributes.get("label_numbers") or []),
        fermi_energy=_read_fermi_eV(bands),
        workchain_pk=wc.pk,
        weights=_read_optional_array(bands, "weights"),
    )

    # DOS — pk=339399 style: array|energy / array|tdos (already in eV).
    if dos is None:
        dos_data = None
    else:
        energy = np.asarray(dos.get_array("energy"))
        tdos = np.asarray(dos.get_array("tdos"))
        dos_data = DosData(
            energy=energy,
            total=tdos,
            fermi_energy=_read_fermi_eV(dos) or band_data.fermi_energy,
            workchain_pk=wc.pk,
        )
    return {"band": band_data, "dos": dos_data}


def _kpath_labels_from_kpoints(kpoints, cell, symprec=1e-5):
    """Reconstruct seekpath path labels and map them onto ``kpoints``.

    aiida-fleur's banddos ``BandsData`` does not store the seekpath path
    labels, but its k-points ARE the seekpath explicit k-points of the
    (primitive) cell, so each special point appears verbatim in the
    k-point list. We re-run seekpath on the cell to get the ordered
    vertex sequence and match every vertex to its index in ``kpoints``.

    Returns ``(labels, label_numbers)``; both empty when the k-points do
    not match a seekpath path (e.g. other ``kpath`` modes), in which case
    the renderer falls back to plain integer ticks.
    """
    import warnings
    from ase import Atoms

    if cell is None:
        return [], []

    kpoints = np.asarray(kpoints)
    cell = np.asarray(cell)
    # One atom at the origin is enough for seekpath to identify the
    # (primitive) Bravais lattice and its special points.
    atoms = Atoms(cell=cell, scaled_positions=[[0.0, 0.0, 0.0]], numbers=[1], pbc=True)
    import contextlib
    import io
    import seekpath

    # seekpath's hpkot (via spglib's dict interface and typing_extensions'
    # ``deprecated`` decorator) emits a harmless DeprecationWarning
    # ("dict interface is deprecated"). It cannot be suppressed reliably
    # through the warnings filters — spglib re-registers its own filter at
    # import time — so swallow stderr for the duration of the call.
    err_buffer = io.StringIO()
    with contextlib.redirect_stderr(err_buffer):
        kpath = seekpath.get_path(
            (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
            with_time_reversal=True,
            symprec=symprec,
        )

    point_coords = kpath["point_coords"]
    vertices: list = []
    for start, end in kpath["path"]:
        if not vertices or vertices[-1][0] != start:
            vertices.append((start, np.asarray(point_coords[start], dtype=float)))
        vertices.append((end, np.asarray(point_coords[end], dtype=float)))

    labels: list[str] = []
    label_numbers: list[int] = []
    search_from = 0
    for label, coord in vertices:
        found = None
        for idx in range(search_from, len(kpoints)):
            if np.allclose(kpoints[idx], coord, atol=1e-6):
                found = idx
                break
        if found is None:
            return [], []  # not a seekpath path — leave ticks default
        labels.append(label)
        label_numbers.append(found)
        search_from = found + 1
    return labels, label_numbers


def _scf_fermi_offset_ev(para: dict) -> float:
    """Return ``(E_F_scf - E_F_run)`` in eV for one ``output_banddos_wc_para``.

    aiida-fleur stores the band / DOS energies already shifted by the
    band / DOS run's own Fermi energy (the masci-tools recipe applies
    ``- fermi_energy``). Per fleur.md §4.3 the E_F of a path run can be
    unreliable — re-referencing to the SCF E_F shifts the plotted data by
    this offset. Returns 0.0 when either value is missing.
    """
    from aiida_uranium_workflow.utils.plot._constants import HA_TO_EV

    scf = para.get("fermi_energy_scf")
    run = para.get("fermi_energy_band")
    if scf is None or run is None:
        return 0.0
    return (float(scf) - float(run)) * HA_TO_EV


def extract_fleur(wc) -> Dict[str, Any]:
    """Extract band + DOS from a ``FleurBandAndDosWorkChain`` (or wc)."""
    bands = wc.outputs.band_structure
    try:
        dos = wc.outputs.dos
    except Exception:
        # Band-only runs carry no DOS output; the band data must still
        # be plottable on its own.
        dos = None

    # FLEUR data produced by aiida-fleur's banddos WorkChain (via the
    # masci-tools FleurSimpleBands / FleurDOS recipes) is stored in
    # **eV already shifted so that E_F = 0**: the recipes apply
    # ``shift_by_attribute('fermi_energy', negative=True)`` followed by
    # ``multiply_scalar(HTR_TO_EV)`` to the Hartree eigenvalues, and
    # FLEUR's DOS energy grid is itself (E - E_F). Do NOT multiply by
    # HA_TO_EV again, and use fermi_energy = 0 so the renderer's
    # ``zero_to_efermi`` shift is a no-op.
    energies = np.asarray(bands.get_array("bands"))
    kpoints = np.asarray(bands.get_array("kpoints"))
    if energies.ndim == 3 and energies.shape[0] == 1:
        energies = energies.squeeze(0).T
    elif energies.ndim == 2 and energies.shape[0] == len(kpoints):
        energies = energies.T

    labels = list(bands.base.attributes.get("labels") or [])
    label_numbers = list(bands.base.attributes.get("label_numbers") or [])
    if not labels:
        # aiida-fleur does not store the seekpath path labels on the
        # BandsData; rebuild them from the k-point coordinates.
        labels, label_numbers = _kpath_labels_from_kpoints(
            kpoints, bands.base.attributes.get("cell")
        )

    out_para = wc.outputs.output_parameters.get_dict()
    band_offset = _scf_fermi_offset_ev(out_para.get("band", {}))
    dos_offset = _scf_fermi_offset_ev(out_para.get("dos", {}))

    band_data = BandData(
        energies=energies,
        kpoints=kpoints,
        labels=labels,
        label_numbers=label_numbers,
        # Stored energies are relative to the band run's E_F; offset
        # re-references them to the SCF E_F (fleur.md §4.3).
        fermi_energy=band_offset,
        workchain_pk=wc.pk,
        weights=_read_optional_array(bands, "weights"),
    )

    # FLEUR DOS — XyData: the x channel is stored as ``x_array`` and the
    # y channels as ``y_array_0..N`` with human-readable names in the
    # ``y_names`` attribute (e.g. ``Total_up`` for the spin-up total).
    # There is no ``dos_tot`` array.
    if dos is None:
        dos_data = None
    else:
        energy = np.asarray(dos.get_array("x_array"))
        y_names = list(dos.base.attributes.get("y_names") or [])
        tdos = None
        # Recipe-path SOC runs split the total DOS into spin channels
        # (Total_up + Total_down); a single "Total" channel only exists
        # for legacy fallback output. Sum the spin channels when both
        # are present.
        if "Total_up" in y_names and "Total_down" in y_names:
            idx_up = y_names.index("Total_up")
            idx_dn = y_names.index("Total_down")
            tdos = np.asarray(dos.get_array(f"y_array_{idx_up}")) + np.asarray(
                dos.get_array(f"y_array_{idx_dn}"))
        else:
            # Single spin channel (non-SOC recipe): only ``Total_up``
            # exists (no ``Total_down``); legacy fallback output uses
            # ``dos_tot``.
            for preferred in ("Total", "Total_up", "dos_tot", "dos_total"):
                if preferred in y_names:
                    idx = y_names.index(preferred)
                    tdos = np.asarray(dos.get_array(f"y_array_{idx}"))
                    break
        if tdos is None and y_names:
            # No recognizable total channel — fall back to the first one.
            tdos = np.asarray(dos.get_array("y_array_0"))
        if tdos is None:
            tdos = np.zeros_like(energy)

        spin: Dict[str, np.ndarray] = {}
        for name in ("dos_mx", "dos_my", "dos_mz"):
            try:
                spin[name] = np.asarray(dos.get_array(name))
            except Exception:
                continue

        dos_data = DosData(
            energy=energy,
            total=tdos,
            spin=spin,
            fermi_energy=dos_offset,  # re-reference to the SCF E_F (see above)
            workchain_pk=wc.pk,
        )
    return {"band": band_data, "dos": dos_data}


_L_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}


def extract_abacus_pdos(wc) -> PdosData:
    """Extract projected DOS from an ``AbacusBandWorkChain``.

    Reads ``outputs.dos_projected`` (XyData / ArrayData) which carries
    ``array|energy`` (eV), ``array|orbital_pdos`` with shape
    ``(norbitals, nenergy, nspin)`` and the ``orbital_metadata``
    attribute (one dict per orbital with an ``l`` key). Orbitals are
    summed across all atoms and both spin channels, grouped by angular
    momentum ``l`` (0=s, 1=p, 2=d, 3=f).

    The ``dos_projected`` output is only produced by the ABACUS band
    WorkChain when ``band_settings['run_proj_dos']`` is True.
    """
    pdos = wc.outputs.dos_projected
    energy = np.asarray(pdos.get_array("energy"))
    pdos_arr = np.asarray(pdos.get_array("orbital_pdos"))
    metadata = pdos.base.attributes.get("orbital_metadata") or []
    fermi_ev = float(pdos.base.attributes.get("fermi_level") or 0.0)

    orbitals: Dict[str, np.ndarray] = {}
    for idx, entry in enumerate(metadata):
        l_int = int(entry["l"])
        label = _L_LABELS.get(l_int, f"l{l_int}")
        if label not in orbitals:
            orbitals[label] = np.zeros(pdos_arr.shape[1])
        orbitals[label] += pdos_arr[idx].sum(axis=1)

    total = pdos_arr.sum(axis=(0, 2))
    return PdosData(
        energy=energy,
        orbitals=orbitals,
        total=total,
        fermi_energy=fermi_ev,
        workchain_pk=wc.pk,
    )


def extract_fleur_pdos(wc) -> PdosData:
    """Extract projected DOS from a ``FleurBandAndDosWorkChain``'s ``dos``.

    The FLEUR DOS run (``output/@dos=T``, see fleur.md §4.4) writes
    LOCAL.1/LOCAL.2 files whose columns are energy + projected DOS per
    MT sphere and angular momentum. aiida-fleur's FleurDOS recipe stores
    these in the ``dos`` XyData with ``y_names`` such as ``MT:1s_up`` /
    ``MT:1p_up`` / ``MT:1d_up`` / ``MT:1f_up`` / ``INT_up`` / ``Sym_up`` /
    ``Total_up`` (1/eV; the energy axis is already relative to E_F).

    The per-l channels of all atom types are summed into ``s`` / ``p`` /
    ``d`` / ``f`` curves (no extra FLEUR calculation is needed).
    """
    import re

    dos = wc.outputs.dos
    energy = np.asarray(dos.get_array("x_array"))
    y_names = list(dos.base.attributes.get("y_names") or [])

    def _y(name):
        return np.asarray(dos.get_array(f"y_array_{y_names.index(name)}"))

    orbitals: Dict[str, np.ndarray] = {}
    for l_ch, l_name in (("s", "s"), ("p", "p"), ("d", "d"), ("f", "f")):
        channels = [n for n in y_names if re.match(rf"^MT:\d+{l_ch}_", n)]
        if channels:
            orbitals[l_name] = sum(_y(n) for n in channels)

    total = None
    # Recipe-path SOC runs store the total DOS split into spin channels
    # (Total_up + Total_down); a single "Total" channel only exists for
    # legacy fallback output. Sum the spin channels when both are present.
    if "Total_up" in y_names and "Total_down" in y_names:
        total = _y("Total_up") + _y("Total_down")
    else:
        # Single spin channel (non-SOC recipe): only ``Total_up`` exists;
        # fall back to legacy names for old fallback output.
        for name in ("Total", "Total_up", "dos_total", "dos_tot"):
            if name in y_names:
                total = _y(name)
                break
    if total is None:
        total = sum(orbitals.values()) if orbitals else np.zeros_like(energy)

    return PdosData(
        energy=energy,
        orbitals=orbitals,
        total=total,
        fermi_energy=0.0,  # FLEUR DOS energy axis is already relative to E_F
        workchain_pk=wc.pk,
    )


# PDOS dispatch — by WorkChain process class name.
PDOS_EXTRACTORS: Dict[str, Callable[..., PdosData]] = {
    "AbacusBandWorkChain": extract_abacus_pdos,
    "FleurBandAndDosWorkChain": extract_fleur_pdos,
}


def extract_pdos(pk_or_uuid) -> PdosData:
    """Top-level entry point for projected-DOS extraction.

    Raises ``ValueError`` when no PDOS extractor is registered for
    ``workchain.process_class``, or when the WorkChain has no
    ``dos_projected`` output (i.e. it was run without
    ``band_settings['run_proj_dos']``).
    """
    _ensure_profile_loaded()
    wc = _resolve_pk(pk_or_uuid)
    class_name = wc.process_class.__name__
    extractor = PDOS_EXTRACTORS.get(class_name)
    if extractor is None:
        raise ValueError(
            f"No PDOS extractor registered for WorkChain {class_name!r} "
            f"(pk={wc.pk}). Supported: {sorted(PDOS_EXTRACTORS)}"
        )
    return extractor(wc)


# Registry — dispatch by WorkChain process class name.
EXTRACTORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "AbacusBandWorkChain":          extract_abacus,
    "FleurBandAndDosWorkChain":     extract_fleur,
    # Future: "FleurBandDosWorkChain": extract_fleur_legacy,
}


def extract_band_dos(pk_or_uuid) -> Dict[str, Any]:
    """Top-level entry point.

    Returns
    -------
    dict
        ``{"band": BandData, "dos": DosData}`` (and any future keys
        like ``"pdos"`` the backend adds). Raises ``ValueError`` when
        no extractor is registered for ``workchain.process_class``.
    """
    _ensure_profile_loaded()
    wc = _resolve_pk(pk_or_uuid)
    class_name = wc.process_class.__name__
    extractor = EXTRACTORS.get(class_name)
    if extractor is None:
        raise ValueError(
            f"No band/dos extractor registered for WorkChain "
            f"{class_name!r} (pk={wc.pk}). "
            f"Supported: {sorted(EXTRACTORS)}"
        )
    return extractor(wc)


__all__ = [
    "BandData",
    "DosData",
    "PdosData",
    "EXTRACTORS",
    "PDOS_EXTRACTORS",
    "extract_band_dos",
    "extract_pdos",
    "extract_abacus",
    "extract_fleur",
    "extract_abacus_pdos",
    "extract_fleur_pdos",
    "_scf_fermi_offset_ev",
]