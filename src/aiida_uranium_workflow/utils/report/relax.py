"""Relax WorkChain report generation module.

Renders the relaxed lattice constants, volume and energy for the
**plugin** relax WorkChains (``abacus.relax`` / ``vasp.relax`` /
``fleur.relax``).

Two data sources:

* ``workchain_node`` (preferred) — the report CLI passes the finished
  plugin WorkChain node; lattice / volume / energy are derived from its
  raw outputs:
    - ABACUS: ``outputs.structure`` + ``outputs.misc['total_energy']`` (eV)
    - VASP:   ``outputs.structure`` (the final static calculation's
      geometry) + ``outputs.misc['total_energies']['energy_extrapolated']``
      (eV, last ionic step); the relax mode is read back from the
      ``relax_settings`` input (ISIF)
    - FLEUR:  ``outputs.optimized_structure`` +
      ``outputs.output_relax_wc_para['last_energy']`` (Hartree → eV)
* ``output_params`` (legacy) — the combined ``output_parameters`` dict
  produced by the former in-repo wrapper WorkChains (kept for reports
  against nodes created before the refactor).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ._common import (
    render_report_footer,
    render_report_header,
)

#: eV per Hartree (FLEUR energies are reported in Hartree).
HA_TO_EV = 27.211386245988


def _fmt(value: Any, prec: int = 6) -> str:
    """Render a scalar (or list of scalars) with fixed precision."""
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, prec) for v in value) + "]"
    try:
        return f"{float(value):.{prec}f}"
    except (TypeError, ValueError):
        return str(value)


#: ISIF note per (positions, shape, volume) degree-of-freedom combination —
#: the same table ``RelaxModeEnum`` uses inside aiida-vasp.
_VASP_MODES = {
    (True, False, False): "positions-only relax (ISIF=2)",
    (True, True, True): "full relax (positions + cell, ISIF=3)",
    (True, True, False): "positions + shape relax (ISIF=4)",
    (False, True, False): "shape-only relax (ISIF=5)",
    (False, True, True): "cell relax, positions fixed (ISIF=6)",
    (False, False, True): "volume-only relax (ISIF=7)",
}


def _port(ports, label: str) -> Any:
    """Fetch ``ports[label]``, unwrapping nested namespaces.

    AiiDA exposes the plugin's ``relax.structure`` output as the namespace
    ``relax`` holding ``structure`` (``node.outputs.relax.structure``), so a
    dotted label has to be traversed part by part. Looking up the literal
    ``"relax__structure"`` instead would hit AiiDA's deprecated
    double-underscore dereferencing (``AiidaDeprecationWarning``).
    """
    # 1. a flat link label (``structure`` / ``misc`` / ``relax_settings``)
    try:
        return ports[label]
    except (AttributeError, KeyError, TypeError):
        pass

    # 2. nested namespace: ``a.b`` -> ports['a']['b']
    parts = label.split(".")
    if len(parts) > 1:
        value = ports
        try:
            for part in parts:
                value = value[part]
        except (AttributeError, KeyError, TypeError):
            value = None
        if value is not None:
            return value

    # 3. attribute access (single, undotted label)
    try:
        return getattr(ports, label)
    except (AttributeError, KeyError, TypeError):
        return None


def _output(node, label: str) -> Any:
    """Fetch an output node by label, tolerating missing outputs/links."""
    try:
        outputs = node.outputs
    except (AttributeError, KeyError):
        return None
    return _port(outputs, label)


def _input(node, label: str) -> Any:
    """Fetch an input node by label, tolerating missing ports."""
    try:
        inputs = node.inputs
    except (AttributeError, KeyError):
        return None
    return _port(inputs, label)


def _last_number(value: Any) -> Any:
    """Return the last entry of a per-ionic-step value as a float."""
    if isinstance(value, (list, tuple)) or hasattr(value, "__len__"):
        try:
            if len(value) == 0:
                return None
            return float(value[-1])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vasp_total_energy(misc) -> Any:
    """Extract the final total energy (eV) from a VASP ``misc`` Dict.

    aiida-vasp stores ``total_energies`` as ``{energy_type: per-step list}``
    (``energy_extrapolated`` by default, see
    ``parsers.content_parsers.vasprun.ENERGY_MAPPING``); the last ionic step
    is the relaxed geometry.
    """
    if misc is None:
        return None
    try:
        misc_dict = misc.get_dict()
    except (AttributeError, KeyError):
        return None
    energies = misc_dict.get("total_energies") or {}
    if isinstance(energies, dict):
        for key in ("energy_extrapolated", "energy_no_entropy", "energy_free"):
            if key in energies:
                return _last_number(energies[key])
        for value in energies.values():
            energy = _last_number(value)
            if energy is not None:
                return energy
    return _last_number(energies)


def _vasp_mode(relax_settings) -> Any:
    """Relax-mode label from the VaspRelaxWorkChain ``relax_settings`` input."""
    if relax_settings is None:
        return None
    if isinstance(relax_settings, dict):
        settings = relax_settings
    else:
        try:
            settings = relax_settings.get_dict()
        except (AttributeError, KeyError):
            return None
    if not isinstance(settings, dict):
        return None
    if settings.get("perform") is False:
        return "no relaxation (static calculation only)"
    dof = tuple(
        bool(settings.get(key, False)) for key in ("positions", "shape", "volume")
    )
    return _VASP_MODES.get(dof)


def _bond_angle(u, v) -> float:
    """Angle between two lattice vectors, in degrees."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _derive_from_node(node, workflow_type: str) -> Dict[str, Any]:
    """Derive lattice constants / volume / energy from a plugin relax node."""
    mode = None
    if workflow_type == "abacus":
        structure = node.outputs.structure
        try:
            energy = node.outputs.misc.get_dict().get("total_energy")
        except (AttributeError, KeyError):
            energy = None
    elif workflow_type == "qe":
        # PwRelaxWorkChain: output_structure (relaxed) + the last SCF's
        # output_parameters (``energy`` in eV — aiida-qe already
        # converted from Hartree).
        structure = None
        energy = None
        mode = None
        try:
            structure = node.outputs.output_structure
        except (AttributeError, KeyError):
            pass
        try:
            para = node.outputs.output_parameters.get_dict()
            energy = para.get("energy")
        except (AttributeError, KeyError):
            pass
        # The report mode reflects the *actual* pw calculation type of
        # the relax children (vc-relax changes the cell; a plain relax
        # keeps it fixed) instead of a hard-coded label.
        try:
            calc = (
                node.inputs.base_relax.pw.parameters.get_dict()
                .get("CONTROL", {})
                .get("calculation", "relax")
            )
            mode = {
                "vc-relax": "cell+positions relax (vc-relax)",
                "relax": "positions relax (cell fixed)",
                "vc-md": "cell+positions dynamics (vc-md)",
                "md": "positions dynamics (md)",
            }.get(calc, f"{calc} relax")
        except (AttributeError, KeyError):
            pass
        if structure is None:
            return {
                "lattice_constants": None,
                "volume": None,
                "energy": energy,
                "mode": mode,
                "energy_units": "eV",
            }
    elif workflow_type == "vasp":
        # VaspRelaxWorkChain exposes the called VaspWorkChain's outputs
        # (``misc``, and ``structure`` when the last run parsed one) and
        # attaches the relaxed geometry as ``relax.structure`` — a nested
        # output, i.e. ``node.outputs.relax.structure``. ``_port`` unwraps
        # the dotted label; the deprecated ``relax__structure`` spelling is
        # deliberately not used (AiidaDeprecationWarning since aiida-core 2).
        structure = None
        for label in ("relax.structure", "structure"):
            structure = _output(node, label)
            if structure is not None:
                break
        energy = _vasp_total_energy(_output(node, "misc"))
        mode = _vasp_mode(_input(node, "relax_settings"))
        if structure is None:
            return {
                "lattice_constants": None,
                "volume": None,
                "energy": energy,
                "mode": mode,
                "energy_units": "eV",
            }
    else:  # fleur
        structure = node.outputs.optimized_structure
        energy = None
        try:
            para = node.outputs.output_relax_wc_para.get_dict()
            # ``last_energy`` is the final total energy in **eV**
            # (aiida-fleur 2.0.0 already converted from Hartree; e.g.
            # bcc-U 2 atoms ≈ -1 528 354 eV, matching the EOS report).
            # The parallel ``energy`` field is the raw value / 27.2114
            # and must not be used.
            e_raw = para.get("last_energy")
            if isinstance(e_raw, (list, tuple)):
                e_raw = e_raw[0] if e_raw else None
            if e_raw is not None:
                energy = float(e_raw)  # already eV
        except (AttributeError, KeyError):
            pass

    cell = np.asarray(structure.cell)
    lengths = [float(x) for x in np.linalg.norm(cell, axis=1)]
    angles = [
        _bond_angle(cell[1], cell[2]),
        _bond_angle(cell[0], cell[2]),
        _bond_angle(cell[0], cell[1]),
    ]
    result = {
        "lattice_constants": lengths,
        "lattice_constant": lengths[0],
        "lattice_angles": angles,
        "volume": float(structure.get_cell_volume()),
        "energy": energy,
        "energy_units": "eV",
        "sites": fractional_coordinates(structure),
    }
    if mode is not None:
        result["mode"] = mode
    return result


def fractional_coordinates(structure) -> List[Dict[str, Any]]:
    """Per-site fractional coordinates of a (relaxed) structure.

    Crystallography calls the free atomic coordinates the *internal
    parameters*: they are the fractional coordinates modulo the lattice
    translations. They are reported verbatim — the free coordinates of a
    fixed Wyckoff orbit are visible by comparing the rows.

    Returns one dict per site: ``{"index", "element", "frac", "cartesian"}``
    with ``frac`` / ``cartesian`` as lists of three floats. Returns an empty
    list when the structure is unavailable or carries no sites.
    """
    if structure is None:
        return []
    try:
        cell = np.asarray(structure.cell, dtype=float)
        sites = list(structure.sites)
    except (AttributeError, TypeError):
        return []
    try:
        frac = np.asarray(
            [site.frac_coords for site in sites], dtype=float
        )
    except (AttributeError, TypeError):
        frac = np.asarray(
            [np.linalg.solve(cell.T, np.asarray(site.position, dtype=float)) for site in sites],
            dtype=float,
        )
    rows: List[Dict[str, Any]] = []
    for index, site in enumerate(sites):
        rows.append(
            {
                "index": index,
                "element": getattr(site, "kind_name", None) or getattr(site, "symbol", "?"),
                "frac": [float(x) for x in frac[index]],
                "cartesian": [float(x) for x in np.asarray(site.position, dtype=float)],
            }
        )
    return rows


def generate_internal_parameters_table(output_params: Dict[str, Any]) -> str:
    """Render the relaxed atomic coordinates (fractional = internal params)."""
    sites = output_params.get("sites") or []
    if not sites:
        return ""

    lines = [
        "## Internal Parameters (relaxed atomic coordinates)",
        "",
        "| site | element | frac x | frac y | frac z | cart x (Å) | cart y (Å) | cart z (Å) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sites:
        frac = row.get("frac") or [None, None, None]
        cart = row.get("cartesian") or [None, None, None]
        lines.append(
            f"| {row.get('index')} | {row.get('element')} | "
            f"{_fmt(frac[0])} | {_fmt(frac[1])} | {_fmt(frac[2])} | "
            f"{_fmt(cart[0])} | {_fmt(cart[1])} | {_fmt(cart[2])} |"
        )
    return "\n".join(lines)


def generate_summary_table(output_params: Dict[str, Any], workflow_type: str) -> str:
    """Render the relaxed lattice / volume / energy summary."""
    lines = ["| Property | Value |", "| --- | --- |"]
    lines.append(f"| backend | {workflow_type} |")

    mode = output_params.get("mode")
    if mode is None:
        if workflow_type == "fleur":
            mode = "full relax (positions + cell)"
        elif workflow_type == "qe":
            mode = "relax"
        else:
            mode = "volume-only relax"
    lines.append(f"| mode | {mode} |")
    lines.append(
        f"| relaxed lattice constants | "
        f"{_fmt(output_params.get('lattice_constants'), 6)} Å |"
    )
    angles = output_params.get("lattice_angles")
    if angles:
        lines.append(
            f"| relaxed lattice angles α, β, γ | {_fmt(angles, 4)} ° |"
        )
    lines.append(f"| relaxed volume | {_fmt(output_params.get('volume'))} Å³ |")
    lines.append(f"| relaxed energy | {_fmt(output_params.get('energy'))} eV |")
    return "\n".join(lines)


def generate_eos_scan_table(output_params: Dict[str, Any]) -> str:
    """Render the EOS volume-scan table (scaling / volume / energy)."""
    scaling = output_params.get("eos_scaling") or []
    volumes = output_params.get("eos_volumes") or []
    energies = output_params.get("eos_energies") or []

    if not volumes:
        return "No EOS scan data available."

    lines = ["| scale | volume (Å³) | energy (eV) |", "| --- | --- | --- |"]
    for i, vol in enumerate(volumes):
        scale = scaling[i] if i < len(scaling) else None
        energy = energies[i] if i < len(energies) else None
        lines.append(f"| {_fmt(scale, 4)} | {_fmt(vol, 4)} | {_fmt(energy, 6)} |")
    return "\n".join(lines)


def generate_report(
    output_params: Dict[str, Any],
    pk: int,
    workflow_type: str,
    workchain_node=None,
) -> str:
    """Generate a complete Markdown report for a relax WorkChain.

    ``workchain_node`` is the finished plugin relax WorkChain node; when
    given, the relaxed lattice / volume / energy are derived from its
    outputs instead of ``output_params``.
    """
    if workchain_node is not None:
        data = _derive_from_node(workchain_node, workflow_type)
        output_params = dict(output_params)
        output_params.update(data)

    lines: List[str] = [
        render_report_header(
            title="Relax WorkChain Report",
            workflow_type=workflow_type,
            pk=pk,
        ),
        "",
        "## Relaxed Structure",
        "",
        generate_summary_table(output_params, workflow_type),
        "",
    ]

    internal = generate_internal_parameters_table(output_params)
    if internal:
        lines += [internal, ""]

    lines += [render_report_footer()]
    return "\n".join(lines)
