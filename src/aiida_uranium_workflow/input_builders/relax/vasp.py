"""VASP input builder for the relax workflow.

Thin wrapper around the **plugin** ``vasp.relax`` WorkChain
(:class:`aiida_vasp.workchains.v2.relax.VaspRelaxWorkChain`) — no custom
WorkChain is needed, mirroring how the abacus / fleur / qe relax adapters
call their plugin relax WorkChains directly.

The SCF base (INCAR / ENCUT / smearing / POTCAR family / k-points) comes
from the per-backend preset ``parameters/vasp/scf.yml``; the relaxation
controls come from the ``vasp`` block of the workflow protocol
(``parameters/relax.yml``).

The protocol block keeps the **ABACUS vocabulary** so one preset file
reads the same for every backend; the translation to aiida-vasp's
``RelaxOptions`` happens in :func:`relax_settings_from_protocol`::

    test:
      abacus:
        relax_type: volume
        force_cutoff: 0.05
        ...
      vasp:
        relax_type: volume             # -> positions/shape/volume dof (ISIF=7)
        relax_method: cg               # -> algo (IBRION=2)
        max_ionic_steps: 50            # -> steps (INCAR NSW)
        force_cutoff: 0.05             # eV/Å -> force_cutoff (EDIFFG = -0.05)
        meta_convergence: true         # -> convergence_on
        volume_convergence: 0.01       # relative |ΔV|/V0 -> absolute Å³
        max_meta_convergence_iterations: 5   # -> convergence_max_iterations
        relax_settings: {}             # raw RelaxOptions, overrides the above

``relax_type`` -> INCAR ``ISIF`` (full table, VASP's own ISIF semantics;
``full`` is accepted as an alias of ``positions_cell``)::

    relax_type        positions  shape  volume   ISIF
    none                  -        -      -      (no relax: static only)
    positions            yes       no     no      2
    volume                no       no    yes      7   <- 1-D Brent optimizer
    shape                 no      yes     no      5
    cell                  no      yes    yes      6
    positions_shape      yes      yes     no      4
    positions_cell       yes      yes    yes      3   <- 全弛豫 (cell + ions)
    positions_volume     yes       no    yes      ✗  VASP has no such ISIF

ISIF=7 and 5/6 relax *only* the cell with VASP's 1-D (Brent) line
minimisation; on a noisy E(V) curve (metals, coarse k-mesh) that aborts
with ``ZBRENT: fatal error in bracketing``. ``positions_cell`` (ISIF=3)
lets the CG/RMM-DIIS ionic optimizer move everything instead, which is the
robust choice for a cell relaxation — for a cubic cell the shape is kept
by symmetry, so the converged volume is the same "volume-only" cell.

Two VASP-specific points are handled here rather than in the YAML:

* ``relax_type`` is **required** — aiida-vasp's default RelaxOptions is a
  full relaxation (positions + shape + volume), which for a ``relax``
  preset meant as a volume-only relaxation would silently change the
  cell shape and the atomic positions;
* the INCAR tags the plugin derives from ``relax_settings`` (NSW / IBRION
  / ISIF / EDIFFG) are **stripped** from the SCF preset. An SCF preset
  pins ``nsw: 0``, and aiida-vasp applies the ``incar`` namespace *after*
  the relax set-functions, so a leftover ``nsw: 0`` would turn the
  relaxation into a static calculation.
"""

from __future__ import annotations

from ..base import filter_scheduler_options, SoftwareAdapter
from typing import Any, Dict, Optional

import copy

#: ABACUS ``RelaxType`` value -> the (positions, shape, volume) degrees of
#: freedom VASP's ISIF can express. The keys are aiida-abacus's
#: ``RelaxType`` strings so ``parameters/relax.yml`` keeps one vocabulary
#: across backends; the values are exactly the combinations of
#: :class:`aiida_vasp.assistant.parameters.RelaxModeEnum` (2/3/4/5/6/7).
RELAX_TYPES: Dict[str, Dict[str, bool]] = {
    "positions": {"positions": True, "shape": False, "volume": False},  # ISIF=2
    "volume": {"positions": False, "shape": False, "volume": True},  # ISIF=7
    "shape": {"positions": False, "shape": True, "volume": False},  # ISIF=5
    "cell": {"positions": False, "shape": True, "volume": True},  # ISIF=6
    "positions_shape": {  # ISIF=4
        "positions": True,
        "shape": True,
        "volume": False,
    },
    "positions_cell": {  # ISIF=3
        "positions": True,
        "shape": True,
        "volume": True,
    },
}

#: ABACUS relax types with **no** VASP counterpart: relaxing the atomic
#: positions and the volume while keeping the cell shape fixed has no ISIF
#: (ISIF only knows 2/3/4/5/6/7).
UNSUPPORTED_RELAX_TYPES = ("positions_volume",)

#: Spelled-out aliases: ``full`` is what the ``qe`` block of the same
#: protocol file uses for ``vc-relax`` (all cell + ionic dof), so accept it
#: here too — it is exactly :data:`RELAX_TYPES["positions_cell"]` (ISIF=3).
RELAX_TYPE_ALIASES: Dict[str, str] = {"full": "positions_cell"}

#: Ionic algorithm: VASP only implements IBRION=2 (CG) and IBRION=1
#: (RMM-DIIS). ABACUS's ``cg_bfgs`` / ``fire`` have no VASP equivalent.
RELAX_METHOD_TO_ALGO: Dict[str, str] = {"cg": "cg", "rd": "rd"}

#: INCAR tags the plugin derives from ``relax_settings`` in
#: ``ParametersMassage``. They are removed from the SCF preset because the
#: ``incar`` namespace is applied last and would win: ``nsw: 0`` (the SCF
#: presets carry it) would make the "relaxation" a static calculation.
RELAX_CONTROLLED_INCAR_TAGS = ("nsw", "ibrion", "isif", "ediffg")

#: Keys accepted in the protocol's ``vasp`` block. Anything else is
#: rejected (a silently ignored typo would change the physics).
PROTOCOL_KEYS = frozenset(
    {
        "relax_type",
        "relax_method",
        "max_ionic_steps",
        "force_cutoff",
        "energy_cutoff",
        "stress_cutoff",
        "meta_convergence",
        "volume_convergence",
        "max_meta_convergence_iterations",
        "convergence_positions",
        "convergence_shape_lengths",
        "convergence_shape_angles",
        "relax_settings",
    }
)


def _relax_options_fields() -> Optional[frozenset]:
    """Keys of aiida-vasp's ``RelaxOptions`` (``None`` if not importable)."""
    try:
        from aiida_vasp.utils.opthold import RelaxOptions
    except ImportError:  # pragma: no cover - aiida-vasp is a hard dependency
        return None
    return frozenset(RelaxOptions.model_fields)


def _relax_options_defaults() -> Dict[str, Any]:
    """Every ``RelaxOptions`` field at its default value.

    ``VaspRelaxWorkChain.initialize`` copies the ``relax_settings`` dict into
    an ``AttributeDict`` and then reads several fields as plain attributes
    (``self.ctx.relax_settings.perform`` in ``perform_relaxation``), so a
    partial dict — e.g. only the positions/shape/volume flags the protocol
    derives — excepts at start-up with::

        AttributeError: 'AttributeDict' object has no attribute 'perform'

    Filling the defaults keeps the workchain's defaults *and* makes the dict
    complete; explicit protocol values still win.
    """
    try:
        from aiida_vasp.utils.opthold import RelaxOptions
    except ImportError:  # pragma: no cover - aiida-vasp is a hard dependency
        return {}
    return dict(RelaxOptions().model_dump())


def _check_relax_options(settings: Dict[str, Any]) -> None:
    """Reject keys aiida-vasp's ``RelaxOptions`` does not know.

    pydantic ignores unknown fields, so a mistyped native key (e.g.
    ``convergence_volum``) would otherwise be dropped without a word.
    """
    known = _relax_options_fields()
    if known is None:
        return
    unknown = sorted(set(settings) - known)
    if unknown:
        raise ValueError(
            f"relax protocol (vasp): unknown relax_settings key(s) {unknown}; "
            f"VaspRelaxWorkChain accepts {', '.join(sorted(known))}"
        )


def relax_settings_from_protocol(
    block: Dict[str, Any], structure_volume: Optional[float] = None
) -> Dict[str, Any]:
    """Translate the protocol's ``vasp`` block into ``RelaxOptions`` keys.

    Pure function (no AiiDA), so it is unit-testable without a profile.
    ``structure_volume`` (Å³) is only needed to convert the **relative**
    ``volume_convergence`` (ABACUS semantics: ``|ΔV| / V``) into
    aiida-vasp's **absolute** ``convergence_volume`` (Å³ between the input
    and output structures of one relax run)::

        convergence_volume = volume_convergence * V_initial
    """
    block = dict(block or {})
    unknown = sorted(set(block) - PROTOCOL_KEYS)
    if unknown:
        raise ValueError(
            f"relax protocol (vasp): unknown key(s) {unknown}; the vasp "
            f"block accepts {', '.join(sorted(PROTOCOL_KEYS))}"
        )

    raw = block.get("relax_settings") or {}
    if not isinstance(raw, dict):
        raise ValueError(
            "relax protocol (vasp): 'relax_settings' must be a mapping of "
            f"VaspRelaxWorkChain RelaxOptions, got {type(raw).__name__}"
        )
    raw = dict(raw)

    relax_type = block.get("relax_type")
    if relax_type is None:
        raise ValueError(
            "relax protocol (vasp): 'relax_type' is required — set it to one "
            f"of {', '.join(sorted(RELAX_TYPES))} (or 'none'). aiida-vasp's "
            "default is a FULL relaxation (positions + shape + volume), so "
            "leaving it out would silently relax more than the preset asks."
        )
    relax_type = RELAX_TYPE_ALIASES.get(str(relax_type), str(relax_type))

    settings: Dict[str, Any] = {}
    if relax_type == "none":
        # No relaxation at all: the workchain skips the relax loop and runs
        # the final static calculation (ABACUS RelaxType.NONE counterpart).
        settings["perform"] = False
    elif relax_type in UNSUPPORTED_RELAX_TYPES:
        raise ValueError(
            f"relax protocol (vasp): relax_type {relax_type!r} has no VASP "
            "counterpart — VASP's ISIF cannot relax the positions and the "
            "volume while keeping the cell shape fixed. Use 'positions', "
            "'volume', 'shape', 'cell', 'positions_shape', 'positions_cell' "
            "(or 'full' = the same full relaxation)."
        )
    else:
        try:
            settings.update(RELAX_TYPES[relax_type])
        except KeyError:
            raise ValueError(
                f"relax protocol (vasp): relax_type must be one of "
                f"{', '.join(sorted(RELAX_TYPES))} (or 'none'/'full'), got "
                f"{relax_type!r}"
            ) from None

    if "relax_method" in block:
        method = str(block["relax_method"])
        try:
            settings["algo"] = RELAX_METHOD_TO_ALGO[method]
        except KeyError:
            raise ValueError(
                f"relax protocol (vasp): relax_method must be one of "
                f"{', '.join(sorted(RELAX_METHOD_TO_ALGO))} (VASP implements "
                f"IBRION=2 'cg' and IBRION=1 'rd' only), got {method!r}"
            ) from None

    if "max_ionic_steps" in block:
        settings["steps"] = int(block["max_ionic_steps"])

    force_cutoff = block.get("force_cutoff")
    energy_cutoff = block.get("energy_cutoff")
    if force_cutoff is not None and energy_cutoff is not None:
        raise ValueError(
            "relax protocol (vasp): set only one of 'force_cutoff' (eV/Å, "
            "EDIFFG < 0) and 'energy_cutoff' (eV, EDIFFG > 0) — VASP's "
            "EDIFFG takes a single criterion."
        )
    if force_cutoff is not None:
        # Same unit as ABACUS's force_cutoff (eV/Å) -> INCAR EDIFFG = -x.
        settings["force_cutoff"] = float(force_cutoff)
    if energy_cutoff is not None:
        # aiida-vasp's RelaxOptions.force_cutoff is a plain float (default
        # 0.03), never None, and ParametersMassage.set_ediffg raises when both
        # criteria are present — so an energy-only criterion cannot be
        # expressed through vasp.relax. Refuse instead of emitting a run that
        # fails inside the plugin's parameter massager.
        raise ValueError(
            "relax protocol (vasp): 'energy_cutoff' cannot be used — "
            "aiida-vasp's RelaxOptions always carries a force_cutoff (float, "
            "default 0.03) and its parameter massager rejects having both "
            "criteria. Use 'force_cutoff' (eV/Å, INCAR EDIFFG < 0) instead."
        )

    if "stress_cutoff" in block:
        raise ValueError(
            "relax protocol (vasp): 'stress_cutoff' has no VASP counterpart — "
            "VASP stops on the force (EDIFFG < 0) or energy (EDIFFG > 0) "
            "criterion and has no separate stress threshold. Drop the key "
            "(the ABACUS preset keeps it; it is a kBar criterion of "
            "abacus.relax)."
        )

    if "meta_convergence" in block:
        settings["convergence_on"] = bool(block["meta_convergence"])
    if "max_meta_convergence_iterations" in block:
        settings["convergence_max_iterations"] = int(
            block["max_meta_convergence_iterations"]
        )

    if "volume_convergence" in block:
        relative = float(block["volume_convergence"])
        if not structure_volume:
            raise ValueError(
                "relax protocol (vasp): 'volume_convergence' is a RELATIVE "
                "threshold (|ΔV|/V, ABACUS semantics) that is converted to "
                "aiida-vasp's absolute 'convergence_volume' (Å³) with the "
                "input cell volume — the structure volume is unavailable."
            )
        settings["convergence_volume"] = relative * float(structure_volume)

    for key in (
        "convergence_positions",
        "convergence_shape_lengths",
        "convergence_shape_angles",
    ):
        if key in block:
            settings[key] = float(block[key])

    _check_relax_options(raw)
    merged = {**_relax_options_defaults(), **settings, **raw}
    _check_relax_options(merged)
    return merged


class VaspRelaxAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspRelaxWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.relax"

    def _block(self) -> Dict[str, Any]:
        """The ``vasp`` section of the relax protocol preset."""
        block = self.workflow_data.get("vasp") or {}
        if not isinstance(block, dict):
            raise ValueError(
                "relax protocol: the 'vasp' block must be a mapping "
                f"(got {type(block).__name__})"
            )
        return dict(block)

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        entry = self.software_params
        params = copy.deepcopy(entry.get("parameters", {}))
        incar = params.get("incar") if isinstance(params, dict) else None
        if not isinstance(incar, dict):
            raise ValueError(
                "relax protocol (vasp): the SCF preset must carry "
                "'parameters.incar' (aiida-vasp rejects a parameters dict "
                "without it)"
            )
        # Let relax_settings own NSW / IBRION / ISIF / EDIFFG: the ``incar``
        # namespace is applied after the relax set-functions by
        # ``ParametersMassage``, so a leftover ``nsw: 0`` from the SCF preset
        # would turn the relaxation into a single-point calculation.
        for tag in RELAX_CONTROLLED_INCAR_TAGS:
            incar.pop(tag, None)

        if "potential_family" not in entry or "potential_mapping" not in entry:
            raise ValueError(
                "relax protocol (vasp): the SCF preset must carry "
                "'potential_family' and 'potential_mapping'"
            )

        options = filter_scheduler_options(
            self.metadata.get("options", {}),
            entry_point=self._workchain_entry_point(),
            backend=self.name,
        )

        vasp: Dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "parameters": orm.Dict(dict=params),
            "potential_family": orm.Str(str(entry["potential_family"])),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
        }
        if options:
            # VaspRelaxWorkChain exposes the called VaspWorkChain as the
            # ``vasp`` namespace, whose ``calc`` namespace carries the
            # CalcJob's scheduler options.
            vasp["calc"] = {"metadata": {"options": options}}
        if "kpoints_spacing" in entry:
            vasp["kpoints_spacing"] = orm.Float(float(entry["kpoints_spacing"]))
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            kpoints = DataFactory("core.array.kpoints")()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            vasp["kpoints"] = kpoints

        settings = relax_settings_from_protocol(
            self._block(), structure.get_cell_volume()
        )

        return {
            "structure": structure,
            "vasp": vasp,
            "relax_settings": orm.Dict(dict=settings),
        }

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation, which submits the plugin
    # WorkChain resolved from ``_workchain_entry_point`` ("vasp.relax");
    # the scheduler options are already placed under
    # ``vasp.calc.metadata.options`` by :meth:`_build_workchain_inputs`.
