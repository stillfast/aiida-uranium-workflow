"""VASP input builder for the phonopy workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:class:`aiida_uranium_workflow.workflows.phonopy.vasp.VaspPhonopyWorkChain`.

* the VASP SCF base (INCAR / ENCUT / smearing / POTCAR family /
  k-points) comes from ``parameters/vasp/scf.yml`` (``software_params``).
  The adapter forces a **force-only static run** — ``IBRION=-1``,
  ``NSW=0``, ``ISIF=2`` — so each displaced supercell returns atomic
  forces in ``misc['forces']`` (eV/Å). Those three tags come from the
  ``vasp.incar`` block of ``parameters/phonopy.yml`` when given (they are
  merged last, so a preset value cannot win by accident);
* the phonopy settings (supercell / primitive matrix / band path / DOS)
  come from the ``parameters/phonopy.yml`` protocol block
  (``workflow_data['phonopy']``) — the same keys as the ABACUS / FLEUR
  adapters, which is why the two pure helpers are shared with
  :class:`aiida_uranium_workflow.input_builders.phonopy.abacus.
  AbacusPhonopyAdapter`.
"""

from __future__ import annotations

from ..base import AdaptedInputs, filter_scheduler_options, SoftwareAdapter
from .abacus import AbacusPhonopyAdapter
from typing import Any, Tuple

import copy

#: INCAR tags of a force-only static run: no ionic update, one electronic
#: loop, forces + stress written to OUTCAR / vasprun.xml. This is the VASP
#: counterpart of the ABACUS ``calculation=scf`` + ``cal_force=1`` pair.
FORCE_RUN_INCAR: dict[str, Any] = {"ibrion": -1, "nsw": 0, "isif": 2}


class VaspPhonopyAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the VASP phonon WorkChain inputs."""

    name = "vasp"

    # The band-path / displacement logic is protocol-level and shared with
    # the ABACUS adapter (identical keys in parameters/phonopy.yml).
    build_phonopy_parameters = staticmethod(
        AbacusPhonopyAdapter.build_phonopy_parameters
    )
    build_displacement_generator = staticmethod(
        AbacusPhonopyAdapter.build_displacement_generator
    )

    def _workchain_entry_point(self) -> str:
        return "uranium.phonopy.vasp"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        ph = dict(self.workflow_data.get("phonopy", {}) or {})
        entry = self.software_params

        params = copy.deepcopy(entry.get("parameters", {}))
        incar = params.get("incar") if isinstance(params, dict) else None
        if not isinstance(incar, dict):
            raise ValueError(
                "phonopy protocol (vasp): the SCF preset must carry "
                "'parameters.incar' (aiida-vasp rejects a parameters dict "
                "without it)"
            )
        if "potential_family" not in entry or "potential_mapping" not in entry:
            raise ValueError(
                "phonopy protocol (vasp): the SCF preset must carry "
                "'potential_family' and 'potential_mapping'"
            )

        # Force run: the tags of ``FORCE_RUN_INCAR`` (IBRION / NSW / ISIF)
        # define *what* the run is and are applied last, so neither the SCF
        # preset (``nsw: 0`` is fine, but a stray relax tag would not be)
        # nor the protocol's ``vasp.incar`` block can turn the supercell
        # runs into something that produces no forces. Every other INCAR
        # key of the block is passed through (``ediff`` / ``algo`` /
        # ``nelm`` / ... for force accuracy).
        overrides = dict((ph.get("vasp") or {}).get("incar") or {})
        incar.update(overrides)
        incar.update(FORCE_RUN_INCAR)
        params["incar"] = incar

        options = filter_scheduler_options(
            self.metadata.get("options", {}),
            entry_point=self._workchain_entry_point(),
            backend=self.name,
        )

        base: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "parameters": orm.Dict(dict=params),
            "potential_family": orm.Str(str(entry["potential_family"])),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
        }
        if options:
            base["calc"] = {"metadata": {"options": options}}
        if "kpoints_spacing" in entry:
            base["kpoints_spacing"] = orm.Float(float(entry["kpoints_spacing"]))
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            kpoints = DataFactory("core.array.kpoints")()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            base["kpoints"] = kpoints

        phonopy_code = self.extra_codes.get("phonopy")
        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "phonopy_parameters": orm.Dict(dict=self.build_phonopy_parameters(ph)),
        }
        if phonopy_code:
            inputs["phonopy_code"] = orm.load_code(phonopy_code)

        # phonopy pre-processing knobs (identical keys to the other backends)
        if "supercell_matrix" in ph:
            inputs["supercell_matrix"] = orm.List(list=list(ph["supercell_matrix"]))
        # ``primitive_matrix: null`` means "keep the input cell" — omit the
        # input so the WorkChain / phonopy default (no reduction) applies.
        if ph.get("primitive_matrix") is not None:
            pm = ph["primitive_matrix"]
            if isinstance(pm, str):
                inputs["primitive_matrix"] = orm.Str(pm)
            else:
                inputs["primitive_matrix"] = orm.List(list=list(pm))
        if "symprec" in ph:
            inputs["symprec"] = orm.Float(float(ph["symprec"]))
        if "is_symmetry" in ph:
            inputs["is_symmetry"] = orm.Bool(bool(ph["is_symmetry"]))
        if "distinguish_kinds" in ph:
            inputs["distinguish_kinds"] = orm.Bool(bool(ph["distinguish_kinds"]))
        inputs["displacement_generator"] = orm.Dict(
            dict=self.build_displacement_generator(ph)
        )

        # band labels for the report / plot (manual mode)
        if ph.get("band_labels"):
            inputs["band_labels"] = orm.List(list=list(ph["band_labels"]))

        # phonopy CalcJob scheduler options. The wall-time is inherited from
        # the backend metadata preset (a duration — valid on any machine);
        # the queue name only when the phonopy code runs on the *same*
        # computer as the VASP code (e.g. ``phonopy@cloud`` next to
        # ``vaspstd@yeesuan`` must not inherit ``q_ysuan``).
        ph_options = dict(ph.get("options", {}) or {})
        if (
            "max_wallclock_seconds" not in ph_options
            and options.get("max_wallclock_seconds") is not None
        ):
            ph_options["max_wallclock_seconds"] = options["max_wallclock_seconds"]
        if "queue_name" not in ph_options and options.get("queue_name"):
            if phonopy_code and self._same_computer(self.code_label, phonopy_code):
                ph_options["queue_name"] = options["queue_name"]
        inputs["phonopy_options"] = orm.Dict(dict=ph_options)

        return inputs

    @staticmethod
    def _same_computer(first_label: str, second_label: str) -> bool:
        """True when two code labels point at the same AiiDA computer."""
        from aiida.orm import load_code

        try:
            return (
                load_code(first_label).computer.pk
                == load_code(second_label).computer.pk
            )
        except Exception:  # noqa: BLE001 — unknown code: fall back to "not same"
            return False

    def _prepare_workflow_inputs(self) -> Tuple[list, list]:
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.phonopy.vasp import (
            VaspPhonopyWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=VaspPhonopyWorkChain,
            inputs=inputs,
        )
