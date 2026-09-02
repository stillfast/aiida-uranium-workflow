"""ABACUS input builder for the phonopy workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:class:`aiida_uranium_workflow.workflows.phonopy.abacus.AbacusPhonopyWorkChain`.

* the ABACUS SCF base (physical / SCF parameters, k-points, pseudos)
  comes from ``parameters/abacus/scf.yml`` (``software_params``) — the
  adapter forces ``calculation=scf`` and ``cal_force=1`` so each
  displaced supercell run returns atomic forces;
* the phonopy settings (supercell / primitive matrix / band path /
  DOS) come from the ``parameters/phonopy.yml`` protocol block
  (``workflow_data['phonopy']``).
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, Tuple

import copy


class AbacusPhonopyAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the ABACUS phonon WorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "uranium.phonopy.abacus"

    # ------------------------------------------------------------------
    # phonopy parameter assembly (pure dict logic, unit-testable)
    # ------------------------------------------------------------------

    @staticmethod
    def build_phonopy_parameters(phonopy_block: dict) -> dict[str, Any]:
        """Compose the ``phonopy_parameters`` Dict from the protocol block.

        Two band-path modes:

        * ``band_mode: auto`` — ``band_paths`` (``"auto"`` via seekpath or a
          named lattice path such as ``"bcc"`` / ``"fcc"``) is forwarded to
          phonopy's ``BAND`` tag (the band-structure run mode; a string value
          triggers the automatic / named path).
        * ``band_mode: manual`` — the explicit ``band`` flat k-point list and
          ``band_labels`` are forwarded to the ``BAND`` / ``BAND_LABELS`` tags.
        """
        ph = dict(phonopy_block or {})
        params: dict[str, Any] = {}

        band_mode = ph.get("band_mode", "auto")
        if band_mode == "manual":
            band = ph.get("band")
            if not band:
                raise ValueError(
                    "band_mode='manual' requires an explicit 'band' k-point "
                    "list in the phonopy.yml preset"
                )
            params["band"] = list(band)
            if ph.get("band_labels"):
                params["band_labels"] = list(ph["band_labels"])
        else:  # auto (default)
            # phonopy 4.x: ``BAND`` is the band-structure *run-mode* tag;
            # a string value ("auto" via seekpath, or a named path like
            # "bcc") triggers the computation. ``BAND_PATHS`` alone only
            # sets the path without running the band structure, which
            # leaves band.hdf5 missing and makes the parser fail.
            params["band"] = ph.get("band_paths", "auto")

        params["band_points"] = ph.get("band_points", 51)

        if ph.get("dos", True):
            params["dos"] = True
            params["mesh"] = list(ph.get("mesh", [21, 21, 21]))
            params["fmin"] = ph.get("fmin", 0.0)
            params["fmax"] = ph.get("fmax", 10.0)
            params["fpitch"] = ph.get("fpitch", 0.01)
        return params

    @staticmethod
    def build_displacement_generator(phonopy_block: dict) -> dict[str, Any]:
        """Displacement-generation dict (phonopy defaults + user distance)."""
        distance = (phonopy_block or {}).get("displacement_distance", 0.01)
        return {
            "distance": float(distance),
            "is_plusminus": "auto",
            "is_diagonal": True,
            "is_trigonal": False,
            "number_of_snapshots": None,
            "random_seed": None,
            "temperature": None,
            "cutoff_frequency": None,
        }

    # ------------------------------------------------------------------
    # WorkChain input assembly
    # ------------------------------------------------------------------

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        ph = dict(self.workflow_data.get("phonopy", {}) or {})
        params = copy.deepcopy(self.software_params["parameters"])

        # Fixed-lattice SCF with forces: this is what phonopy needs.
        params["input"]["calculation"] = "scf"
        params["input"]["cal_force"] = 1

        options = self.metadata.get("options", {})
        base: dict[str, Any] = {
            "abacus": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(params),
                "metadata": {"options": options} if options else {},
            },
        }
        if "kpoints_distance" in self.software_params:
            base["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            base["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            base["pseudo_family"] = orm.Str(str(self.software_params["pseudo_family"]))

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "phonopy_parameters": orm.Dict(dict=self.build_phonopy_parameters(ph)),
        }

        # phonopy pre-processing knobs
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

        # phonopy code + scheduler options
        phonopy_code = self.extra_codes.get("phonopy")
        if phonopy_code:
            inputs["phonopy_code"] = orm.load_code(phonopy_code)
        ph_options = dict(ph.get("options", {}) or {})
        # The phonopy CalcJob runs single-process but needs a long enough
        # wall-time (force-constant fitting + band / DOS). Inherit the
        # wall-time / queue from the backend metadata when the protocol
        # does not override them — without a wall-time the scheduler's
        # default (often short) can kill the job mid-run, leaving an
        # incomplete stdout (aiida-phonopy exit 312).
        meta_opts = self.metadata.get("options", {}) or {}
        for key in ("max_wallclock_seconds", "queue_name"):
            if key not in ph_options and meta_opts.get(key) is not None:
                ph_options[key] = meta_opts[key]
        inputs["phonopy_options"] = orm.Dict(dict=ph_options)

        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[list, list]:
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.phonopy.abacus import (
            AbacusPhonopyWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=AbacusPhonopyWorkChain,
            inputs=inputs,
        )
