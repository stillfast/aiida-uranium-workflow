"""FLEUR input builder for the phonopy workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:class:`aiida_uranium_workflow.workflows.phonopy.fleur.FleurPhonopyWorkChain`.

* the FLEUR SCF base (physical / SCF parameters, k-points) comes from
  ``parameters/fleur/scf.yml`` (``software_params``) — the adapter
  switches the SCF ``wf_parameters`` to ``mode='force'`` so each
  displaced supercell run returns atomic forces (FLEUR computes forces
  only with the geometry optimisation active, ``l_f="T"`` — fleur.md
  §4.5; aiida-fleur writes them to ``relax.xml``, exposed as
  ``relax_parameters['posforces']``);
* the phonopy settings (supercell / primitive matrix / band path /
  DOS) come from the ``parameters/phonopy.yml`` protocol block
  (``workflow_data['phonopy']``) — reused verbatim from the ABACUS
  adapter's helpers.
"""

from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from ..phonopy.abacus import AbacusPhonopyAdapter
from typing import Any, Tuple

#: Force-convergence settings used by FleurScfWorkChain's ``force`` mode
#: (merged into the SCF ``wf_parameters``; user values win).
_FORCE_WF_DEFAULTS = {
    "force_converged": 0.002,  # Htr/bohr
    "force_dict": {
        "qfix": 2,
        "forcealpha": 0.5,
        "forcemix": "straight",
    },
}


class FleurPhonopyAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the FLEUR phonon WorkChain inputs."""

    name = "fleur"

    def _workchain_entry_point(self) -> str:
        return "uranium.phonopy.fleur"

    # build_phonopy_parameters / build_displacement_generator are shared
    # with the ABACUS adapter (pure dict logic).
    build_phonopy_parameters = staticmethod(AbacusPhonopyAdapter.build_phonopy_parameters)
    build_displacement_generator = staticmethod(
        AbacusPhonopyAdapter.build_displacement_generator
    )

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        preset = dict(self.software_params)
        if "wf_parameters" not in preset or "calc_parameters" not in preset:
            raise ValueError(
                "FLEUR phonopy preset is missing 'wf_parameters' or "
                "'calc_parameters' — the SCF base can't be built."
            )

        ph = dict(self.workflow_data.get("phonopy", {}) or {})

        # Switch the SCF to force mode: FLEUR only computes atomic forces
        # when the geometry optimisation is active (l_f="T"), and
        # aiida-fleur's mode='force' sets that and exposes the forces via
        # relax_parameters.
        wf_parameters = dict(preset["wf_parameters"])
        wf_parameters["mode"] = "force"
        wf_parameters.setdefault("force_converged", _FORCE_WF_DEFAULTS["force_converged"])
        wf_parameters.setdefault("force_dict", dict(_FORCE_WF_DEFAULTS["force_dict"]))

        options = self.metadata.get("options", {})
        base: dict[str, Any] = {
            "fleur": orm.load_code(self.code_label),
            "wf_parameters": orm.Dict(dict=wf_parameters),
            "calc_parameters": orm.Dict(dict=dict(preset["calc_parameters"])),
        }
        inpgen_label = self.extra_codes.get("inpgen")
        if inpgen_label:
            base["inpgen"] = orm.load_code(inpgen_label)
        if options:
            base["options"] = orm.Dict(dict=options)

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "phonopy_parameters": orm.Dict(dict=self.build_phonopy_parameters(ph)),
        }

        # phonopy pre-processing knobs (same layout as the ABACUS adapter)
        if "supercell_matrix" in ph:
            inputs["supercell_matrix"] = orm.List(list=list(ph["supercell_matrix"]))
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

        if ph.get("band_labels"):
            inputs["band_labels"] = orm.List(list=list(ph["band_labels"]))

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
        from aiida_uranium_workflow.workflows.phonopy.fleur import (
            FleurPhonopyWorkChain,
        )

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=FleurPhonopyWorkChain,
            inputs=inputs,
        )
