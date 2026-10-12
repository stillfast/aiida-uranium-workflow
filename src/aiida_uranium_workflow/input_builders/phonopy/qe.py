"""QE (pw.x + ph.x) input builder for the phonopy workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:class:`aiida_uranium_workflow.workflows.phonopy.qe.QePhonopyWorkChain`
— the QE-native **DFPT** phonon chain (pw.x SCF → ``ph.x`` on a q-mesh
→ ``q2r.x`` → ``matdyn.x``), no frozen-phonon supercells:

* the pw SCF base (physical / SCF parameters, k-points, pseudos) comes
  from ``parameters/qe/scf.yml`` (``software_params``);
* the DFPT knobs (ph.x q-mesh, tr2_ph, asr, DOS mesh, ...) come from
  the ``qe`` sub-block of the ``parameters/phonopy.yml`` preset
  (``workflow_data['phonopy']['qe']``);
* the ph.x / q2r.x / matdyn.x code labels come from
  ``input.json["code"]`` via ``extra_codes`` (keys ``ph`` / ``q2r`` /
  ``matdyn``); the ``qe`` code key (``code_label``) is the pw.x code of
  the parent SCF.
"""

from __future__ import annotations

from aiida.common.exceptions import MultipleObjectsError, NotExistent
from aiida.orm import load_group
from ..base import filter_scheduler_options, AdaptedInputs, SoftwareAdapter
from typing import Any, Dict, Tuple

import copy

#: ph.x INPUTPH defaults (overridable via the preset's ``qe.ph.INPUTPH``).
#: Mirrors the values validated on the yeesuan cluster (test_qe_ph.py):
#: ``tr2_ph 1e-12`` + ``epsil=false`` converged quickly; ``asr=true`` pins
#: the acoustic branches at Γ already at the ph.x level (q2r.x then reads
#: ``zasr`` and matdyn.x reads ``asr`` for the interpolation).
_PH_DEFAULTS = {
    "tr2_ph": 1.0e-12,
    "epsil": False,
    "asr": True,
    "alpha_mix": 0.4,
    "nmix_ph": 8,
}


def _drop_spin_without_magnetization(params):
    """Remove an explicit nspin=2 when no starting magnetization is given.

    QE errors out with "some starting_magnetization MUST be set" for a
    spin-polarised (nspin=2) run without any initial moment. The shared
    SCF presets (``parameters/qe/scf.yml``) carry nspin=2 for the magmom
    sweep, which injects the magnetisation itself; non-magnetic
    workflows (relax / elastic / phonopy) must fall back to nspin=1.
    """
    system = params.get("SYSTEM", {})
    if system.get("nspin") != 2:
        return
    mag_keys = ("starting_magnetization", "tot_magnetization", "constrained_magnetization")
    if any(key in system for key in mag_keys):
        return
    del system["nspin"]


class QePhonopyAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the QE phonon WorkChain inputs."""

    name = "qe"

    def _workchain_entry_point(self) -> str:
        return "uranium.phonopy.qe"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        pw_block = self.software_params.get("pw", self.software_params)
        params = copy.deepcopy(pw_block.get("parameters", {}))
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )

        # The parent SCF stays a plain fixed-lattice SCF (ph.x reads its
        # charge density); non-magnetic runs drop any preset nspin=2.
        params.setdefault("CONTROL", {})["calculation"] = "scf"
        _drop_spin_without_magnetization(params)

        family = str(self.software_params.get("pseudo_family", ""))
        group = self._load_family() if family else None

        pseudos = {}
        if group is not None:
            elements = {
                element for kind in structure.kinds for element in kind.symbols
            }
            by_element = group.get_pseudos(elements=sorted(elements))
            for kind in structure.kinds:
                element = kind.symbols[0]
                try:
                    pseudos[kind.name] = by_element[element]
                except KeyError:
                    raise ValueError(
                        f"pseudo family {family!r} has no pseudopotential "
                        f"for element {element!r} "
                        f"(structure kinds: {[k.name for k in structure.kinds]})"
                    ) from None

            # Recommended cutoffs (Ry) override the preset reference values.
            system = params.setdefault("SYSTEM", {})
            try:
                ecutwfc, ecutrho = group.get_recommended_cutoffs(
                    structure=structure, unit="Ry"
                )
                system["ecutwfc"] = float(ecutwfc)
                system["ecutrho"] = float(ecutrho)
            except (AttributeError, TypeError, ValueError):
                pass  # keep the preset cutoffs when the family has none

        pw: Dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "parameters": orm.Dict(dict=params),
            "structure": structure,
        }
        if pseudos:
            pw["pseudos"] = pseudos
        if options:
            pw["metadata"] = {"options": options}

        # The ``base`` namespace is forwarded to the child PwBaseWorkChain
        # (the workflow only overrides ``pw.structure`` with the primitive
        # cell returned by seekpath).
        base: Dict[str, Any] = {"pw": pw}
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

        # ---- DFPT knobs from the phonopy.yml preset's ``qe`` block ----
        ph_block = dict(self.workflow_data.get("phonopy", {}) or {})
        qe = dict(ph_block.get("qe", {}) or {})

        def _need_code(key: str) -> str:
            label = self.extra_codes.get(key)
            if not label:
                raise ValueError(
                    f"QE phonopy (DFPT) needs a {key!r} code in input.json['code'] "
                    f"(e.g. \"{key}\": \"{key}@<host>\") — got extra_codes={self.extra_codes}"
                )
            return str(label)

        # ph.x namelist: defaults + preset overrides (INPUTPH).
        ph_params = copy.deepcopy(_PH_DEFAULTS)
        preset_inputph = dict(qe.get("ph", {}).get("INPUTPH", {}) or {})
        ph_params.update(preset_inputph)
        # q2r.x namelist: acoustic-sum rule by default (INPUT), preset wins.
        # NOTE: q2r.x reads ``zasr`` (not ``asr`` — that is matdyn.x's key);
        # an unknown namelist variable makes q2r.x abort with
        # "error reading input namelist".
        q2r_params = {"INPUT": {"zasr": "crystal"}}
        preset_q2r = dict(qe.get("q2r", {}).get("INPUT", {}) or {})
        q2r_params["INPUT"].update(preset_q2r)
        # matdyn.x namelist overrides (the dos flag is managed by the workflow).
        matdyn_params = dict(qe.get("matdyn", {}) or {})

        qpoints_mesh = list(qe.get("qpoints_mesh") or [4, 4, 4])
        dos = bool(qe.get("dos", True))

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "ph_code": orm.load_code(_need_code("ph")),
            "ph_parameters": orm.Dict(dict={"INPUTPH": ph_params}),
            "qpoints_mesh": orm.List(list=qpoints_mesh),
            "q2r_code": orm.load_code(_need_code("q2r")),
            "matdyn_code": orm.load_code(_need_code("matdyn")),
            "dos": orm.Bool(dos),
        }
        if "dos_mesh" in qe:
            inputs["dos_mesh"] = orm.List(list=list(qe["dos_mesh"]))
        if q2r_params["INPUT"]:
            inputs["q2r_parameters"] = orm.Dict(dict=q2r_params)
        if matdyn_params:
            inputs["matdyn_parameters"] = orm.Dict(dict=matdyn_params)
        if options:
            inputs["options"] = orm.Dict(dict=options)

        return inputs

    def _load_family(self):
        """Return the pseudo family group named in the preset.

        Same disambiguation as the relax / magmom / elastic qe adapters:
        when a collection and a family group share the label, the
        family-type group is selected.
        """
        family = str(self.software_params.get("pseudo_family", ""))
        if not family:
            return None
        try:
            return load_group(family)
        except MultipleObjectsError:
            pass
        except NotExistent:
            raise ValueError(f"pseudo family group {family!r} does not exist") from None

        from aiida.orm import Group, QueryBuilder

        rows = (
            QueryBuilder()
            .append(Group, filters={"label": family}, project=["id", "type_string"])
            .all()
        )
        family_rows = [row for row in rows if "family" in row[1]]
        if len(family_rows) == 1:
            return Group.collection.get(id=family_rows[0][0])
        raise ValueError(
            f"multiple groups named {family!r} and no unique family-type "
            f"candidate ({rows}); rename or delete the duplicates"
        )

    def _prepare_workflow_inputs(self) -> Tuple[list, list]:
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.phonopy.qe import QePhonopyWorkChain

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=QePhonopyWorkChain,
            inputs=inputs,
        )
