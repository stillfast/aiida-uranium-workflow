"""QE input builder for the eos workflow.

Reads the pw SCF base from ``parameters/qe/scf.yml`` (via the shared
QE pseudo-family / k-point helpers) and the EOS scan settings from the
workflow protocol (``parameters/eos.yml``'s ``qe`` block), then
assembles inputs for
:class:`aiida_uranium_workflow.workflows.eos.qe.QeEosWorkChain`.
"""

from __future__ import annotations

from aiida.common.exceptions import MultipleObjectsError, NotExistent
from aiida.orm import load_group
from ..base import filter_scheduler_options, AdaptedInputs, SoftwareAdapter
from typing import Any, Dict

import copy

#: EOS scan settings consumed from the protocol's ``qe`` block.
_EOS_KEYS = ("points", "step", "guess")


class QeEosAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the QE EOS WorkChain inputs."""

    name = "qe"

    def _workchain_entry_point(self) -> str:
        return "uranium.eos.qe"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        pw_block = self.software_params.get("pw", self.software_params)
        params = copy.deepcopy(pw_block.get("parameters", {}))
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )

        # Fixed-lattice SCF — the cell must stay at the scaled volume.
        params.setdefault("CONTROL", {})["calculation"] = "scf"
        # Drop any nspin=2 without an initial magnetisation (QE errors
        # out on a spin-polarised run with no starting moment).
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

        # EOS scan settings from the protocol's qe block.
        eos_proto = dict(self.workflow_data.get("qe", {}) or {})
        defaults = {"points": 9, "step": 0.005, "guess": 1.00}
        eos_settings = dict(defaults)
        eos_settings.update({k: v for k, v in eos_proto.items() if k in _EOS_KEYS})

        inputs: dict[str, Any] = {
            "structure": structure,
            "base": base,
            "eos_parameters": orm.Dict(dict=eos_settings),
        }
        return inputs

    def _load_family(self):
        """Return the pseudo family group named in the preset (see the
        relax / magmom / elastic qe adapters for the disambiguation)."""
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

    def _prepare_workflow_inputs(self):
        return [], []

    def adapt(self, structure) -> AdaptedInputs:
        from aiida_uranium_workflow.workflows.eos.qe import QeEosWorkChain

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=QeEosWorkChain,
            inputs=inputs,
        )


def _drop_spin_without_magnetization(params):
    """Remove an explicit nspin=2 when no starting magnetization is given."""
    system = params.get("SYSTEM", {})
    if system.get("nspin") != 2:
        return
    mag_keys = ("starting_magnetization", "tot_magnetization", "constrained_magnetization")
    if any(key in system for key in mag_keys):
        return
    del system["nspin"]
