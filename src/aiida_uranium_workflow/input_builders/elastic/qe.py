"""QE input builder for the elastic workflow.

Reads the pw (SCF / relax) parameters from ``parameters/qe/elastic.yml``
(self-contained preset) and the strain lists from the workflow protocol
(``parameters/elastic.yml``'s ``qe`` block), then assembles inputs for
:class:`aiida_uranium_workflow.workflows.elastic.qe.QeElasticWorkChain`.
"""

from __future__ import annotations

from aiida.common.exceptions import MultipleObjectsError, NotExistent
from aiida.orm import load_group
from ..base import filter_scheduler_options, AdaptedInputs, SoftwareAdapter
from typing import Any, Dict

import copy



def _drop_spin_without_magnetization(params):
    """Remove an explicit nspin=2 when no starting magnetization is given.

    QE errors out with "some starting_magnetization MUST be set" for a
    spin-polarised (nspin=2) run without any initial moment. The shared
    SCF presets (``parameters/qe/scf.yml``) carry nspin=2 for the magmom
    sweep, which injects the magnetisation itself; non-magnetic
    workflows (relax / elastic / base) must fall back to nspin=1.
    """
    system = params.get("SYSTEM", {})
    if system.get("nspin") != 2:
        return
    mag_keys = ("starting_magnetization", "tot_magnetization", "constrained_magnetization")
    if any(key in system for key in mag_keys):
        return
    del system["nspin"]


class QeElasticAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the QE elastic WorkChain inputs."""

    name = "qe"

    def _workchain_entry_point(self) -> str:
        return "uranium.elastic.qe"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        pw_block = self.software_params.get("pw", self.software_params)
        params = copy.deepcopy(pw_block.get("parameters", {}))
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )
        _drop_spin_without_magnetization(params)
        # QE only computes the stress tensor when it is needed (vc-relax
        # and friends); a plain 'scf' / 'relax' run silently skips it
        # (read_namelists resets ``tstress`` to .FALSE. for them). The
        # elastic stress method needs the stress on every deformed
        # cell, so force ``tstress=.true.`` explicitly.
        params.setdefault("CONTROL", {})["tstress"] = True

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

        inputs: dict[str, Any] = {
            "structure": structure,
            "pw": pw,
        }
        if "kpoints_distance" in self.software_params:
            inputs["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            inputs["kpoints"] = kpoints_mesh

        elastic_proto = dict(self.workflow_data.get("qe", {}) or {})
        norm = elastic_proto.get("norm_strains") or [-0.010, -0.005, 0.005, 0.010]
        shear = elastic_proto.get("shear_strains") or [-0.010, -0.005, 0.005, 0.010]
        inputs["norm_strains"] = orm.List(list=norm)
        inputs["shear_strains"] = orm.List(list=shear)
        # Relax internal coordinates under each strain (pw
        # calculation='relax', cell fixed) — default True, matching the
        # abacus / vasp elastic workflows.
        inputs["relax_internal"] = orm.Bool(
            elastic_proto.get("relax_internal", True)
        )
        return inputs

    def _load_family(self):
        """Return the pseudo family group named in the preset (see the
        relax / magmom qe adapters for the disambiguation notes)."""
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
        from aiida_uranium_workflow.workflows.elastic.qe import QeElasticWorkChain

        inputs = self._build_workchain_inputs(structure)
        return AdaptedInputs(
            workchain_cls=QeElasticWorkChain,
            inputs=inputs,
        )
