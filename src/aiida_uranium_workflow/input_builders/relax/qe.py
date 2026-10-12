"""QE (pw.x) input builder for the relax workflow.

Thin wrapper around the **plugin** ``quantumespresso.pw.relax``
WorkChain (:class:`aiida_quantumespresso.workflows.relax.
PwRelaxWorkChain`) — no custom WorkChain is needed, mirroring how the
abacus / fleur relax methods call the plugin relax WorkChains directly.

The pw (SCF / relax) parameters live in the self-contained preset file
``parameters/qe/relax.yml`` (the SCF base: cutoffs / smearing /
pseudopotential family / k-points); the PwRelaxWorkChain-level controls
(``meta_convergence`` etc.) come from the workflow protocol
(``parameters/relax.yml``'s ``qe`` block).
"""

from __future__ import annotations

from aiida.common.exceptions import MultipleObjectsError, NotExistent
from aiida.orm import load_group
from ..base import filter_scheduler_options, SoftwareAdapter
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


class QeRelaxAdapter(SoftwareAdapter):
    """Translate a ParamBundle into the plugin PwRelaxWorkChain inputs."""

    name = "qe"

    def _workchain_entry_point(self) -> str:
        return "quantumespresso.pw.relax"

    @staticmethod
    def _apply_relax_mode(params: Dict[str, Any], relax_proto: Dict[str, Any]) -> None:
        """Map the protocol's ``relax_type`` / ``force_criterion`` onto pw params.

        Defaults (no protocol keys) keep the preset's own settings; the
        preset's ``calculation`` (usually 'relax') is only overridden
        when ``relax_type`` explicitly asks for a cell relaxation.
        """
        control = params.setdefault("CONTROL", {})

        relax_type = relax_proto.get("relax_type")
        cell = params.setdefault("CELL", {})
        # QE limitation: CELL.cell_dofree='volume' (strictly isotropic,
        # shape-locked volume change) is only accepted for ibrav=1
        # (simple cubic). For any other lattice the volume-only
        # relaxation is done with a plain ``vc-relax`` — under a cubic /
        # orthorhombic lattice the hydrostatic stress keeps the cell
        # shape by symmetry, so the converged cell is the volume-only
        # result (for cells without free internal coordinates this is a
        # pure volume relaxation). Pass an explicit ``cell_dofree`` in
        # the protocol to lock specific degrees of freedom.
        if relax_type is None:
            # Default to a positions-only relaxation unless the preset
            # already chose a calculation mode.
            control.setdefault("calculation", "relax")
        elif relax_type in ("volume", "full"):
            control["calculation"] = "vc-relax"
            if relax_type == "full":
                cell.setdefault("cell_dofree", "all")
            # aiida-qe flags the calculation as failed
            # (ERROR_IONIC_CONVERGENCE_REACHED_FINAL_SCF_FAILED) when the
            # final SCF's pressure exceeds its default press_conv_thr of
            # 0.5 kbar — QE itself usually stops with a slightly larger
            # residual (e.g. 0.6 kbar for the bcc-U volume relax). Use a
            # more permissive default, overridable via the protocol.
            cell.setdefault(
                "press_conv_thr", float(relax_proto.get("press_conv_thr", 1.0))
            )
        elif relax_type == "positions":
            control["calculation"] = "relax"
        else:
            raise ValueError(
                f"qe relax_type must be one of 'positions' / 'volume' / "
                f"'full', got {relax_type!r}"
            )
        # Explicit protocol override wins over the relax_type mapping.
        if "cell_dofree" in relax_proto:
            cell["cell_dofree"] = relax_proto["cell_dofree"]

        force_criterion = relax_proto.get("force_criterion")
        if force_criterion is not None:
            # eV/Å → QE CONTROL.forc_conv_thr (Ry/bohr; 1 Ry/bohr ≈
            # 25.711 eV/Å). QE 7.5 puts the force criterion in &CONTROL
            # (under the ``forc_conv_thr`` name) — placing it in &IONS
            # makes pw.x abort with "bad line in namelist &ions".
            control["forc_conv_thr"] = float(force_criterion) / 25.711

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        pw_block = self.software_params.get("pw", self.software_params)
        params = copy.deepcopy(pw_block.get("parameters", {}))
        options = filter_scheduler_options(
            self.metadata.get("options", {}), entry_point=self._workchain_entry_point(), backend=self.name
        )

        # Relax-mode controls come from the protocol's qe block:
        #   * ``relax_type`` — 'positions' (default: ions move, cell
        #     fixed), 'volume' (vc-relax + CELL.cell_dofree='volume':
        #     isotropic cell-volume change, shape preserved — for cells
        #     without free internal coordinates this is a pure
        #     volume-only relaxation, the QE counterpart of the abacus
        #     volume relax) or 'full' (vc-relax, cell_dofree='all').
        #   * ``force_criterion`` — converged largest force in eV/Å,
        #     converted to QE IONS.force_conv_thr (Ry/bohr:
        #     1 Ry/bohr ≈ 25.711 eV/Å).
        relax_proto = dict(self.workflow_data.get("qe", {}) or {})
        self._apply_relax_mode(params, relax_proto)
        _drop_spin_without_magnetization(params)

        family = str(self.software_params.get("pseudo_family", ""))
        group = self._load_family() if family else None

        # Pseudos keyed by kind name (split species all map to their
        # element's pseudopotential), fetched by element.
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

        base_relax: Dict[str, Any] = {"pw": pw}
        if "kpoints_distance" in self.software_params:
            base_relax["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            base_relax["kpoints"] = kpoints_mesh

        inputs: dict[str, Any] = {
            "structure": structure,
            "base_relax": base_relax,
        }

        # PwRelaxWorkChain-level controls from the protocol's qe block.
        relax_proto = dict(self.workflow_data.get("qe", {}) or {})
        bool_keys = ("meta_convergence", "clean_workdir")
        int_keys = ("max_meta_convergence_iterations",)
        for key in bool_keys:
            if key in relax_proto:
                inputs[key] = orm.Bool(bool(relax_proto[key]))
        for key in int_keys:
            if key in relax_proto:
                inputs[key] = orm.Int(int(relax_proto[key]))
        return inputs

    def _load_family(self):
        """Return the pseudo family group named in the preset.

        Same disambiguation as the magmom qe adapter: when a collection
        and a family group share the label, the family-type group is
        selected.
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

    def _prepare_workflow_inputs(self):
        return [], []

    # ``adapt`` uses the base implementation, which submits the plugin
    # WorkChain resolved from ``_workchain_entry_point``
    # ("quantumespresso.pw.relax"); the scheduler options are already
    # placed under ``base_relax.pw.metadata`` by
    # :meth:`_build_workchain_inputs`.
