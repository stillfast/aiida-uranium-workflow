"""FLEUR magmom WorkChain.

Sweeps one ``FleurScfWorkChain`` per magnetic configuration in
``magmom_list`` and gathers the resulting magnetism / energy outputs.

Each ``magmom_list`` entry is a dict describing one initial magnetic
configuration on top of the preset's SCF namespace (mirrors the
hand-written ``magmom_test/fleur`` runner, see
``aiida-uranium-scripts/magmom_test/fleur/magmom/``):

* ``bmu``            — inpgen-level seed: ``calc_parameters.atom.bmu``
                       (0.0 → non-magnetic, +4.0 → ferromagnetic seed)
* ``inpxml_changes`` — optional extra FLEUR input changes, appended to
                       the preset's ``wf_parameters.inpxml_changes``.
                       FM: ``set_species`` ``nocoParams.l_magn=T``;
                       AFM: that plus ``set_atomgroup`` ``beta=Pi`` to
                       flip the second atom's spin quantisation axis.
* ``label``          — optional short tag used in the child label.

The preset (``parameters/fleur/magmom.yml``) carries the shared SCF
namespace — ``l_noco=T``, ``l_soc`` per preset, ``bzIntegration``
mode, ``jspins=2``, cutoffs, k-mesh — exactly like the banddos FLEUR
presets (``parameters/fleur/banddos.yml``), so the two preset files
share one format.

Layout::

    inputs
    ├── fleur / inpgen / structure / options / options_inpgen
    ├── wf_parameters      (Dict) — FleurScfWorkChain settings (SCF namespace)
    ├── calc_parameters    (Dict) — inpgen parameters (kpt / comp / atom)
    ├── magmom_list        (List) — one entry per child (dict, see above)
    outputs
    └── output_parameters  (Dict) — per-child magnetism / energy / status
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

#: Hartree → eV (FLEUR reports energies in Hartree).
HA_TO_EV = 27.211386245988

FleurScfWorkChain = WorkflowFactory("fleur.scf")


class FleurMagmomWorkChain(WorkChain):
    """Sweep over a list of FLEUR magnetic configurations."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            FleurScfWorkChain,
            include=[
                "fleur",
                "inpgen",
                "wf_parameters",
                "calc_parameters",
                "structure",
                "options",
                "options_inpgen",
                "settings",
                "settings_inpgen",
            ],
        )

        spec.input(
            "magmom_list",
            valid_type=orm.List,
            help=(
                "List of initial magnetic configurations, one per child "
                "FleurScfWorkChain. Each entry is a dict: "
                "{'bmu': 4.0, 'inpxml_changes': [[...]], 'label': 'FM'}."
                "'bmu' sets calc_parameters.atom.bmu; 'inpxml_changes' "
                "are appended to wf_parameters.inpxml_changes."
            ),
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "magmom scan completed successfully.")
        spec.exit_code(300, "ERROR_CHILD", "A child calculation failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse magnetism.")

    def _base_namespace(self):
        """Return (wf_parameters, calc_parameters) plain dicts."""
        wf = dict(self.inputs.wf_parameters.get_dict())
        calc = dict(self.inputs.calc_parameters.get_dict())
        return wf, calc

    def submit_children(self):
        """Submit one child FleurScfWorkChain per magmom configuration."""
        magmom_list = self.inputs.magmom_list.get_list()
        base_wf, base_calc = self._base_namespace()

        base_child_inputs = {
            "fleur": self.inputs.fleur,
            "structure": self.inputs.structure,
        }
        for port in ("inpgen", "options", "options_inpgen", "settings", "settings_inpgen"):
            if port in self.inputs:
                base_child_inputs[port] = self.inputs[port]

        for idx, entry in enumerate(magmom_list):
            if not isinstance(entry, dict):
                self.report(
                    f"magmom_list entry #{idx} is not a dict: {entry!r}"
                )
                return self.exit_codes.ERROR_PARSER

            tag = str(entry.get("label") or f"case{idx}")
            safe_tag = _tag_to_label(tag)
            label = f"magmom_{idx:03d}_{safe_tag}"

            wf = dict(base_wf)
            calc = dict(base_calc)

            # inpgen seed: override the per-species magnetic moment.
            if "bmu" in entry:
                atom = dict(calc.get("atom", {}))
                atom["bmu"] = float(entry["bmu"])
                calc["atom"] = atom

            # Magnetic-configuration specific input changes: append to
            # the preset's base changes (l_noco / l_soc / smearing …).
            extra_changes = entry.get("inpxml_changes") or []
            if extra_changes:
                wf["inpxml_changes"] = list(wf.get("inpxml_changes", [])) + list(
                    extra_changes
                )

            child_inputs = dict(base_child_inputs)
            child_inputs["wf_parameters"] = orm.Dict(wf)
            child_inputs["calc_parameters"] = orm.Dict(calc)
            child_inputs["metadata"] = {
                "label": label,
                "description": f"FLEUR magmom config {tag}",
            }

            try:
                running = self.submit(FleurScfWorkChain, **child_inputs)
            except Exception as exc:
                self.report(f"submission failed for magmom={tag}: {exc}")
                raise

            self.report(f"submitted child pk={running.pk} for magmom={tag}")
            self.to_context(**{label: running})

    def gather_results(self):
        magmom_list = self.inputs.magmom_list.get_list()

        all_finished_ok = True
        child_pks = []

        for idx, entry in enumerate(magmom_list):
            tag = str(entry.get("label") or f"case{idx}")
            safe_tag = _tag_to_label(tag)
            label = f"magmom_{idx:03d}_{safe_tag}"
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_magmom_results(
            child_pks=orm.List(child_pks),
            magmom_configs=orm.List(list=magmom_list),
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_magmom_results(child_pks, magmom_configs):
    """Parse the magnetism outputs of each child FLEUR SCF and gather them.

    ``output_scf_wc_para`` carries ``total_energy`` (Hartree) plus the
    last_calc ``output_parameters`` (masci-tools out.xml parse) which
    carries ``magnetic_vec_moments`` (per-atom 3-vectors, from
    ``<globalMagMoment vec="…"/>``) for non-collinear runs.
    """
    from aiida.orm import load_node

    magnetization = {}
    total_energy_hartree = {}
    final_energy = {}  # eV, for report consistency with abacus/vasp
    wall_time_seconds = {}
    status = {}
    config_labels = {}

    for idx, pk in enumerate(child_pks.get_list()):
        child = load_node(pk)
        status[pk] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )
        configs = magmom_configs.get_list()
        config_labels[pk] = str(
            configs[idx].get("label") if idx < len(configs) else f"case{idx}"
        )

        if not child.is_finished_ok:
            continue

        try:
            scf_para = child.outputs.output_scf_wc_para.get_dict()
        except (AttributeError, KeyError):
            continue

        e_hartree = scf_para.get("total_energy")
        total_energy_hartree[pk] = e_hartree
        if e_hartree is not None:
            final_energy[pk] = float(e_hartree) * HA_TO_EV
        wall_time_seconds[pk] = scf_para.get("total_wall_time")

        # Per-atom magnetic moments from the last calculation's out.xml
        # parse (magnetic_vec_moments: list of [mx, my, mz] per atom).
        try:
            last_para = child.outputs.last_calc.output_parameters.get_dict()
        except (AttributeError, KeyError):
            last_para = {}
        magnetization[pk] = last_para.get("magnetic_vec_moments")

    result = {
        "magnetization": magnetization,
        "total_energy_hartree": total_energy_hartree,
        "final_energy": final_energy,
        "energy_units": "eV",
        "wall_time_seconds": wall_time_seconds,
        "status": status,
        "config_labels": config_labels,
    }
    return orm.Dict(result)


def _tag_to_label(tag: str) -> str:
    """Render a magmom config tag as a filesystem-friendly label."""
    return str(tag).replace("/", "_").replace(" ", "_")
