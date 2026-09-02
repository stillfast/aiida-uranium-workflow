from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from aiida_uranium_workflow.utils.labels import format_magmom_label

ChildWorkChain = WorkflowFactory("vasp.v2.vasp")


class VaspMagmomWorkChain(WorkChain):
    """Sweep over a list of initial magnetic configurations and gather the
    resulting magnetism outputs.

    Two mutually-exclusive input styles are supported:

    * ``magmom_list`` — one entry per child is a per-species mapping dict
      (``{"Si": 1.0}`` / ``{"Si": [1.0, -1.0]}``); passed to the child
      ``VaspWorkChain`` via its ``magmom_mapping`` port.
    * ``magmom_per_atom_list`` — one entry per child is a per-atom list of
      initial magnetic moments in site order (``[0.0, 0.0]`` /
      ``[4.0, -4.0]``); passed to the child ``VaspWorkChain`` via its
      ``magmom_per_atom`` port (which takes precedence over
      ``magmom_mapping`` inside aiida-vasp).

    Outline:  submit_children → gather_results

    Exit codes
    ----------
    * 0   ``SUCCESS``         — sweep completed normally.
    * 300 ``ERROR_CHILD``     — a child calculation failed.
    * 305 ``ERROR_PARSER``    — failed to parse magnetism outputs.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=[
                "code",
                "structure",
                "kpoints",
                "kpoints_spacing",
                "parameters",
                "potential_family",
                "potential_mapping",
                "calc",
            ],
        )

        spec.input(
            "magmom_list",
            valid_type=orm.List,
            required=False,
            help=(
                "List of ``magmom_mapping`` dictionaries, one per child "
                "calculation. Each entry maps element symbol to its "
                "magnetization, e.g. {'Si': 1.0} or {'Si': [1.0, -1.0]}."
                "Mutually exclusive with ``magmom_per_atom_list``."
            ),
        )
        spec.input(
            "magmom_per_atom_list",
            valid_type=orm.List,
            required=False,
            help=(
                "List of per-atom initial magnetic moments, one entry per "
                "child calculation. Each entry is a list in site order, "
                "e.g. [0.0, 0.0] or [4.0, -4.0] (collinear) / "
                "[ [1.0,0.0,0.0], ... ] (non-collinear, 3 components per "
                "site). Passed to the child VaspWorkChain's "
                "``magmom_per_atom`` port. Mutually exclusive with "
                "``magmom_list``."
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

    def _magmom_entries(self):
        """Return ``(kind, entries)`` for the supplied magnetic configs.

        ``kind`` is ``"mapping"`` (per-species dicts) or ``"per_atom"``
        (per-site lists), depending on which input port was provided.
        """
        if "magmom_per_atom_list" in self.inputs:
            return "per_atom", self.inputs.magmom_per_atom_list.get_list()
        return "mapping", self.inputs.magmom_list.get_list()

    def submit_children(self):
        """Submit one child VaspWorkChain per magmom configuration."""
        kind, magmom_entries = self._magmom_entries()

        base_inputs = self.exposed_inputs(ChildWorkChain, agglomerate=True)

        for idx, mag_value in enumerate(magmom_entries):
            label = format_magmom_label(mag_value, index=idx)

            child_inputs = dict(base_inputs)
            if kind == "per_atom":
                # aiida-vasp v2 ``VaspWorkChain.magmom_per_atom`` port:
                # one initial moment per site (scalar, or 3-vector for
                # non-collinear); takes precedence over magmom_mapping.
                child_inputs["magmom_per_atom"] = orm.List(list=mag_value)
            else:
                child_inputs["magmom_mapping"] = orm.Dict(mag_value)
            child_inputs["metadata"] = {"label": label}
            child_inputs["calc"] = {
                **base_inputs.get("calc", {}),
            }

            try:
                running = self.submit(ChildWorkChain, **child_inputs)
            except Exception as exc:
                self.report(
                    f"submission failed for magmom={mag_value}: {exc}"
                )
                raise

            self.report(
                f"submitted child pk={running.pk} for magmom={mag_value}"
            )
            self.to_context(**{label: running})

    def gather_results(self):
        _, magmom_entries = self._magmom_entries()

        all_finished_ok = True
        child_pks = []

        for idx, mag_value in enumerate(magmom_entries):
            label = format_magmom_label(mag_value, index=idx)
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_magmom_results(
            child_pks=orm.List(child_pks),
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_magmom_results(child_pks):
    """Parse the magnetism outputs of each child VASP calculation and gather them.

    The ``misc`` output of a VASP calculation exposes ``magnetization`` (total
    magnetization) and ``site_magnetization`` (per-site magnetization).
    """
    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parser_energy_time import fetch_vasp

    magnetization = {}
    site_magnetization = {}
    final_energy = {}
    wall_time_seconds = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        # Capture exit status for every submitted child, even those
        # that did not finish OK. ``exit_status`` is ``None`` for
        # unfinished processes; we render that as ``-1``.
        status[pk] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )

        if not child.is_finished_ok:
            continue

        try:
            misc = child.outputs.misc.get_dict()
        except (AttributeError, KeyError):
            continue

        magnetization[pk] = misc.get("magnetization")
        site_magnetization[pk] = misc.get("site_magnetization")

        # Shared parser: consistent with smear / convergence.
        energy, wall_time = fetch_vasp(child)
        final_energy[pk] = energy
        wall_time_seconds[pk] = wall_time

    result = {
        "magnetization": magnetization,
        "site_magnetization": site_magnetization,
        "final_energy": final_energy,
        "wall_time_seconds": wall_time_seconds,
        "status": status,
    }
    return orm.Dict(result)
