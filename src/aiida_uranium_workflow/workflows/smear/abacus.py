"""ABACUS smear WorkChain (standard AiiDA ``WorkChain``).

Mirrors the file layout of ``aiida_abacus.workflows.base``:

* :class:`AbacusSmearWorkChain` — sweeps over ``(smearing_method, sigma)``

The actual AiiDA ``inputs`` dict is assembled by
:class:`aiida_uranium_workflow.input_builders.AbacusAdapter`.

Note
----
The abacus WorkChain is registered under the entry-point
``abacus.smear``.  Falling back to the plugin-provided
``AbacusBaseWorkChain`` allows the workflow to be submitted before the
full ``*SmearWorkChain`` class is published as an entry point.

Dry-run support
----------------
Setting the standard AiiDA flag ``metadata.dry_run = True`` on submit
is honoured in two ways:

1. The top-level ``engine.submit()`` dispatches to ``run_get_node()``
   so the workchain outline runs synchronously.
2. ``AbacusSmearWorkChain.submit_children`` records the planned
   ``(smear, sigma)`` pairs into ``self.ctx.smear_pairs`` and exits
   the outline via the ``DRY_RUN_SUCCESS`` exit code without ever
   submitting a CalcJob — so no scheduler / transport action happens.

Tests therefore only need a valid AiiDA profile; the configured
``Computer`` does not need to actually accept SSH connections.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory
from itertools import product

ChildWorkChain = WorkflowFactory("abacus.base")


class AbacusSmearWorkChain(WorkChain):
    """Sweep over ``(smearing_method, sigma)`` combinations and gather the
    electronic entropy.

    Outline:  submit_children → check_children → gather_results

    Exit codes
    ----------
    * 0   ``SUCCESS``              — sweep completed normally.
    * 300 ``ERROR_CHILD``          — a child calculation failed.
    * 305 ``ERROR_PARSER``         — failed to parse electronic entropy.
    * 404 ``DRY_RUN_SUCCESS``      — submitted in ``metadata.dry_run``
      mode; the outline ran to ``submit_children`` and the planned
      ``(smear, sigma)`` sweep was recorded, but no CalcJob was
      launched.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            ChildWorkChain,
            include=(
                "abacus",
                "kpoints",
                "kpoints_distance",
                "pseudo_family",
                "abacus.metadata",
            ),
        )

        spec.input(
            "smear",
            valid_type=orm.List,
            help="List of ABACUS smearing-method keywords.",
        )
        spec.input(
            "sigma",
            valid_type=orm.List,
            help="List of smearing sigma values in Rydberg.",
        )
        spec.input("metadata.dry_run", valid_type=bool, required=False, non_db=True)
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "smear scan completed successfully.")
        spec.exit_code(300, "ERROR_CHILD", "A child calculation failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to parse electronic entropy.")
        spec.exit_code(
            404,
            "DRY_RUN_SUCCESS",
            "dry-run: workchain reached submit_children; no CalcJobs were launched.",
        )

    def _dry_run(self) -> bool:
        try:
            return bool(self.inputs.metadata.dry_run)
        except (AttributeError, KeyError):
            return False

    def submit_children(self):
        """Submit one child WorkChain per (smearing_method, sigma) pair."""
        smear_list = self.inputs.smear.get_list()
        sigma_list = self.inputs.sigma.get_list()

        planned = list(product(smear_list, sigma_list))
        self.ctx.smear_pairs = planned

        if self._dry_run():
            self.report(
                f"dry-run: skipping child submission for {len(planned)} "
                f"(smear, sigma) pair(s)"
            )
            for smear, sigma in planned:
                self.report(f"dry-run: planned child for smear={smear}, sigma={sigma}")
            self.node.base.extras.set("dry_run", True)
            self.node.base.extras.set("smear_pairs", [list(p) for p in planned])
            return self.exit_codes.DRY_RUN_SUCCESS

        for smear, sigma in planned:
            abacus_block = self.inputs.abacus

            if "parameters" not in abacus_block:
                self.report(
                    f"no parameters under self.inputs.abacus; "
                    f"available keys: {sorted(abacus_block.keys())}"
                )
                continue

            label = f"smear_{smear}_sigma_{sigma}".replace(".", "_")

            param_dict = abacus_block.parameters.get_dict()
            param_dict.setdefault("input", {})["smearing_method"] = smear
            param_dict["input"]["smearing_sigma"] = sigma

            self.report(f"submitting child for smear={smear}, sigma={sigma}")

            child_inputs = {
                "abacus": {
                    **abacus_block,
                    "parameters": orm.Dict(param_dict),
                },
                "metadata": {"label": label},
                "pseudo_family": self.inputs.pseudo_family,
            }
            if hasattr(self.inputs, "kpoints"):
                child_inputs["kpoints"] = self.inputs.kpoints
            else:
                child_inputs["kpoints_distance"] = self.inputs.kpoints_distance

            try:
                running = self.submit(ChildWorkChain, **child_inputs)
            except Exception as exc:
                self.report(
                    f"submission failed for smear={smear}, sigma={sigma}: {exc}"
                )
                raise

            self.report(
                f"submitted child pk={running.pk} for smear={smear}, sigma={sigma}"
            )
            self.to_context(**{label: running})

    def gather_results(self):
        if self._dry_run():
            planned = self.ctx.get("smear_pairs") or []
            output = orm.Dict(dict={"dry_run": True, "smear_pairs": planned})
            output.store()
            self.out("output_parameters", output)
            return

        smear_list = self.inputs.smear.get_list()
        sigma_list = self.inputs.sigma.get_list()

        all_finished_ok = True
        child_pks = []

        for smear, sigma in product(smear_list, sigma_list):
            label = f"smear_{smear}_sigma_{sigma}".replace(".", "_")
            child = getattr(self.ctx, label, None)

            if child is None:
                all_finished_ok = False
                continue

            if not child.is_finished_ok:
                all_finished_ok = False

            child_pks.append(child.pk)

        results_node = parse_and_gather_smear_results(
            child_pks=orm.List(child_pks)
        )

        self.out("output_parameters", results_node)

        if not all_finished_ok:
            return self.exit_codes.ERROR_CHILD


@calcfunction
def parse_and_gather_smear_results(child_pks):
    """通过 calcfunction 将原始文件解析过程记录在 Provenance Graph 中。"""
    from aiida.orm import load_node

    from aiida_uranium_workflow.utils.parser_energy_time import fetch_abacus

    eentropy = {}
    num_atoms = {}
    eentropy_per_atom = {}
    total_energy = {}
    wall_time_seconds = {}
    status = {}

    for pk in child_pks.get_list():
        child = load_node(pk)
        param_dict = child.inputs.abacus.parameters.get_dict()
        input_block = param_dict.get("input", {})
        smear = input_block.get("smearing_method")
        sigma = input_block.get("smearing_sigma")
        label = f"smear_{smear}_sigma_{sigma}".replace(".", "_")

        # Capture exit status for every submitted child, even those
        # that did not finish OK — the report needs to surface
        # failures explicitly. ``exit_status`` is ``None`` for
        # unfinished processes; we render that as ``-1`` so the
        # Markdown table shows a non-empty cell.
        status[label] = (
            int(child.exit_status)
            if child.exit_status is not None
            else -1
        )

        if not child.is_finished_ok:
            continue

        # Energy + wall-time come from misc / log files via the shared
        # parser. We collect them independently of eentropy so a
        # failed eentropy parse doesn't drop the energy data.
        try:
            energy, wall_time = fetch_abacus(child)
            total_energy[label] = energy
            wall_time_seconds[label] = wall_time
        except Exception as e:  # noqa: BLE001 — defensive: misc missing
            total_energy[label] = f"Parsing failed: {str(e)}"
            wall_time_seconds[label] = None

        try:
            retrieved = child.outputs.retrieved
            structure = child.inputs.abacus.structure

            with retrieved.open("OUT.aiida/running_scf.log") as f:
                lines = f.readlines()

            eentropy_values = []
            for line in lines:
                if "E_entropy(-TS)" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        eentropy_eV = float(parts[2])
                        eentropy_values.append(eentropy_eV)

            if eentropy_values:
                last_eentropy = eentropy_values[-1]
                n_atoms = len(structure.sites)

                eentropy[label] = last_eentropy
                num_atoms[label] = n_atoms
                eentropy_per_atom[label] = last_eentropy / n_atoms
            else:
                eentropy[label] = None

        except Exception as e:
            eentropy[label] = f"Parsing failed: {str(e)}"

    return orm.Dict(
        {
            "eentropy": eentropy,
            "num_atoms": num_atoms,
            "eentropy_per_atom": eentropy_per_atom,
            "total_energy": total_energy,
            "wall_time_seconds": wall_time_seconds,
            "status": status,
        }
    )
