"""Audit: every input builder must forward the scheduler options from
``static/metadata.yml`` into the submitted WorkChain inputs.

Regression: the ABACUS banddos adapter used to drop them entirely — it
hardcoded an empty ``base.abacus.metadata`` while the generic
``_inject_options`` helper only looks for a *top-level* ``abacus`` /
``calc`` key, which the banddos layout (``base.abacus``) does not have.
The submitted job then ran with default resources (1 task), no queue
and no wall-time instead of the configured ``yeesuan`` / ``xinghe``
settings.

This test parametrises **every** adapter: with a canonical metadata
``options`` block (``queue_name: q_test`` ...) the options must appear
somewhere in the final adapted inputs.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida_uranium_workflow.input_builders import (
    base_workchain,
    banddos,
    convergence,
    defects,
    elastic,
    eos,
    magmom,
    phonopy,
    relax,
    smear,
)

#: Canonical scheduler options — every adapter must forward them.
TEST_OPTIONS = {
    "resources": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 56,
        "tot_num_mpiprocs": 56,
    },
    "max_wallclock_seconds": 7200,
    "queue_name": "q_test",
    "withmpi": True,
}

_ABACUS_SCF = {"parameters": {"input": {"calculation": "scf"}}}
_VASP_SCF = {
    "parameters": {"incar": {}},
    "potential_family": "test_family",
    "potential_mapping": {},
}
_FLEUR_SCF = {
    "wf_parameters": {"mode": "density", "itmax_per_run": 100},
    "calc_parameters": {"comp": {"kmax": 7.0}},
}

#: (case_id, adapter_factory) — factory builds the adapter given
#: (code_label, inpgen_label).
ADAPTER_FACTORIES = [
    ("base_abacus", lambda c, i: base_workchain.AbacusBaseWorkChainAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("base_vasp", lambda c, i: base_workchain.VaspBaseWorkChainAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("smear_abacus", lambda c, i: smear.AbacusAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("smear_vasp", lambda c, i: smear.VaspAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("convergence_abacus", lambda c, i: convergence.AbacusConvergenceAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("convergence_vasp", lambda c, i: convergence.VaspConvergenceAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("magmom_abacus", lambda c, i: magmom.AbacusMagmomAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("magmom_vasp", lambda c, i: magmom.VaspMagmomAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("magmom_fleur", lambda c, i: magmom.FleurMagmomAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={
            "magmom_lists": {"fleur": {"magmom_list": [
                {"label": "FM", "bmu": 1.0, "inpxml_changes": []}]}}},
        extra_codes={"inpgen": i})),
    ("banddos_abacus", lambda c, i: banddos.AbacusBandAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("banddos_fleur", lambda c, i: banddos.FleurBandAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={
            "fleur": {"band_wf": {"mode": "band"}, "dos_wf": {"mode": "dos"}}},
        extra_codes={"inpgen": i})),
    ("elastic_abacus", lambda c, i: elastic.AbacusElasticAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("elastic_vasp", lambda c, i: elastic.VaspElasticAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("elastic_fleur", lambda c, i: elastic.FleurElasticAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={},
        extra_codes={"inpgen": i})),
    ("eos_abacus", lambda c, i: eos.AbacusEosAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("eos_fleur", lambda c, i: eos.FleurEosAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={},
        extra_codes={"inpgen": i})),
    ("relax_abacus", lambda c, i: relax.AbacusRelaxAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("relax_fleur", lambda c, i: relax.FleurRelaxAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={},
        extra_codes={"inpgen": i})),
    ("phonopy_abacus", lambda c, i: phonopy.AbacusPhonopyAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("defects_abacus", lambda c, i: defects.AbacusDefectsAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={
            "abacus": {"defect": {"type": "vacancy", "site_index": 0,
                                  "element": "U", "label": "vac"}}},
        extra_codes={})),
    ("defects_fleur", lambda c, i: defects.FleurDefectsAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={
            "fleur": {"defect": {"type": "vacancy", "site_index": 0,
                                 "element": "U", "label": "vac"}}},
        extra_codes={"inpgen": i})),
]


def _collect_values(obj, needle):
    """Recursively collect keys whose value equals ``needle``."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value == needle:
                found.append(key)
            found += _collect_values(value, needle)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found += _collect_values(value, needle)
    elif hasattr(obj, "get_dict"):
        try:
            found += _collect_values(obj.get_dict(), needle)
        except Exception:
            pass
    return found


def test_every_adapter_forwards_scheduler_options(aiida_profile, aiida_localhost):
    """With ``queue_name: q_test`` in the metadata options, the queue name
    must survive into the adapted inputs of *every* adapter."""
    from aiida import orm
    from ase.build import bulk

    computer = aiida_localhost
    structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))

    failures = []
    for case_id, factory in ADAPTER_FACTORIES:
        # One code per adapter avoids label collisions in the shared profile.
        label = f"opt_{case_id}"
        code = orm.InstalledCode(
            label=label, computer=computer, filepath_executable="/bin/true"
        ).store()
        code_label = f"{label}@{computer.label}"
        inpgen = orm.InstalledCode(
            label=f"{label}_inpgen", computer=computer, filepath_executable="/bin/true"
        ).store()
        inpgen_label = f"{label}_inpgen@{computer.label}"

        adapter = factory(code_label, inpgen_label)
        adapter.metadata = {"options": dict(TEST_OPTIONS)}
        try:
            adapted = adapter.adapt(structure)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the audit
            failures.append(f"{case_id}: adapt raised {type(exc).__name__}: {exc}")
            continue

        found = _collect_values(adapted.inputs, "q_test")
        if not found:
            failures.append(f"{case_id}: scheduler options dropped from inputs")

    assert not failures, "\n".join(failures)
