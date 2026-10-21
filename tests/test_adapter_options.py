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
_QE_SCF = {"pw": {"parameters": {"CONTROL": {"calculation": "scf"}, "SYSTEM": {"ecutwfc": 60}}}}

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
    ("elastic_qe", lambda c, i: elastic.QeElasticAdapter(
        code_label=c, software_params=_QE_SCF, metadata={}, workflow_data={},
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
    ("eos_qe", lambda c, i: eos.QeEosAdapter(
        code_label=c, software_params=_QE_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("relax_abacus", lambda c, i: relax.AbacusRelaxAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("relax_vasp", lambda c, i: relax.VaspRelaxAdapter(
        code_label=c, software_params=_VASP_SCF, metadata={}, workflow_data={
            "vasp": {"relax_type": "volume"}},
        extra_codes={})),
    ("relax_fleur", lambda c, i: relax.FleurRelaxAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={},
        extra_codes={"inpgen": i})),
    ("relax_qe", lambda c, i: relax.QeRelaxAdapter(
        code_label=c, software_params=_QE_SCF, metadata={}, workflow_data={},
        extra_codes={})),
    ("phonopy_abacus", lambda c, i: phonopy.AbacusPhonopyAdapter(
        code_label=c, software_params=_ABACUS_SCF, metadata={}, workflow_data={
            "phonopy": {"supercell_matrix": [2, 2, 2]}},
        extra_codes={})),
    ("phonopy_fleur", lambda c, i: phonopy.FleurPhonopyAdapter(
        code_label=c, software_params=_FLEUR_SCF, metadata={}, workflow_data={
            "phonopy": {"supercell_matrix": [2, 2, 2]}},
        extra_codes={"inpgen": i})),
    ("phonopy_qe", lambda c, i: phonopy.QePhonopyAdapter(
        code_label=c, software_params=_QE_SCF, metadata={}, workflow_data={
            "phonopy": {"qe": {"qpoints_mesh": [2, 2, 2]}}},
        extra_codes={
            key: f"{c.split('@')[0]}_{key}@{c.split('@')[1]}"
            for key in ("ph", "q2r", "matdyn")
        })),
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
        # The QE phonopy (DFPT) adapter needs ph / q2r / matdyn codes too.
        for suffix in ("ph", "q2r", "matdyn"):
            orm.InstalledCode(
                label=f"{label}_{suffix}",
                computer=computer,
                filepath_executable="/bin/true",
            ).store()

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


# ---------------------------------------------------------------------------
# Foreign keys: metadata presets are shared across backends, so they may
# carry options that belong to another plugin (``yunhe`` has
# ``optimize_resources``, a FLEUR/QE-style key no CalcJob knows).
# ``metadata.options`` is *not* a dynamic namespace, so forwarding one of
# those makes the submission die with::
#
#     ValueError: Error occurred validating port
#     'inputs.base.abacus.metadata.options': Unexpected ports
#     {'optimize_resources': False}, for a non dynamic namespace
# ---------------------------------------------------------------------------

#: The ``yunhe`` preset's foreign key, on top of the canonical options.
FOREIGN_OPTIONS = {**TEST_OPTIONS, "optimize_resources": False}


def _collect_metadata_options(obj, out: list) -> None:
    """Collect every ``…metadata.options`` dict reachable in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "metadata" and isinstance(value, dict) and isinstance(value.get("options"), dict):
                out.append(value["options"])
            _collect_metadata_options(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_metadata_options(value, out)
    elif hasattr(obj, "get_dict"):
        try:
            _collect_metadata_options(obj.get_dict(), out)
        except Exception:  # noqa: BLE001 — not every node is a Dict
            pass


def _contains_key(obj, name: str) -> bool:
    """True when ``name`` appears as a key anywhere in ``obj``."""
    if isinstance(obj, dict):
        return name in obj or any(_contains_key(value, name) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(value, name) for value in obj)
    if hasattr(obj, "get_dict"):
        try:
            return _contains_key(obj.get_dict(), name)
        except Exception:  # noqa: BLE001 — not every node is a Dict
            return False
    return False


def test_filter_scheduler_options_is_per_backend():
    """Each backend keeps exactly the keys its own namespace accepts.

    ``optimize_resources`` is a **FLEUR workchain** option
    (``aiida_fleur`` pops it from the plugin ``options`` ``Dict``), not a
    CalcJob option: neither ABACUS (25 keys) nor FLEUR's own CalcJob
    (26 keys) declares it, so it is dropped on the ``metadata.options``
    path for both.
    """
    from aiida_uranium_workflow.input_builders.base import (
        filter_scheduler_options,
        scheduler_option_keys,
    )

    # ABACUS: the CalcJob namespace, 25 standard keys.
    keys = scheduler_option_keys("abacus.base", "abacus")
    assert keys is not None and "optimize_resources" not in keys
    assert "queue_name" in keys
    # FLEUR's own CalcJob namespace — also without the workchain-only key.
    fleur_keys = scheduler_option_keys("fleur.fleur", "fleur")
    assert fleur_keys is not None and "optimize_resources" not in fleur_keys

    for entry_point, backend in (("abacus.base", "abacus"), ("fleur.fleur", "fleur")):
        kept = filter_scheduler_options(
            FOREIGN_OPTIONS, entry_point=entry_point, backend=backend
        )
        assert "optimize_resources" not in kept
        assert kept["queue_name"] == "q_test"
        assert kept["resources"] == TEST_OPTIONS["resources"]

    assert filter_scheduler_options(None) == {}
    assert filter_scheduler_options({}) == {}


def test_every_adapter_keeps_only_its_own_options(aiida_profile, aiida_localhost, capsys):
    """Per-backend audit of the metadata options every adapter forwards.

    Regression for ``banddos/14ve``: the ``yunhe`` metadata preset carries
    ``optimize_resources``, which reached ``base.abacus.metadata.options``
    verbatim and made ``aiida-uranium run`` abort with a port-validation
    ``ValueError`` before anything was submitted.

    Two invariants, both backend-specific:

    * a CalcJob ``metadata.options`` dict may only contain keys of *its*
      namespace (ABACUS: 25, FLEUR's CalcJob: 26) — no foreign keys;
    * FLEUR keeps ``optimize_resources``, because its workchains take the
      options through their plugin ``options`` ``Dict`` input, where the
      key is meaningful.  ABACUS drops it.
    """
    from aiida import orm
    from ase.build import bulk

    from aiida_uranium_workflow.input_builders.base import scheduler_option_keys

    computer = aiida_localhost
    structure = orm.StructureData(ase=bulk("U", "bcc", a=3.45))

    failures = []
    for case_id, factory in ADAPTER_FACTORIES:
        label = f"fopt_{case_id}"
        orm.InstalledCode(
            label=label, computer=computer, filepath_executable="/bin/true"
        ).store()
        orm.InstalledCode(
            label=f"{label}_inpgen", computer=computer, filepath_executable="/bin/true"
        ).store()
        for suffix in ("ph", "q2r", "matdyn"):
            orm.InstalledCode(
                label=f"{label}_{suffix}", computer=computer, filepath_executable="/bin/true"
            ).store()

        adapter = factory(f"{label}@{computer.label}", f"{label}_inpgen@{computer.label}")
        adapter.metadata = {"options": dict(FOREIGN_OPTIONS)}
        try:
            adapted = adapter.adapt(structure)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the audit
            failures.append(f"{case_id}: adapt raised {type(exc).__name__}: {exc}")
            continue

        allowed = scheduler_option_keys(
            adapter._workchain_entry_point(), adapter.name
        )
        option_dicts: list = []
        _collect_metadata_options(adapted.inputs, option_dicts)
        for opts in option_dicts:
            if allowed is not None:
                foreign = sorted(set(opts) - allowed)
                if foreign:
                    failures.append(
                        f"{case_id} ({adapter.name}): foreign keys reached "
                        f"metadata.options: {foreign}"
                    )
            if "queue_name" not in opts:
                failures.append(f"{case_id}: metadata.options lost the scheduler options")

        forwards_foreign = _contains_key(adapted.inputs, "optimize_resources")
        if adapter.name == "fleur" and not forwards_foreign:
            failures.append(
                f"{case_id}: FLEUR lost `optimize_resources` (it reads it from its "
                "own options Dict)"
            )
        if adapter.name != "fleur" and forwards_foreign:
            failures.append(f"{case_id} ({adapter.name}): foreign key forwarded")

    assert capsys.readouterr().err == "", "过滤是静默的，不应再打印提示"
    assert not failures, "\n".join(failures)
