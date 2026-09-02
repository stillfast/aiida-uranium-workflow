"""ABACUS input builder for the banddos workflow.

Translates a :class:`ParamBundle` into the ``inputs`` dict expected by
:func:`aiida.plugins.WorkflowFactory('abacus.band')`
(:class:`AbacusBandWorkChain`). Unlike smear / convergence / magmom this
workflow is **not** a parameter sweep — one WorkChain is submitted per
(preset, structure). The ``band_settings`` namespace (``run_bands``,
``run_dos``, ``band_mode``, distances, …) is taken verbatim from the
``banddos.yml`` protocol section loaded by the orchestrator.
"""
from __future__ import annotations

from ..base import AdaptedInputs, SoftwareAdapter
from typing import Any, List, Tuple


class AbacusBandAdapter(SoftwareAdapter):
    """Translate a ParamBundle into AbacusBandWorkChain inputs."""

    name = "abacus"

    def _workchain_entry_point(self) -> str:
        return "abacus.band"

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        params = copy.deepcopy(self.software_params["parameters"])
        options = self.metadata.get("options", {})

        inputs: dict[str, Any] = {
            "structure": structure,
            # ``AbacusBandWorkChain`` exposes the ``base`` namespace from
            # ``AbacusBaseWorkChain``. The block below mirrors the
            # ``make_base_namespace`` helper used in
            # ``test_abacus_band.py`` so a user can submit the band
            # workflow without hand-assembling every input.
            "base": {
                "abacus": {
                    "code": orm.load_code(self.code_label),
                    "parameters": orm.Dict(params),
                    # Scheduler options (resources / queue_name /
                    # max_wallclock_seconds / withmpi) from
                    # ``static/metadata.yml`` land here — the generic
                    # ``_inject_options`` in the base adapter only looks
                    # for a *top-level* ``abacus`` / ``calc`` key, which
                    # the banddos layout (``base.abacus``) does not have.
                    "metadata": {"options": options} if options else {},
                },
            },
        }
        if "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(
                list(self.software_params["kpoints_mesh"])
            )
            inputs["base"]["kpoints"] = kpoints_mesh
        if "pseudo_family" in self.software_params:
            inputs["base"]["pseudo_family"] = orm.Str(
                str(self.software_params["pseudo_family"])
            )
        return inputs

    def _prepare_workflow_inputs(self) -> Tuple[List, List[float]]:
        """Unused for banddos — the parent ``adapt()`` only forwards
        ``smear`` / ``sigma`` lists, which the band WorkChain doesn't
        consume. The band-specific data lives in
        :meth:`_band_settings` and is injected from
        :meth:`adapt` below.
        """
        return [], []

    def _band_settings(self) -> dict[str, Any]:
        """Return the ``band_settings`` dict to forward to the WorkChain.

        The schema mirrors the keys declared in
        ``AbacusBandWorkChain.spec.input.band_settings``:

        * ``run_bands`` / ``run_dos`` (bool)
        * ``band_mode`` (``"seekpath-aiida"`` / ``"brillouin-zone-database"`` / ``"manual"``)
        * ``band_kpoints_distance`` (float) — reciprocal-Å spacing for
          the auto k-path.
        * ``dos_kpoints_distance`` (float) — spacing for the DOS mesh.
        * ``symprec`` (float) — symmetry-detection tolerance.
        * ``additional_band_analysis_parameters`` (dict)

        Anything missing from the YAML falls back to the same defaults
        as ``test_abacus_band.py``.
        """
        defaults: dict[str, Any] = {
            "run_bands": True,
            "run_dos": True,
            "band_mode": "seekpath-aiida",
            "band_kpoints_distance": 0.02,
            "dos_kpoints_distance": 0.1,
            "symprec": 1e-5,
            "additional_band_analysis_parameters": {},
        }
        band_data = self.workflow_data.get("band_settings", {}) or {}
        merged = dict(defaults)
        if isinstance(band_data, dict):
            merged.update(band_data)
        return merged

    def adapt(self, structure) -> AdaptedInputs:
        """Compose the final AiiDA inputs + workchain class.

        Overrides :meth:`SoftwareAdapter.adapt` so the band-specific
        ``band_settings`` namespace is forwarded (the parent only knows
        about ``smear`` / ``sigma``).
        """
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        band_settings = self._band_settings()

        inputs = self._build_workchain_inputs(structure)
        if band_settings:
            inputs["band_settings"] = orm.Dict(dict=band_settings)

        self._inject_options(inputs, options)

        return AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )