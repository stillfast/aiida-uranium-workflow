"""VASP input builder for the banddos workflow.

Wraps aiida-vasp's ``vasp.v2.bands`` (``VaspBandsWorkChain``): it runs the
SCF, generates the high-symmetry k-path, runs the non-self-consistent band
pass and — when asked — a constant-charge DOS pass. Outputs:

* ``band_structure`` — an ``orm.BandsData`` whose k-points carry the seekpath
  labels and whose ``fermi_level`` attribute holds E_F (eV);
* ``dos`` — an ``orm.ArrayData`` with ``energy`` / ``tdos`` / ``pdos``.

The protocol block (the ``vasp`` section of a ``parameters/banddos.yml``
preset) reuses the ABACUS ``band_settings`` vocabulary plus the aiida-vasp
extras::

    tdos:
      vasp:
        run_bands: true              # non-SCF band pass on the k-path
        run_dos: true                # constant-charge DOS pass
        run_proj_dos: false          # adds LORBIT=11 -> lm-decomposed pdos
        band_mode: seekpath-aiida    # or sumo
        band_kpoints_distance: 0.025 # 1/Å spacing along the path
        dos_kpoints_distance: 0.03   # 1/Å spacing of the DOS mesh
        symprec: 1.0e-5              # seekpath symmetry tolerance
        additional_band_analysis_parameters: {}

The SCF namespace comes from the usual per-backend preset file
(``parameters/vasp/scf.yml``): ``parameters.incar`` + ``potential_family`` /
``potential_mapping`` / ``kpoints_spacing`` (or ``kpoints_mesh``).
"""

from __future__ import annotations

from typing import Any, Dict

from ..base import AdaptedInputs, SoftwareAdapter, filter_scheduler_options


#: Keys copied verbatim into aiida-vasp's ``band_settings`` Dict.  Anything
#: else in the YAML block is either handled here (``run_bands`` /
#: ``run_dos`` / ``run_proj_dos``) or rejected by aiida-vasp's BandOptions.
BAND_SETTING_KEYS = (
    "symprec",
    "band_mode",
    "band_kpoints_distance",
    "line_density",
    "dos_kpoints_distance",
    "kpoints_per_split",
    "additional_band_analysis_parameters",
)

DEFAULT_BAND_SETTINGS: Dict[str, Any] = {
    "band_mode": "seekpath-aiida",
    "band_kpoints_distance": 0.025,
    "dos_kpoints_distance": 0.03,
    "symprec": 1.0e-5,
    "additional_band_analysis_parameters": {},
}


class VaspBandAdapter(SoftwareAdapter):
    """Translate a ParamBundle into VaspBandsWorkChain inputs."""

    name = "vasp"

    def _workchain_entry_point(self) -> str:
        return "vasp.v2.bands"

    # ---- protocol block ---------------------------------------------------

    def _block(self) -> Dict[str, Any]:
        """The ``vasp`` section of the banddos protocol preset."""
        block = self.workflow_data.get("vasp") or {}
        if not isinstance(block, dict):
            raise ValueError(
                "banddos protocol: the 'vasp' block must be a mapping "
                f"(got {type(block).__name__})"
            )
        return dict(block)

    def _flags(self) -> Dict[str, bool]:
        block = self._block()
        run_bands = bool(block.get("run_bands", True))
        run_dos = bool(block.get("run_dos", False))
        run_proj_dos = bool(block.get("run_proj_dos", False))
        if run_proj_dos:
            # Projected DOS lives in the DOS pass: it implies run_dos.
            run_dos = True
        if not run_bands and not run_dos:
            raise ValueError(
                "banddos protocol (vasp): run_bands and run_dos are both false — "
                "nothing to compute. Set at least one of them in the preset's "
                "'vasp' block of parameters/banddos.yml."
            )
        return {
            "run_bands": run_bands,
            "run_dos": run_dos,
            "run_proj_dos": run_proj_dos,
        }

    def _band_settings(self) -> Dict[str, Any]:
        """Build the ``band_settings`` Dict for ``VaspBandsWorkChain``.

        aiida-vasp drives the passes with ``only_dos`` (skip the band pass)
        and ``run_dos`` (run the DOS pass), so the ABACUS-style
        ``run_bands``/``run_dos`` pair maps onto them here.
        """
        block = self._block()
        flags = self._flags()

        settings = dict(DEFAULT_BAND_SETTINGS)
        for key in BAND_SETTING_KEYS:
            if key in block:
                settings[key] = block[key]
        settings["run_dos"] = flags["run_dos"]
        settings["only_dos"] = not flags["run_bands"]
        return settings

    # ---- AiiDA inputs -----------------------------------------------------

    def _build_workchain_inputs(self, structure) -> dict[str, Any]:
        from aiida import orm

        import copy

        entry = self.software_params
        options = filter_scheduler_options(
            self.metadata.get("options", {}),
            entry_point=self._workchain_entry_point(),
            backend=self.name,
        )
        flags = self._flags()

        scf: dict[str, Any] = {
            "code": orm.load_code(self.code_label),
            "parameters": orm.Dict(copy.deepcopy(entry.get("parameters", {}))),
            "potential_family": orm.Str(str(entry["potential_family"])),
            "potential_mapping": orm.Dict(dict=entry["potential_mapping"]),
        }
        if options:
            scf["calc"] = {"metadata": {"options": options}}

        if "kpoints_spacing" in entry:
            scf["kpoints_spacing"] = orm.Float(float(entry["kpoints_spacing"]))
        elif "kpoints_mesh" in entry:
            from aiida.plugins import DataFactory

            kpoints = DataFactory("core.array.kpoints")()
            kpoints.set_kpoints_mesh(list(entry["kpoints_mesh"]))
            scf["kpoints"] = kpoints

        inputs: dict[str, Any] = {"structure": structure, "scf": scf}

        if flags["run_proj_dos"]:
            # LORBIT=11 only for the DOS pass — the vasprun then carries the
            # per-atom lm-decomposed DOS that ``extract_vasp_pdos`` reads.
            # ``settings`` is passed explicitly because aiida-vasp only adds
            # the ``include_node`` for the DOS parser when the namespace is
            # left to it.
            inputs["dos"] = {
                # lower-case INCAR keys: aiida-vasp normalises them anyway
                # and warns about mixed case.
                "parameters": orm.Dict(dict={"incar": {"lorbit": 11}}),
                "settings": orm.Dict(
                    dict={
                        "parser_settings": {
                            "include_node": ["dos", "bands"],
                        }
                    }
                ),
            }

        return inputs

    def adapt(self, structure) -> AdaptedInputs:
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        inputs = self._build_workchain_inputs(structure)
        inputs["band_settings"] = orm.Dict(dict=self._band_settings())

        return AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )
