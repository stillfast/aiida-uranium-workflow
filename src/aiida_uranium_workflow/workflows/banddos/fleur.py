"""FLEUR band + DOS WorkChain.

This WorkChain runs ``FleurBandDosWorkChain`` twice in parallel — once
with ``wf_parameters['mode'] = 'band'`` and once with
``mode = 'dos'`` — both reusing the same SCF inputs. It then exposes
the per-mode outputs plus a combined ``output_parameters`` carrying
both Fermi levels (eV).

Layout::

    inputs
    ├── scf               (Dict) — SCF namespace shared by both runs
    ├── band_inputs       (Dict) — band-mode ``wf_parameters`` overrides
    ├── dos_inputs        (Dict) — DOS-mode ``wf_parameters`` overrides
    ├── structure, codes, options ... (mirrors ``test_fleur_band.py``)
    outputs
    ├── band_structure    (XyData)  — band k-points + energies
    ├── dos               (XyData)  — DOS arrays
    └── output_parameters (Dict)    — {fermi_level_band, fermi_level_dos,
                                       bandgap_band, bandgap_dos, mode, …}

Usage example (mirrors ``test_fleur_band.py``'s ``submit_banddos``)::

    wc = submit(FleurBandAndDosWorkChain, structure=structure,
                fleur=load_code('fleur@yunhe'),
                inpgen=load_code('inpgen@yunhe'),
                scf=Dict(dict=scf_namespace),
                band_wf=Dict(dict={'kpath': 'seek',
                                   'kpoints_distance': 0.02,
                                   'proj': {...}}),
                dos_wf=Dict(dict={'kpath': 'auto',
                                  'kpoints_distance': 0.1,
                                  'energy_range': {...}}))
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

HA_TO_EV = 27.211386245988  # FLEUR reports energies in Hartree.

# Underlying plugin workflow we run twice.
FleurBandDosWorkChain = WorkflowFactory("fleur.banddos")


class FleurBandAndDosWorkChain(WorkChain):
    """Run FLEUR band + DOS and surface both sets of outputs.

    Outline: submit_children → gather_results

    Exit codes
    ----------
    * 0   ``SUCCESS``          — both band and DOS calculations finished.
    * 300 ``ERROR_CHILD``      — one of the children failed.
    * 305 ``ERROR_PARSER``     — could not read the per-run output ports.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- Top-level inputs ------------------------------------------------
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="Crystal structure (passed to both runs).",
        )
        spec.input(
            "fleur",
            valid_type=orm.InstalledCode,
            help="FLEUR code (passed to both runs).",
        )
        spec.input(
            "inpgen",
            valid_type=orm.InstalledCode,
            required=False,
            help="inpgen code (used by the SCF namespace).",
        )
        spec.input(
            "options",
            valid_type=orm.Dict,
            help="Parallel scheduler options for FLEUR.",
        )
        spec.input(
            "options_inpgen",
            valid_type=orm.Dict,
            required=False,
            help="Serial scheduler options for inpgen.",
        )
        spec.input_namespace("scf", help=(
            "SCF namespace forwarded to each child FleurBandDosWorkChain. "
            "Must contain 'wf_parameters' (Dict), 'calc_parameters' "
            "(Dict), and 'structure' (StructureData). The adapter "
            "fills in 'fleur' / 'inpgen' / 'options_inpgen' from "
            "input.json's top-level 'code' mapping before forwarding "
            "the namespace to each child run."
        ))
        spec.input(
            "scf.wf_parameters",
            valid_type=orm.Dict,
            help="FleurBaseWorkChain settings (mode='density', …).",
        )
        spec.input(
            "scf.calc_parameters",
            valid_type=orm.Dict,
            help="FLEUR input parameters (kmax / kpt / atom / …).",
        )
        spec.input(
            "scf.structure",
            valid_type=orm.StructureData,
            help="Crystal structure (SCF namespace).",
        )
        spec.input(
            "scf.fleur",
            valid_type=orm.InstalledCode,
            required=False,
            help=(
                "FLEUR code forwarded to the child runs. The adapter "
                "fills this in from input.json['code']['fleur']."
            ),
        )
        spec.input(
            "scf.inpgen",
            valid_type=orm.InstalledCode,
            required=False,
            help=(
                "inpgen code forwarded to the child runs. The adapter "
                "fills this in from input.json['code']['inpgen']. "
                "Required by FleurBandDosWorkChain's SCF namespace."
            ),
        )
        spec.input(
            "scf.options_inpgen",
            valid_type=orm.Dict,
            required=False,
            help="Serial scheduler options for inpgen.",
        )
        spec.input(
            "band_wf",
            valid_type=orm.Dict,
            help="wf_parameters for the band run (mode='band').",
        )
        spec.input(
            "dos_wf",
            valid_type=orm.Dict,
            help="wf_parameters for the DOS run (mode='dos').",
        )

        # ---- Outputs ---------------------------------------------------------
        spec.output(
            "band_structure",
            valid_type=orm.XyData,
            required=True,
            help="Bands XyData from the band-mode child.",
        )
        spec.output(
            "dos",
            valid_type=orm.XyData,
            required=True,
            help="DOS XyData from the DOS-mode child.",
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            required=True,
            help=(
                "Combined summary: fermi_level_band / fermi_level_dos (eV), "
                "bandgap_band / bandgap_dos, modes, pk maps, statuses."
            ),
        )

        spec.outline(
            cls.submit_children,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "band + DOS completed successfully.")
        spec.exit_code(
            300, "ERROR_CHILD", "A child FleurBandDosWorkChain failed."
        )
        spec.exit_code(
            305,
            "ERROR_PARSER",
            "Failed to parse band / DOS outputs from the children.",
        )

    # ------------------------------------------------------------------
    # Outline steps
    # ------------------------------------------------------------------

    def _build_run_inputs(self, wf_parameters: dict, mode: str) -> dict:
        """Compose one ``FleurBandDosWorkChain`` inputs dict.

        ``wf_parameters`` already carries ``mode`` / ``kpath`` /
        ``proj`` / ``energy_range``; we only make sure ``mode`` matches
        the requested branch.
        """
        wf = dict(wf_parameters)
        wf["mode"] = mode
        # ``kpath`` defaults mirror test_fleur_band.py
        wf.setdefault("kpath", "seek" if mode == "band" else "auto")

        # Copy the SCF namespace into a plain dict so we can extend
        # it with ``fleur`` / ``inpgen`` / ``options_inpgen`` /
        # ``structure`` before forwarding to the child WorkChain.
        scf = dict(self.inputs.scf)
        scf["fleur"] = self.inputs.fleur
        if "inpgen" in self.inputs:
            scf["inpgen"] = self.inputs.inpgen
        if "options_inpgen" in self.inputs:
            scf["options_inpgen"] = self.inputs.options_inpgen
        # The SCF namespace must always carry the structure.
        scf["structure"] = self.inputs.structure

        inputs: dict = {
            "fleur":         self.inputs.fleur,
            "wf_parameters": orm.Dict(wf),
            "options":       self.inputs.options,
            "scf":           scf,
            "metadata": {
                "label":       f"fleur-banddos-{mode}",
                "description": f"FLEUR {mode} run (combined band+DOS wc)",
            },
        }
        return inputs

    def submit_children(self):
        """Submit the band and DOS ``FleurBandDosWorkChain`` instances."""
        try:
            band_wf = self.inputs.band_wf.get_dict()
            dos_wf = self.inputs.dos_wf.get_dict()
        except (AttributeError, KeyError) as exc:
            self.report(f"missing band_wf / dos_wf inputs: {exc}")
            return self.exit_codes.ERROR_PARSER

        try:
            band_node = self.submit(
                FleurBandDosWorkChain,
                **self._build_run_inputs(band_wf, "band"),
            )
            dos_node = self.submit(
                FleurBandDosWorkChain,
                **self._build_run_inputs(dos_wf, "dos"),
            )
        except Exception as exc:
            self.report(f"submission failed: {exc}")
            return self.exit_codes.ERROR_CHILD

        self.report(f"submitted band run pk={band_node.pk}")
        self.report(f"submitted DOS  run pk={dos_node.pk}")
        self.to_context(band_run=band_node, dos_run=dos_node)

    def gather_results(self):
        """Forward band / DOS outputs and merge ``output_parameters``."""
        band_run = getattr(self.ctx, "band_run", None)
        dos_run = getattr(self.ctx, "dos_run", None)

        if band_run is None or dos_run is None:
            return self.exit_codes.ERROR_PARSER

        if not band_run.is_finished_ok or not dos_run.is_finished_ok:
            return self.exit_codes.ERROR_CHILD

        try:
            band_bands = band_run.outputs.output_banddos_wc_bands
            dos_dos = dos_run.outputs.output_banddos_wc_dos
        except AttributeError as exc:
            self.report(
                f"missing expected outputs from children: {exc}; "
                f"band outputs: {sorted(band_run.outputs.keys())}, "
                f"dos outputs: {sorted(dos_run.outputs.keys())}"
            )
            return self.exit_codes.ERROR_PARSER

        self.out("band_structure", band_bands)
        self.out("dos", dos_dos)

        self.out(
            "output_parameters",
            _combine_outputs(band_run=band_run, dos_run=dos_run),
        )
        return None  # SUCCESS


# ---------------------------------------------------------------------------
# calcfunction: merge the two ``output_banddos_wc_para`` dicts and convert
# Hartree → eV for the Fermi levels.
# ---------------------------------------------------------------------------


@calcfunction
def _combine_outputs(band_run, dos_run):
    """Return a Dict combining both runs' summary dicts + metadata."""
    from aiida.orm import load_node

    band_para = band_run.outputs.output_banddos_wc_para.get_dict()
    dos_para = dos_run.outputs.output_banddos_wc_dos  # kept for symmetry

    dos_para_dict = dos_run.outputs.output_banddos_wc_para.get_dict()

    def _fermi_ev(para: dict) -> float | None:
        """Pull ``fermi_energy_band`` (Hartree) → eV; ``None`` if missing."""
        for key in ("fermi_energy_band", "fermi_energy_scf"):
            if key not in para:
                continue
            units = str(para.get("fermi_energy_units", "Htr")).strip().lower()
            value = float(para[key])
            if units in ("ha", "hartree", "htr"):
                return value * HA_TO_EV
            if units == "ev":
                return value
            # Unknown unit — assume Hartree.
            return value * HA_TO_EV
        return None

    combined = {
        # Band-mode summary (Hartree fields stay as-is; convenience eV
        # versions are added below).
        "band": dict(band_para),
        "dos": dict(dos_para_dict),
        # Convenience fields: Fermi levels in eV (the script-friendly
        # unit used by the FLEUR UI and our plotting helpers).
        "fermi_level_band": _fermi_ev(band_para),
        "fermi_level_dos": _fermi_ev(dos_para_dict),
        "fermi_level_units": "eV",
        # Diff between band / DOS runs (bandgap_scf should be 0.0 for
        # both runs when re-using the same SCF density).
        "bandgap_band": band_para.get("bandgap_band"),
        "bandgap_dos": dos_para_dict.get("bandgap_band"),
        "diff_bandgap": (
            (band_para.get("bandgap_band") or 0.0)
            - (dos_para_dict.get("bandgap_band") or 0.0)
        ),
        # Provenance: pks / uuids of the two child workchains.
        "band_run_pk": band_run.pk,
        "band_run_uuid": band_run.uuid,
        "dos_run_pk": dos_run.pk,
        "dos_run_uuid": dos_run.uuid,
    }
    return orm.Dict(combined)