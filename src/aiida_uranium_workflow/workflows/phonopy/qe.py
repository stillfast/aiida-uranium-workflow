"""QE (pw.x + ph.x) phonon WorkChain — DFPT via the aiida-qe toolchain.

Computes the phonon band structure and DOS of a structure with the
**density-functional perturbation theory** chain that aiida-qe ships —
the QE-native counterpart of the frozen-phonon ABACUS / FLEUR phonopy
workflows:

1. ``prepare`` — run the aiida-qe ``seekpath_structure_analysis``
   calcfunction on the input structure: the chain then runs on the
   returned *primitive* cell (so the ph.x q-points and the matdyn band
   path live in the same reciprocal space) and the explicit high
   symmetry q-point path is kept for matdyn.
2. ``run_scf`` — one ``PwBaseWorkChain`` (``quantumespresso.pw.base``)
   SCF of the primitive cell (``base`` inputs from
   ``parameters/qe/scf.yml``); its charge density is the parent of ph.x.
3. ``run_ph`` — one ``PhBaseWorkChain`` (``quantumespresso.ph.base``):
   ph.x (DFPT) on a gamma-centred **q-mesh** (``qpoints_mesh``) reading
   the SCF density through ``parent_folder`` (same computer). The
   dynamical matrices stay in the remote ``DYN_MAT`` folder.
4. ``run_q2r`` — one ``Q2rBaseWorkChain`` (``quantumespresso.q2r.base``)
   turning the q-mesh dynamical matrices into real-space force
   constants (``ForceConstantsData``).
5. ``run_matdyn`` — two ``MatdynBaseWorkChain``
   (``quantumespresso.matdyn.base``) runs on the same force constants:
   *band* — frequencies along the seekpath high-symmetry q path; *dos* —
   a phonon DOS on a dense q mesh. ``asr='crystal'`` is enforced at
   interpolation time so the acoustic branches anchor at Γ = 0.
6. ``gather_results`` — parse matdyn's raw frequency file (the aiida-qe
   5.0 parser misaligns QE 7.5's ``&plot`` layout and its BandsData is
   garbage — see :func:`_parse_matdyn_frequencies`), rescale the DOS
   cm⁻¹ → THz, forward the force constants and build the summary
   (backend "qe").

No supercells and no phonopy code are needed — this is pure DFPT.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.plugins import WorkflowFactory

import numpy as np

from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)
from aiida_quantumespresso.data.force_constants import ForceConstantsData

#: The four plugin WorkChains that make up the chain.
PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
PhBaseWorkChain = WorkflowFactory("quantumespresso.ph.base")
Q2rBaseWorkChain = WorkflowFactory("quantumespresso.q2r.base")
MatdynBaseWorkChain = WorkflowFactory("quantumespresso.matdyn.base")

#: cm⁻¹ → THz (QE matdyn writes cm⁻¹ by default; everything else in this
#: workflow family — and the phonopy reports / figures — uses THz).
_INVCM_TO_THZ = 0.0299792458
_THZ_TO_INVCM = 1.0 / _INVCM_TO_THZ

#: seekpath reports high-symmetry labels in ASCII ("GAMMA", ...); the
#: figures use the conventional Greek symbols.
_GREEK_LABELS = {
    "GAMMA": "Γ",
    "DELTA": "Δ",
    "LAMBDA": "Λ",
    "SIGMA": "Σ",
}


class QePhonopyWorkChain(WorkChain):
    """Compute phonon bands / DOS with the QE DFPT chain (ph.x).

    Outline: prepare → run_scf → run_ph → run_q2r → run_matdyn →
    gather_results.

    Exit codes
    ----------
    * 0   ``SUCCESS``      — phonon calculation finished.
    * 300 ``ERROR_CHILD``  — a chain child (SCF / ph / q2r / matdyn) failed.
    * 305 ``ERROR_PARSER`` — the chain outputs could not be read.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData, required=True)
        spec.input_namespace(
            "base",
            dynamic=True,
            help="PwBaseWorkChain inputs for the parent SCF (pw / kpoints from scf.yml).",
        )

        # ---- DFPT chain: codes + knobs ------------------------------------
        spec.input("ph_code", valid_type=orm.InstalledCode, required=True)
        spec.input(
            "ph_parameters",
            valid_type=orm.Dict,
            required=True,
            help="ph.x INPUTPH parameters (tr2_ph / alpha_mix / nmix_ph / ...).",
        )
        spec.input(
            "qpoints_mesh",
            valid_type=orm.List,
            required=True,
            help="Gamma-centred q-point mesh of the ph.x run, e.g. [4, 4, 4].",
        )
        spec.input("q2r_code", valid_type=orm.InstalledCode, required=True)
        spec.input("matdyn_code", valid_type=orm.InstalledCode, required=True)
        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler options (resources / wall-time / queue) for the ph, q2r and matdyn children.",
        )
        spec.input(
            "q2r_parameters",
            valid_type=orm.Dict,
            required=False,
            help="q2r.x INPUT namelist overrides (fildyn / flfrc are managed by the plugin).",
        )
        spec.input(
            "matdyn_parameters",
            valid_type=orm.Dict,
            required=False,
            help="matdyn.x INPUT namelist overrides (the dos flag is managed by the workflow).",
        )
        spec.input("dos", valid_type=orm.Bool, required=False, default=lambda: orm.Bool(True))
        spec.input(
            "dos_mesh",
            valid_type=orm.List,
            required=False,
            help="Q-point mesh of the matdyn DOS run, e.g. [21, 21, 21].",
        )

        spec.output(
            "phonon_bands",
            valid_type=orm.BandsData,
            required=True,
            help="Phonon band structure (THz) along the seekpath high-symmetry path.",
        )
        spec.output(
            "total_phonon_dos",
            valid_type=orm.XyData,
            required=False,
            help="Total phonon DOS (THz) from the matdyn DOS run.",
        )
        spec.output(
            "force_constants",
            # aiida-qe's ForceConstantsData is a SinglefileData, *not* an
            # ArrayData — validate against the concrete plugin type.
            valid_type=(ForceConstantsData, orm.ArrayData),
            required=False,
            help="Real-space force constants (ForceConstantsData) from q2r.",
        )
        spec.output("output_parameters", valid_type=orm.Dict, required=True)

        spec.outline(
            cls.prepare,
            cls.run_scf,
            cls.run_ph,
            cls.run_q2r,
            cls.run_matdyn,
            cls.gather_results,
        )

        spec.exit_code(0, "SUCCESS", "Phonon calculation completed.")
        spec.exit_code(300, "ERROR_CHILD", "A chain child (SCF / ph / q2r / matdyn) failed.")
        spec.exit_code(305, "ERROR_PARSER", "Failed to read the chain outputs.")

    # ------------------------------------------------------------------
    # outline steps
    # ------------------------------------------------------------------

    def prepare(self):
        """Primitivise the structure and compute the band q-point path."""
        result = seekpath_structure_analysis(structure=self.inputs.structure)
        self.ctx.primitive_structure = result["primitive_structure"]
        self.ctx.band_kpoints = result["explicit_kpoints"]
        self.report(
            f"primitive cell: {self.ctx.primitive_structure.get_formula()} "
            f"(input {self.inputs.structure.get_formula()}); "
            f"band path with {len(self.ctx.band_kpoints.get_kpoints())} q-points"
        )

    def _scf_inputs(self):
        """The PwBaseWorkChain inputs of the parent SCF on the primitive cell."""
        base = dict(self.inputs.base) if "base" in self.inputs else {}
        pw = dict(base.get("pw", {}))
        params = pw.get("parameters", {})
        if hasattr(params, "get_dict"):
            params = params.get_dict()
        control = params.setdefault("CONTROL", {})
        control["calculation"] = "scf"
        pw["parameters"] = orm.Dict(dict=params)
        pw["structure"] = self.ctx.primitive_structure
        child_inputs = dict(base)
        child_inputs["pw"] = pw
        child_inputs["metadata"] = {
            "label": "phonon-scf",
            "description": "PwBaseWorkChain SCF of the primitive cell (parent of ph.x)",
        }
        return child_inputs

    def run_scf(self):
        """Submit the parent pw.x SCF of the primitive cell."""
        running = self.submit(PwBaseWorkChain, **self._scf_inputs())
        self.report(f"submitted parent SCF pk={running.pk}")
        self.to_context(scf=running)

    def _child_options(self):
        """Scheduler options for the ph / q2r / matdyn children."""
        if "options" in self.inputs:
            return dict(self.inputs.options.get_dict())
        return {}

    def run_ph(self):
        """Submit the ph.x (DFPT) run on the q-mesh."""
        scf = getattr(self.ctx, "scf", None)
        if scf is None or not scf.is_finished_ok:
            return self.exit_codes.ERROR_CHILD

        qpoints = orm.KpointsData()
        qpoints.set_kpoints_mesh(list(self.inputs.qpoints_mesh.get_list()))

        inputs = {
            "ph": {
                "code": self.inputs.ph_code,
                "parameters": self.inputs.ph_parameters,
                # NOTE: aiida-qe exposes the CalcJob remote folder under
                # the link label ``remote_folder`` (not
                # ``output_remote_folder``) on PwBaseWorkChain / PhBase
                # outputs — using the wrong label raises here.
                "parent_folder": scf.outputs.remote_folder,
                "metadata": {
                    "label": "phonon-ph",
                    "description": "ph.x (DFPT) on the q-mesh of the primitive cell",
                    "options": self._child_options(),
                },
            },
            "qpoints": qpoints,
        }
        running = self.submit(PhBaseWorkChain, **inputs)
        self.report(f"submitted ph.x pk={running.pk}")
        self.to_context(ph_run=running)

    def run_q2r(self):
        """Turn the q-mesh dynamical matrices into real-space force constants."""
        ph_run = getattr(self.ctx, "ph_run", None)
        if ph_run is None or not ph_run.is_finished_ok:
            return self.exit_codes.ERROR_CHILD

        inputs = {
            "q2r": {
                "code": self.inputs.q2r_code,
                "parent_folder": ph_run.outputs.remote_folder,
                "metadata": {
                    "label": "phonon-q2r",
                    "description": "q2r.x real-space force constants",
                    "options": self._child_options(),
                },
            }
        }
        if "q2r_parameters" in self.inputs:
            inputs["q2r"]["parameters"] = self.inputs.q2r_parameters
        running = self.submit(Q2rBaseWorkChain, **inputs)
        self.report(f"submitted q2r.x pk={running.pk}")
        self.to_context(q2r_run=running)

    def _matdyn_inputs(self, force_constants, kpoints, parameters, label):
        """One MatdynBaseWorkChain submission."""
        options = self._child_options()
        return {
            "matdyn": {
                "code": self.inputs.matdyn_code,
                "force_constants": force_constants,
                "kpoints": kpoints,
                "parameters": parameters,
                "metadata": {
                    "label": label,
                    "description": "matdyn.x phonon run",
                    "options": options,
                },
            }
        }

    def _matdyn_parameters(self, dos: bool):
        """matdyn.x INPUT namelist, with the dos flag set as requested.

        ``asr='crystal'`` is forced (unless the preset overrides it): the
        real-space force constants from q2r.x are truncated at the q-mesh
        range, so the interpolation can leave a spurious few-cm⁻¹ gap at
        Γ — enforcing the acoustic-sum rule at interpolation time keeps
        the acoustic branches anchored at zero. QE 7.5 matdyn.x reads
        ``asr`` (q2r.x reads ``zasr``).
        """
        params: dict = {}
        if "matdyn_parameters" in self.inputs:
            params = dict(self.inputs.matdyn_parameters.get_dict())
        namelist = dict(params.get("INPUT", {}))
        namelist.setdefault("asr", "crystal")
        namelist["dos"] = dos
        return orm.Dict(dict={"INPUT": namelist})

    def run_matdyn(self):
        """Submit matdyn.x along the band path and (optionally) on the DOS mesh."""
        q2r_run = getattr(self.ctx, "q2r_run", None)
        if q2r_run is None or not q2r_run.is_finished_ok:
            return self.exit_codes.ERROR_CHILD

        force_constants = q2r_run.outputs.force_constants

        # Band structure along the seekpath high-symmetry q path.
        band_inputs = self._matdyn_inputs(
            force_constants,
            self.ctx.band_kpoints,
            self._matdyn_parameters(dos=False),
            "phonon-matdyn-band",
        )
        band_run = self.submit(MatdynBaseWorkChain, **band_inputs)
        self.report(f"submitted matdyn.x (band) pk={band_run.pk}")
        self.to_context(matdyn_band=band_run)

        # Total DOS on a dense q mesh (matdyn.x needs a mesh to set nk1..3).
        if self.inputs.dos.value:
            dos_mesh = (
                list(self.inputs.dos_mesh.get_list())
                if "dos_mesh" in self.inputs
                else [21, 21, 21]
            )
            dos_kpoints = orm.KpointsData()
            dos_kpoints.set_kpoints_mesh(dos_mesh)
            dos_inputs = self._matdyn_inputs(
                force_constants,
                dos_kpoints,
                self._matdyn_parameters(dos=True),
                "phonon-matdyn-dos",
            )
            dos_run = self.submit(MatdynBaseWorkChain, **dos_inputs)
            self.report(f"submitted matdyn.x (dos) pk={dos_run.pk}")
            self.to_context(matdyn_dos=dos_run)

    def gather_results(self):
        """Forward the outputs and build the summary."""
        band_run = getattr(self.ctx, "matdyn_band", None)
        if band_run is None or not band_run.is_finished_ok:
            self.report(
                f"matdyn band run pk={getattr(band_run, 'pk', None)} failed "
                f"(state={getattr(band_run, 'process_state', None)})"
            )
            return self.exit_codes.ERROR_CHILD

        # The aiida-qe matdyn parser misaligns QE 7.5's "&plot" frequencies
        # file (it advances 6 tokens per q-point while matdyn writes 4
        # q-coordinate tokens + 6 frequencies per q) — the parsed
        # ``output_phonon_bands`` would contain rolling zeros. Parse the
        # raw file here instead (cm⁻¹ → THz) and re-attach the seekpath
        # labels.
        try:
            phonon_bands = _parse_matdyn_frequencies(
                retrieved=band_run.outputs.retrieved,
                path_kpoints=self.ctx.band_kpoints,
            )
        except (AttributeError, KeyError, ValueError, OSError) as exc:
            self.report(f"could not parse matdyn frequencies: {exc}")
            return self.exit_codes.ERROR_PARSER
        self.out("phonon_bands", phonon_bands)

        dos_node = None
        dos_run = getattr(self.ctx, "matdyn_dos", None)
        if dos_run is not None:
            if not dos_run.is_finished_ok:
                return self.exit_codes.ERROR_CHILD
            dos_node = _dos_to_thz(dos=dos_run.outputs.output_phonon_dos)
            self.out("total_phonon_dos", dos_node)

        force_constants = self.ctx.q2r_run.outputs.force_constants
        self.out("force_constants", force_constants)

        labels = _band_label_list(self.ctx.band_kpoints)
        self.out(
            "output_parameters",
            _summarize(
                bands=phonon_bands,
                structure_formula=self.inputs.structure.get_formula(),
                qpoints_mesh=list(self.inputs.qpoints_mesh.get_list()),
                dos=self.inputs.dos.value,
                dos_mesh=(
                    list(self.inputs.dos_mesh.get_list())
                    if "dos_mesh" in self.inputs
                    else None
                ),
                band_labels=labels,
                scf_pk=self.ctx.scf.pk,
                ph_pk=self.ctx.ph_run.pk,
                q2r_pk=self.ctx.q2r_run.pk,
                matdyn_band_pk=band_run.pk,
                matdyn_band_uuid=str(band_run.uuid),
                matdyn_dos_pk=getattr(dos_run, "pk", None) if dos_run is not None else None,
                phonopy_pk=band_run.pk,
                phonopy_uuid=str(band_run.uuid),
            ),
        )
        return None  # SUCCESS


# ---------------------------------------------------------------------------
# calcfunctions
# ---------------------------------------------------------------------------


@calcfunction
def _parse_matdyn_frequencies(retrieved, path_kpoints):
    """Parse matdyn.x's raw ``phonon_frequencies.dat`` (QE 7.5 layout).

    QE 7.5's matdyn.x writes the frequency file in gnuplot form::

        &plot nbnd=   6, nks= 202 /
        <q coords, 4 numbers>
        <nbnd frequencies in cm⁻¹>

    per q-point. The aiida-qe 5.0 parser advances only 6 tokens per
    q-point (it was written for an older layout without the per-q
    coordinate line), so every row after the first is misaligned — the
    stored ``output_phonon_bands`` shows frequencies jumping to ~0 as
    q-coordinate tokens leak in. Here the raw file is parsed correctly
    (cm⁻¹ → THz) and the seekpath q-path kpoints + labels are attached.
    """
    import re

    names = retrieved.base.repository.list_object_names()
    candidates = [name for name in names if "frequenc" in name]
    if not candidates:
        raise ValueError(
            f"matdyn retrieved folder has no frequency file ({names})"
        )
    content = retrieved.base.repository.get_object_content(candidates[0])

    match = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", content)
    if match is None:
        raise ValueError(
            "unrecognised matdyn frequency-file header — expected "
            "'&plot nbnd=..., nks=...' (QE 7.5 gnuplot format)"
        )
    nbnd, nks = int(match.group(1)), int(match.group(2))

    tokens = content.split()
    if "/" not in tokens:
        raise ValueError("matdyn frequency file has no '/' header terminator")
    numbers = [float(token) for token in tokens[tokens.index("/") + 1:]]
    per_q = 4 + nbnd  # 4 q-coordinate tokens + nbnd frequencies
    if len(numbers) != nks * per_q:
        raise ValueError(
            f"matdyn frequency file token count mismatch: "
            f"got {len(numbers)} floats, expected {nks} q-points x {per_q}"
        )

    freqs = np.zeros((nks, nbnd))
    for index in range(nks):
        start = index * per_q + 4
        freqs[index] = np.asarray(numbers[start:start + nbnd]) * _INVCM_TO_THZ

    output = orm.BandsData()
    output.set_kpointsdata(path_kpoints)
    output.set_bands(freqs, units="THz")

    # KpointsData paths store the labels split across label_numbers/labels;
    # expose them as [index, label] pairs like the phonopy workflows
    # (seekpath ASCII names mapped to the conventional Greek symbols).
    label_numbers = path_kpoints.base.attributes.get("label_numbers")
    raw_labels = path_kpoints.base.attributes.get("labels")
    if label_numbers and raw_labels:
        output.base.attributes.set(
            "labels",
            [
                [
                    int(index),
                    _GREEK_LABELS.get(str(symbol), str(symbol)),
                ]
                for index, symbol in zip(label_numbers, raw_labels)
            ],
        )
    return output


@calcfunction
def _dos_to_thz(dos):
    """Rescale a matdyn phonon DOS (cm⁻¹) to THz.

    matdyn.x reports the DOS in cm⁻¹; the phonopy workflows of the other
    backends (and the shared reports / figures) use THz. The density is
    rescaled so the total integral is preserved.
    """
    x_name, x_cm, _x_units = dos.get_x()
    (y_name, y_cm, _y_units), = dos.get_y()
    output = orm.XyData()
    output.set_x(
        np.asarray(x_cm) * _INVCM_TO_THZ,
        "frequency",
        "THz",
    )
    output.set_y(
        np.asarray(y_cm) * _THZ_TO_INVCM,
        y_name,
        "states/THz",
    )
    return output


def _band_label_list(path_kpoints):
    """Return the high-symmetry labels of a seekpath q-point path.

    Accepts both storage layouts: plain label strings (the ``labels``
    attribute of a ``KpointsData`` path) and ``[index, label]`` pairs
    (e.g. a ``BandsData`` path).
    """
    raw = path_kpoints.base.attributes.get("labels")
    if not isinstance(raw, list):
        return None
    labels = []
    for item in raw:
        if isinstance(item, str):
            symbol = item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            symbol = str(item[1])
        else:
            continue
        if symbol in ("", "?"):
            continue
        if symbol not in labels:
            labels.append(symbol)
    return labels if len(labels) >= 2 else None


@calcfunction
def _summarize(
    bands,
    structure_formula=None,
    qpoints_mesh=None,
    dos=None,
    dos_mesh=None,
    band_labels=None,
    scf_pk=None,
    ph_pk=None,
    q2r_pk=None,
    matdyn_band_pk=None,
    matdyn_band_uuid=None,
    matdyn_dos_pk=None,
    phonopy_pk=None,
    phonopy_uuid=None,
):
    """Build the ``output_parameters`` Dict of the QE phonon WorkChain."""
    freqs = np.asarray(bands.get_bands())
    result = {
        "workflow": "phonopy",
        "backend": "qe",
        "method": "dfpt",
        "structure_formula": structure_formula,
        "qpoints_mesh": qpoints_mesh,
        "dos": bool(dos),
        "dos_mesh": dos_mesh,
        "band_labels": band_labels,
        "phonopy_parameters": {
            "band": "auto",
            "band_points": len(freqs),
            "dos": bool(dos),
            "mesh": dos_mesh,
        },
        "frequency_min_thz": float(freqs.min()) if freqs.size else None,
        "frequency_max_thz": float(freqs.max()) if freqs.size else None,
        "n_imaginary_modes": int((freqs < 0).sum()) if freqs.size else None,
        "scf_pk": scf_pk,
        "ph_pk": ph_pk,
        "q2r_pk": q2r_pk,
        "matdyn_band_pk": matdyn_band_pk,
        "matdyn_band_uuid": matdyn_band_uuid,
        "matdyn_dos_pk": matdyn_dos_pk,
        "phonopy_pk": phonopy_pk,
        "phonopy_uuid": phonopy_uuid,
    }
    return orm.Dict(dict=result)
