"""Shared CLI helpers used by every ``cli.*`` module and the unified
``cli.main`` entry point.

Goals:

* Centralise the WorkChain detection / report-writing pipeline shared
  by the unified ``aiida-uranium {run,report,archive,copy}`` entry
  point (base, smear, convergence, magmom, banddos).
* Expose a :data:`METHOD_SPECS` registry that drives the unified
  ``aiida-uranium {run,report,archive}`` entry point — adding a new
  method only requires extending this dict.

The module deliberately stays import-friendly — it pulls in
``aiida``/``aiida.orm`` lazily so that ``argparse``-only paths (e.g.
``--help``) do not require a configured profile.
"""

from __future__ import annotations

from aiida import load_profile
from aiida.orm import load_node
from aiida_uranium_workflow.schedulers import get_orchestrator
from aiida_uranium_workflow.utils.config import ConfigLoader
from aiida_uranium_workflow.utils.report.convergence import (
    generate_report as generate_convergence_report,
)
from aiida_uranium_workflow.utils.report.elastic import (
    generate_report as generate_elastic_report,
)
from aiida_uranium_workflow.utils.report.magmom import (
    generate_report as generate_magmom_report,
)
from aiida_uranium_workflow.utils.report.phonopy import (
    generate_report as generate_phonopy_report,
)
from aiida_uranium_workflow.utils.report.relax import (
    generate_report as generate_relax_report,
)
from aiida_uranium_workflow.utils.report.smear import (
    generate_report as generate_smear_report,
)
from aiida_uranium_workflow.utils.report.eos import (
    generate_report as generate_eos_report,
)
from aiida_uranium_workflow.utils.report.defects import (
    generate_report as generate_defects_report,
)
from aiida_uranium_workflow.utils.report.supercell import (
    generate_report as generate_supercell_report,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import argparse
import inspect
import json
import sys

# ---------------------------------------------------------------------------
# Method registry (drives the unified CLI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """All the per-method bits the unified CLI needs to know about."""

    #: Canonical method name (``"base"`` / ``"smear"`` /
    #: ``"convergence"`` / ``"magmom"`` / ``"banddos"``).
    name: str
    #: ``class_name -> backend`` mapping used for filter / dispatch.
    class_to_backend: Mapping[str, str]
    #: ``generate_report(output_params, pk, backend) -> str`` importable.
    generate_report: Callable[..., str]
    #: ``backend -> key`` layout written to the output.json pk map.
    backend_to_key: Mapping[str, str] = field(default_factory=dict)


#: ``class_name -> backend`` for each supported workflow. The order
#: (vasp-first) matches the messages the original shims emitted so log
#: scrapers stay happy.
SMEAR_CLASS_TO_BACKEND: dict[str, str] = {
    "VaspSmearWorkChain": "vasp",
    "AbacusSmearWorkChain": "abacus",
}
CONVERGENCE_CLASS_TO_BACKEND: dict[str, str] = {
    "VaspConvergenceWorkChain": "vasp",
    "AbacusConvergenceWorkChain": "abacus",
}
MAGMOM_CLASS_TO_BACKEND: dict[str, str] = {
    "VaspMagmomWorkChain": "vasp",
    "AbacusMagmomWorkChain": "abacus",
    "FleurMagmomWorkChain": "fleur",
    "QeMagmomWorkChain": "qe",
}
BASE_CLASS_TO_BACKEND: dict[str, str] = {
    "VaspWorkChain": "vasp",
    "AbacusBaseWorkChain": "abacus",
}
BANDDOS_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusBandWorkChain": "abacus",
}
RELAX_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusRelaxWorkChain": "abacus",
    "FleurRelaxWorkChain": "fleur",
}
ELASTIC_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusElasticWorkChain": "abacus",
    "VaspElasticWorkChain": "vasp",
    "FleurElasticWorkChain": "fleur",
}
PHONOPY_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusPhonopyWorkChain": "abacus",
    "FleurPhonopyWorkChain": "fleur",
}
EOS_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusEosWorkChain": "abacus",
    "FleurEosWorkChain": "fleur",
}
DEFECTS_CLASS_TO_BACKEND: dict[str, str] = {
    "AbacusDefectsWorkChain": "abacus",
    "FleurDefectsWorkChain": "fleur",
}
SUPERCELL_CLASS_TO_BACKEND: dict[str, str] = {
    "SupercellScfWorkChain": "abacus",
}


def _unsupported_base_report(*_args, **_kwargs) -> str:
    raise NotImplementedError("Reports are not supported for direct base workflows")


METHOD_SPECS: dict[str, MethodSpec] = {
    "base": MethodSpec(
        name="base",
        class_to_backend=BASE_CLASS_TO_BACKEND,
        generate_report=_unsupported_base_report,
        backend_to_key={"abacus": "abacus", "vasp": "vasp"},
    ),
    "smear": MethodSpec(
        name="smear",
        class_to_backend=SMEAR_CLASS_TO_BACKEND,
        generate_report=generate_smear_report,
        backend_to_key={"abacus": "smear", "vasp": "smear"},
    ),
    "convergence": MethodSpec(
        name="convergence",
        class_to_backend=CONVERGENCE_CLASS_TO_BACKEND,
        generate_report=generate_convergence_report,
        backend_to_key={"abacus": "convergence", "vasp": "convergence"},
    ),
    "magmom": MethodSpec(
        name="magmom",
        class_to_backend=MAGMOM_CLASS_TO_BACKEND,
        generate_report=generate_magmom_report,
        backend_to_key={"abacus": "magmom", "vasp": "magmom",
                        "fleur": "magmom", "qe": "magmom"},
    ),
    "banddos": MethodSpec(
        name="banddos",
        class_to_backend=BANDDOS_CLASS_TO_BACKEND,
        generate_report=_unsupported_base_report,
        backend_to_key={"abacus": "scf", "fleur": "scf"},
    ),
    "relax": MethodSpec(
        name="relax",
        class_to_backend=RELAX_CLASS_TO_BACKEND,
        generate_report=generate_relax_report,
        backend_to_key={"abacus": "scf", "fleur": "scf"},
    ),
    "elastic": MethodSpec(
        name="elastic",
        class_to_backend=ELASTIC_CLASS_TO_BACKEND,
        generate_report=generate_elastic_report,
        backend_to_key={"abacus": "scf", "vasp": "elastic", "fleur": "scf"},
    ),
    "phonopy": MethodSpec(
        name="phonopy",
        class_to_backend=PHONOPY_CLASS_TO_BACKEND,
        generate_report=generate_phonopy_report,
        backend_to_key={"abacus": "scf", "fleur": "scf"},
    ),
    "eos": MethodSpec(
        name="eos",
        class_to_backend=EOS_CLASS_TO_BACKEND,
        generate_report=generate_eos_report,
        backend_to_key={"abacus": "scf", "fleur": "scf"},
    ),
    "defects": MethodSpec(
        name="defects",
        class_to_backend=DEFECTS_CLASS_TO_BACKEND,
        generate_report=generate_defects_report,
        backend_to_key={"abacus": "scf", "fleur": "scf"},
    ),
    "supercell": MethodSpec(
        name="supercell",
        class_to_backend=SUPERCELL_CLASS_TO_BACKEND,
        generate_report=generate_supercell_report,
        backend_to_key={"abacus": "scf"},
    ),
}


SUPPORTED_METHODS: tuple[str, ...] = tuple(METHOD_SPECS.keys())


def get_method_spec(method: str) -> MethodSpec:
    """Return the :class:`MethodSpec` for ``method`` (raises ``ValueError``)."""
    try:
        return METHOD_SPECS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown method '{method}'. Supported: {list(METHOD_SPECS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Method resolution (CLI <-> input.json <-> output.json)
# ---------------------------------------------------------------------------


def _read_json_workflow_field(path: str | Path | None) -> str | None:
    """Read a JSON file's top-level ``"workflow"`` field if present.

    Returns ``None`` when the file is missing, malformed, or doesn't
    contain a non-empty string ``"workflow"`` key. Defensive against
    bad paths so callers can simply chain sources without try/except.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("workflow")
    return value if isinstance(value, str) and value else None


def resolve_method(
    *,
    cli_method: str | None,
    input_json: str | Path | None = None,
    output_json: str | Path | None = None,
) -> str:
    """Pick the canonical method name from the available sources.

    Resolution order (highest priority first):

    1. ``cli_method`` (explicit ``--method`` flag) — if provided, wins.
    2. ``output_json["workflow"]`` (modern output.json files) — used by
       ``report`` / ``copy`` / ``archive`` which never see input.json.
    3. ``input_json["workflow"]`` (used by ``run``).

    Raises :class:`ValueError` if no source can determine a valid method.
    No fallback inference — callers must provide a method explicitly.
    """
    if cli_method:
        if cli_method not in METHOD_SPECS:
            raise ValueError(
                f"Unknown method '{cli_method}'. " f"Supported: {list(METHOD_SPECS)}"
            )
        return cli_method

    workflow = _read_json_workflow_field(output_json)
    if workflow:
        if workflow not in METHOD_SPECS:
            raise ValueError(
                f"output.json['workflow']='{workflow}' is not a known method. "
                f"Supported: {list(METHOD_SPECS)}"
            )
        return workflow

    workflow = _read_json_workflow_field(input_json)
    if workflow:
        if workflow not in METHOD_SPECS:
            raise ValueError(
                f"input.json['workflow']='{workflow}' is not a known method. "
                f"Supported: {list(METHOD_SPECS)}"
            )
        return workflow

    raise ValueError(
        "Cannot determine the workflow method. Provide --method, or use an "
        "input.json / output.json that contains a 'workflow' field. "
        f"Supported: {list(METHOD_SPECS)}"
    )


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


def execute_workflow(
    *,
    input_json: str,
    profile: str | None,
    only: str | None,
) -> list:
    """Run the orchestrator that matches ``input_json``'s workflow.

    Returns the list of :class:`SubmittedJob` produced by
    :meth:`WorkflowOrchestrator.run_with_jobs`. Each record carries
    both the integer pk and the WorkChain UUID string (the canonical
    identifier written into ``output.json``).

    Raises ``FileNotFoundError`` / ``ValueError`` (from
    :class:`ConfigLoader`) on bad input — the CLI is expected to let
    those propagate.
    """
    bundle = ConfigLoader(input_json).load_all()
    if profile:
        bundle.input_params["profile"] = profile

    backends = (only,) if only else None
    orchestrator = get_orchestrator(bundle, backends=backends)
    return orchestrator.run_with_jobs()


# ---------------------------------------------------------------------------
# Report-script helpers (smear_report / convergence_report / magmom_report)
# ---------------------------------------------------------------------------


def load_finished_workchain(node_identifier: int | str, profile: str | None):
    """``load_profile`` + ``load_node(id)`` + ``is_finished`` check.

    ``node_identifier`` can be an integer pk (legacy) or a UUID string
    (modern ``output.json`` layout) — :func:`aiida.orm.load_node`
    transparently accepts both.

    Returns a ``(workchain, status)`` tuple where ``status`` is one of:

    * ``"ok"`` — workchain loaded successfully and ``is_finished``.
    * ``"load_failed"`` — :func:`load_node` raised.
    * ``"not_finished"`` — workchain is loaded but not finished.

    No output is written; the caller decides how to surface the
    failure (convergence/magmom use ``print(..., file=sys.stderr)``,
    smear returns a status string instead).
    """
    load_profile(profile)
    try:
        workchain = load_node(node_identifier)
    except Exception as e:
        return (None, e), "load_failed"

    if not workchain.is_finished:
        return (workchain, workchain.process_state.value), "not_finished"

    return workchain, "ok"


def resolve_backend(
    class_name: str,
    class_to_backend: Mapping[str, str],
) -> str | None:
    """Map a WorkChain class name to its backend (``"abacus"`` / ``"vasp"``).

    Returns ``None`` when the class name isn't supported. The caller is
    responsible for the user-facing error message (different report
    shims want different wording).
    """
    return class_to_backend.get(class_name)


def _short_id(node_identifier: int | str) -> str:
    """Return a filesystem-friendly, mostly-unique short id for ``id``.

    For integer pks this is just ``str(pk)``. For UUID strings we keep
    the leading 8 hex characters (the equivalent of ``git short
    hash``); collision risk across a single ``output.json`` is
    negligible, and the result stays short and unambiguous in logs.
    """
    if isinstance(node_identifier, int):
        return str(node_identifier)
    text = str(node_identifier)
    if len(text) >= 8 and text.replace("-", "")[:8]:
        return text[:8]
    return text


def write_text_report(
    report_text: str,
    output_path: str | Path,
) -> bool:
    """Write ``report_text`` to ``output_path``.

    Returns ``True`` on success, ``False`` on IO failure.
    """
    output_path = Path(output_path)
    try:
        output_path.write_text(report_text)
    except Exception:
        return False
    return True


def generate_one_report(
    *,
    node_identifier: int | str,
    output_path: Path,
    profile: str | None,
    class_to_backend: Mapping[str, str],
    generate_report: Callable,
) -> str:
    """End-to-end "one node → one Markdown report" pipeline (smear_report).

    ``node_identifier`` is either an integer pk (legacy output.json) or
    a WorkChain UUID string (modern output.json); ``load_node(...)``
    accepts both. The short 8-hex prefix of the UUID is used in log
    strings to keep them readable, while the full identifier is
    forwarded to AiiDA.

    Steps:

    1. :func:`load_finished_workchain` — verify the node loads and the
       workflow is finished.
    2. Map ``process_class.__name__`` to ``"abacus"`` / ``"vasp"`` via
       :func:`resolve_backend`.
    3. Pull ``output_parameters`` (failure surfaced as
       ``"failed: ... output_parameters ..."``).
    4. Call ``generate_report(output_parameters, node_identifier, workflow_type)``.
    5. Write the report (failure surfaced as
       ``"failed: write ... -> ..."``).

    Returns one of:

    * ``"ok -> <path>"``
    * ``"failed: load_node(<id>) -> <e>"``
    * ``"failed: id=<id> output_parameters -> <e>"``
    * ``"skipped: id=<id> unsupported WorkChain type '<ClassName>'"``
    * ``"skipped: id=<id> not finished (state=<...>)"``
    * ``"skipped: id=<id> not successful (exit_status=<code>)"``
    * ``"failed: write <path> -> <e>"``

    The status strings mention ``id=`` (instead of ``pk=``) so a UUID
    isn't misread as a pk. Log scrapers matching on the leading
    ``"failed: load_node("`` / ``"failed: "`` text remain functional.
    """
    short = _short_id(node_identifier)
    result, status = load_finished_workchain(node_identifier, profile)
    if status == "load_failed":
        _, exc = result
        return f"failed: load_node({node_identifier}) -> {exc}"

    # ``result`` is a ``(workchain, state_value)`` tuple while the node is
    # not finished yet — unpack it BEFORE touching the workchain (reading
    # ``process_class`` first used to crash on the tuple).
    if status == "not_finished":
        _, state_value = result
        return f"skipped: id={short} not finished " f"(state={state_value})"

    workchain = result
    class_name = workchain.process_class.__name__
    backend = resolve_backend(class_name, class_to_backend)
    if backend is None:
        return f"skipped: id={short} unsupported WorkChain type {class_name!r}"

    # Finished but unsuccessful (non-zero exit status, e.g. a failed EOS
    # SCF child): ignore the data point instead of emitting a misleading
    # report. The caller still generates the remaining reports.
    if not workchain.is_finished_ok:
        return f"skipped: id={short} not successful (exit_status={workchain.exit_status})"

    # Optional per-report extras: report generators that can also render
    # figures (e.g. phonopy's band + DOS image) accept ``figure_dir`` so
    # the figure lands next to the Markdown report. Generators that work
    # from the raw node (e.g. relax, where the plugin ``abacus.relax`` /
    # ``fleur.relax`` WorkChains have no combined ``output_parameters``)
    # accept ``workchain_node``. ``report_stem`` lets figure-generating
    # reports name their images after the report file
    # (``report_<key>_<short>.md``). Generators without such parameters
    # are called exactly as before.
    report_kwargs: dict[str, Any] = {}
    signature = inspect.signature(generate_report)
    if "figure_dir" in signature.parameters:
        report_kwargs["figure_dir"] = str(output_path.parent)
    if "report_stem" in signature.parameters:
        report_kwargs["report_stem"] = output_path.stem
    if "workchain_node" in signature.parameters:
        report_kwargs["workchain_node"] = workchain

    try:
        output_parameters = workchain.outputs.output_parameters.get_dict()
    except Exception as exc:
        # Plugin WorkChains (relax / band / ...) often expose no combined
        # ``output_parameters``; let generators that accept the node render
        # the report from its raw outputs instead of failing.
        if "workchain_node" not in report_kwargs:
            return f"failed: id={short} output_parameters -> {exc}"
        output_parameters = {}

    # For magmom workflows, ``AbacusMagmomWorkChain`` /
    # ``VaspMagmomWorkChain`` do not stash the original magmom config in
    # ``output_parameters``. The list lives only on the WorkChain's
    # ``inputs.magmom_list`` (per-species dicts) or
    # ``inputs.magmom_per_atom_list`` (per-site lists) port, so inject it
    # here so the report can render the ``initial magmom`` column. The
    # injection is done on a local copy — AiiDA's stored
    # ``output_parameters`` Dict is untouched.
    if class_name in MAGMOM_CLASS_TO_BACKEND and "magmom_list" not in output_parameters:
        try:
            if "magmom_list" in workchain.inputs:
                magmom_list_node = workchain.inputs.magmom_list
                output_parameters = dict(output_parameters)
                output_parameters["magmom_list"] = list(magmom_list_node.get_list())
            elif "magmom_per_atom_list" in workchain.inputs:
                magmom_per_atom_node = workchain.inputs.magmom_per_atom_list
                output_parameters = dict(output_parameters)
                output_parameters["magmom_list"] = list(magmom_per_atom_node.get_list())
        except (AttributeError, KeyError):
            pass

    report_text = generate_report(
        output_parameters, node_identifier, backend, **report_kwargs
    )
    if write_text_report(report_text, output_path):
        return f"ok -> {output_path}"
    return f"failed: write {output_path}"


# ---------------------------------------------------------------------------
# Unified subcommand helpers
# ---------------------------------------------------------------------------


def parse_method(value: str) -> str:
    """Validate ``--method`` against :data:`SUPPORTED_METHODS`."""
    if value not in METHOD_SPECS:
        raise argparse.ArgumentTypeError(
            f"unknown method '{value}', choose one of {list(METHOD_SPECS)}"
        )
    return value


def build_unified_parser(
    *,
    prog: str = "aiida-uranium",
    description: str = (
        "Unified CLI for aiida-uranium-workflow (run / report / archive /"
        " copy; methods: base / smear / convergence / magmom / banddos)."
    ),
) -> argparse.ArgumentParser:
    """Top-level parser for ``aiida-uranium {run,report,archive,copy}``."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    sub = p.add_subparsers(dest="command", required=True)

    # run -----------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        prog=f"{prog} run",
        help="Run a workflow (base / smear / convergence / magmom / banddos).",
        description=(
            "Submit one base / smear / convergence / magmom / banddos WorkChain per "
            "(backend, preset, structure) combination defined in the unified "
            "input JSON."
        ),
    )
    run_p.add_argument(
        "--method",
        type=parse_method,
        default=None,
        help=(
            "Workflow method: base / smear / convergence / magmom / banddos. "
            "If omitted, the value is read from input.json['workflow']."
        ),
    )
    run_p.add_argument(
        "-i",
        "--input",
        dest="input_json",
        required=True,
        help="Path to the unified input JSON file.",
    )
    run_p.add_argument(
        "-p",
        "--profile",
        default=None,
        help="AiiDA profile name (overrides input.json['profile']).",
    )
    run_p.add_argument(
        "--only",
        choices=("abacus", "vasp"),
        default=None,
        help="Restrict to a single backend.",
    )
    run_p.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        metavar="PATH",
        help=(
            "Where to write the output.json pk map. "
            "Default: <input_dir>/output.json."
        ),
    )

    # report --------------------------------------------------------------
    report_p = sub.add_parser(
        "report",
        prog=f"{prog} report",
        help="Generate Markdown reports from an output.json.",
        description=(
            "For each WorkChain node identifier (UUID string in modern "
            "output.json files; integer pk in legacy files) found in "
            "the output.json file, generate a Markdown report via the "
            "matching method's report generator."
        ),
    )
    report_p.add_argument(
        "--method",
        type=parse_method,
        default=None,
        help=(
            "Workflow method: base / smear / convergence / magmom. "
            "If omitted, the value is read from output.json['workflow']. "
            "Required if output.json lacks a 'workflow' field."
        ),
    )
    report_p.add_argument(
        "-i",
        "--input",
        dest="input_json",
        required=True,
        help="Path to the output.json produced by `aiida-uranium run`.",
    )
    report_p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=("Directory to write reports into. " "Default: <input_dir>/reports."),
    )
    report_p.add_argument(
        "-p",
        "--profile",
        default=None,
        help="AiiDA profile name.",
    )

    # copy ----------------------------------------------------------------
    copy_p = sub.add_parser(
        "copy",
        prog=f"{prog} copy",
        help=(
            "Copy remote_folder contents of each CalcJob child of the "
            "WorkChains listed in output.json to a local PATH, using "
            "AiiDA's transport API."
        ),
        description=(
            "Read an output.json produced by `aiida-uranium run`, walk "
            "the provenance of every WorkChain it references, and copy "
            "the contents of every CalcJobNode's outputs.remote_folder "
            "into PATH/<backend>/<key>/<preset>/<calcjob_label> on the "
            "current host. Use --dry-run to only list the source / "
            "destination paths."
        ),
    )
    copy_p.add_argument(
        "--method",
        type=parse_method,
        default=None,
        help=(
            "Workflow method used to validate WorkChain classes and "
            "to look up the inner JSON key. If omitted, the value is "
            "read from output.json['workflow']."
        ),
    )
    copy_p.add_argument(
        "-i",
        "--input",
        dest="input_json",
        required=True,
        help="Path to the output.json produced by `aiida-uranium run`.",
    )
    copy_p.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        metavar="PATH",
        help=(
            "Local destination directory. The full layout is "
            "PATH/<backend>/<key>/<preset>/<calcjob_label>/."
        ),
    )
    copy_p.add_argument(
        "-p",
        "--profile",
        default=None,
        help="AiiDA profile name.",
    )
    copy_p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only list source remote paths and the planned local "
            "destinations; do not transfer any bytes."
        ),
    )

    # archive -------------------------------------------------------------
    archive_p = sub.add_parser(
        "archive",
        prog=f"{prog} archive",
        help="Export node identifiers from an output.json into a .aiida archive.",
        description=(
            "Collect every node identifier (UUID string in modern "
            "output.json files; integer pk in legacy files) in the "
            "output.json file, validate each one against the selected "
            "method's WorkChain class set, and export them into a "
            "single AiiDA archive."
        ),
    )
    archive_p.add_argument(
        "--method",
        type=parse_method,
        default=None,
        help=(
            "Workflow method (used to validate the WorkChain class of "
            "each node identifier). If omitted, the value is read from "
            "output.json['workflow']."
        ),
    )
    archive_p.add_argument(
        "-i",
        "--input",
        dest="input_json",
        required=True,
        help="Path to the output.json produced by `aiida-uranium run`.",
    )
    archive_p.add_argument(
        "-o",
        "--output",
        dest="output",
        default="archive.aiida",
        help="Output archive file (default: archive.aiida).",
    )
    archive_p.add_argument(
        "-p",
        "--profile",
        default=None,
        help="AiiDA profile name.",
    )
    archive_p.add_argument(
        "--no-comments",
        dest="include_comments",
        action="store_false",
        help="Exclude comments from the archive.",
    )
    archive_p.add_argument(
        "--no-logs",
        dest="include_logs",
        action="store_false",
        help="Exclude logs from the archive.",
    )
    archive_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the node identifiers that would be archived.",
    )

    # check ---------------------------------------------------------------
    check_p = sub.add_parser(
        "check",
        prog=f"{prog} check",
        help="Dry-run validate an input.json and preview what would run.",
        description=(
            "Parse and validate the input.json against the workflow "
            "protocol / static tables (no AiiDA process is submitted). "
            "Prints the resolved backend / preset names, the SCF preset "
            "parameters, parsed protocol (workflow_data), scheduler "
            "options and codes that would reach each WorkChain; exits "
            "non-zero when the configuration is invalid."
        ),
    )
    check_p.add_argument(
        "-i",
        "--input",
        dest="input_json",
        required=True,
        help="Path to the unified input JSON file.",
    )
    check_p.add_argument(
        "-p",
        "--profile",
        default=None,
        help="AiiDA profile name (overrides input.json['profile']).",
    )

    # example -------------------------------------------------------------
    example_p = sub.add_parser(
        "example",
        prog=f"{prog} example",
        help="Generate a reference input.json for a method.",
        description=(
            "Write a minimal runnable input.json (structure / metadata "
            "presets resolved, code names are placeholders you must "
            "replace) into examples/<method>/."
        ),
    )
    example_p.add_argument(
        "-m",
        "--method",
        type=parse_method,
        default=None,
        help=(
            "Workflow method (base / smear / convergence / magmom / "
            "banddos / relax / elastic / eos / phonopy / defects / "
            "supercell). Required."
        ),
    )
    example_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory to write into (default: ./examples).",
    )

    # plot ----------------------------------------------------------------
    plot_p = sub.add_parser(
        "plot",
        prog=f"{prog} plot",
        help="Render band / DOS / pdos / band-compare / phonon figures.",
        description=(
            "Replaces aiida-uranium-plot-banddos / aiida-uranium-plot-"
            "phonon: read one or more spec JSON files and dispatch on "
            "their 'mode' (band / dos / pdos / band_compare / phonon)."
        ),
    )
    plot_p.add_argument(
        "-i",
        "--input",
        dest="specs",
        required=True,
        nargs="+",
        help="One or more JSON spec files.",
    )
    plot_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=".",
        help="Directory to write figures into (default: current directory).",
    )

    return p


def collect_pk_map(
    input_json: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Read an output.json into the nested ``{backend: {key: {preset: id}}}`` shape.

    ``id`` is typically a WorkChain UUID string (modern ``output.json``
    files) but can also be an integer pk (legacy files). The shape is
    preserved verbatim — callers use
    :func:`collect_identifiers_from_json` (or ``load_node(id)``
    directly) to consume the leaves.

    Raises :class:`ValueError` if the file is missing or malformed.
    """
    p = Path(input_json)
    if not p.is_file():
        raise ValueError(f"Input JSON not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a JSON object, got {type(data).__name__}")
    return data


def default_result_path(input_json: str | Path) -> Path:
    """``<input_dir>/output.json`` — used when ``--output`` is omitted."""
    return Path(input_json).resolve().parent / "output.json"


def list_archive_pks(
    pk_map: Mapping[str, Any],
    *,
    method: str,
    class_to_backend: Mapping[str, str],
) -> tuple[list[str], list[tuple[Any, str]]]:
    """Load each node identifier in ``pk_map`` and split into valid / mismatched lists.

    Each leaf of ``pk_map`` is treated as a *node identifier* — an
    integer pk (legacy) or a UUID string (modern). :func:`load_node`
    transparently handles both. ``mismatched`` collects ``(id,
    class_name)`` tuples for nodes whose ``process_class`` doesn't
    match ``method``.
    """
    valid: list[str] = []
    mismatched: list[tuple[Any, str]] = []
    for backend, by_key in pk_map.items():
        if not isinstance(by_key, dict):
            continue
        for _key, presets in by_key.items():
            if not isinstance(presets, dict):
                continue
            for _preset, node_id in presets.items():
                # Accept anything ``load_node`` can consume: integer pk
                # or UUID string. Skip other scalars defensively.
                if not isinstance(node_id, (int, str)) or isinstance(node_id, bool):
                    continue
                try:
                    node = load_node(node_id)
                except Exception as exc:
                    short = _short_id(node_id)
                    print(
                        f"[archive] failed to load id={short}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                class_name = node.process_class.__name__
                if class_name not in class_to_backend:
                    mismatched.append((node_id, class_name))
                    continue
                if class_to_backend[class_name] != backend:
                    mismatched.append((node_id, class_name))
                    continue
                valid.append(node_id)
    # Preserve a stable, sorted order (mixed int/str would raise in
    # Python 3 — sort by string form to keep order stable across runs).
    valid_sorted = sorted(set(valid), key=str)
    return valid_sorted, mismatched


__all__ = [
    # Method registry
    "MethodSpec",
    "METHOD_SPECS",
    "SUPPORTED_METHODS",
    "SMEAR_CLASS_TO_BACKEND",
    "CONVERGENCE_CLASS_TO_BACKEND",
    "MAGMOM_CLASS_TO_BACKEND",
    "BANDDOS_CLASS_TO_BACKEND",
    "get_method_spec",
    "parse_method",
    "resolve_method",
    # Run helpers
    "execute_workflow",
    # JSON helpers
    "collect_pk_map",
    "default_result_path",
    # Report helpers
    "load_finished_workchain",
    "resolve_backend",
    "write_text_report",
    "generate_one_report",
    # Archive helpers
    "list_archive_pks",
    # Parser
    "build_unified_parser",
]
