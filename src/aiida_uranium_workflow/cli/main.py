#!/usr/bin/env python3
"""Unified CLI entry point for ``aiida-uranium-workflow``.

Command form::

    aiida-uranium {run,report,archive,copy} --method {smear,convergence,magmom}

Subcommands:

* ``run``     — submit WorkChains from a unified input JSON and write a
                 ``output.json`` identifier map next to it (WorkChain
                 UUIDs as leaves, falling back to pks for legacy
                 readers).
* ``report``  — read an ``output.json`` produced by ``run`` and emit a
                 Markdown report per WorkChain (filename uses a short
                 8-character identifier).
* ``archive`` — read an ``output.json``, validate that every identifier
                 is a WorkChain of the selected method, and export
                 them into a single ``.aiida`` archive.
* ``copy``    — read an ``output.json``, walk every referenced
                 WorkChain's provenance, and copy each CalcJob's
                 ``outputs.remote_folder`` into
                 ``PATH/<backend>/<key>/<preset>/<calcjob_label>/`` on
                 the current host via AiiDA's transport API. Supports
                 ``--dry-run`` to only list source/destination paths.

The per-method bits (WorkChain class names, ``backend_to_key`` layout,
report generator callable) all live in :data:`cli._common.METHOD_SPECS`,
so adding a new method only requires extending that dict and the legacy
``*_run.py`` / ``*_report.py`` shims can keep using their own helpers.
"""

from __future__ import annotations

from aiida import load_profile
from aiida.tools.archive import create_archive
from aiida_uranium_workflow.cli._common import (
    MethodSpec,
    _short_id,
    build_unified_parser,
    collect_pk_map,
    default_result_path,
    execute_workflow,
    generate_one_report,
    get_method_spec,
    list_archive_pks,
    resolve_method,
)
from aiida_uranium_workflow.utils.cal_json import write_cal_json
from aiida_uranium_workflow.utils.copy_remote import (
    execute_copy_plan as _execute_copy_plan,
)
from aiida_uranium_workflow.utils.copy_remote import (
    load_copy_plan,
)

import sys
from pathlib import Path


def _resolve(args, *, log_tag: str) -> tuple[str, MethodSpec] | None:
    """Resolve method → ``(name, MethodSpec)`` or ``None`` on error.

    Logs the resolution source to stderr and prints a helpful hint so
    users can tell where the method came from. Returns ``None`` when
    :func:`resolve_method` raises; the caller should then ``return 1``.
    """
    try:
        method = resolve_method(
            cli_method=getattr(args, "method", None),
            input_json=getattr(args, "input_json", None),
            output_json=getattr(args, "input_json", None),
        )
    except ValueError as exc:
        print(f"[{log_tag}] {exc}", file=sys.stderr)
        return None
    return method, get_method_spec(method)


def _run(args) -> int:
    resolved = _resolve(args, log_tag=f"{getattr(args, 'method', None) or 'auto'}-run")
    if resolved is None:
        return 1
    method, spec = resolved

    submitted = execute_workflow(
        input_json=args.input_json,
        profile=args.profile,
        only=args.only,
    )

    if not submitted:
        print(f"[{spec.name}-run] no workchain submitted", file=sys.stderr)
        return 1

    print(
        f"[{spec.name}-run] submitted {len(submitted)} workchain(s); "
        f"check with: verdi process list"
    )

    out_path = Path(args.output) if args.output else default_result_path(args.input_json)
    write_cal_json(
        submitted,
        output_path=out_path,
        workflow=spec.name,
        backend_to_key=spec.backend_to_key or None,
    )
    print(f"[{spec.name}-run] saved output.json to {out_path}")
    return 0


def _report(args) -> int:
    resolved = _resolve(args, log_tag=f"{getattr(args, 'method', None) or 'auto'}-report")
    if resolved is None:
        return 1
    method, spec = resolved

    input_path = Path(args.input_json)
    try:
        output_data = collect_pk_map(input_path)
    except ValueError as exc:
        print(f"[{spec.name}-report] {exc}", file=sys.stderr)
        return 1

    reports = []
    for backend_data in output_data.values():
        if not isinstance(backend_data, dict):
            continue
        for method_data in backend_data.values():
            if not isinstance(method_data, dict):
                continue
            for key, node_id in method_data.items():
                if isinstance(node_id, (int, str)) and not isinstance(node_id, bool):
                    reports.append((str(key), node_id))

    if not reports:
        print(
            f"[{spec.name}-report] no node identifiers found in {input_path}",
            file=sys.stderr,
        )
        return 1

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.resolve().parent / "reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{spec.name}-report] writing {len(reports)} report(s) to {out_dir}")
    ok = 0
    for key, node_id in reports:
        short = _short_id(node_id)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        out = out_dir / f"report_{safe_key}_{short}.md"
        status = generate_one_report(
            node_identifier=node_id,
            output_path=out,
            profile=args.profile,
            class_to_backend=spec.class_to_backend,
            generate_report=spec.generate_report,
        )
        print(f"  id={short}: {status}")
        if status.startswith("ok"):
            ok += 1

    print(f"[{spec.name}-report] {ok}/{len(reports)} succeeded.")
    return 0 if ok == len(reports) else 1


def _archive(args) -> int:
    resolved = _resolve(args, log_tag="archive")
    if resolved is None:
        return 1
    method, spec = resolved
    try:
        pk_map = collect_pk_map(args.input_json)
    except ValueError as exc:
        print(f"[archive] {exc}", file=sys.stderr)
        return 1

    load_profile(args.profile)

    valid_ids, mismatched = list_archive_pks(
        pk_map,
        method=spec.name,
        class_to_backend=spec.class_to_backend,
    )

    if mismatched:
        print(
            f"[archive] filtered {len(mismatched)} node identifier(s) "
            f"whose WorkChain type doesn't match method '{spec.name}':",
            file=sys.stderr,
        )
        for node_id, class_name in mismatched:
            print(f"  - id={node_id} -> {class_name}", file=sys.stderr)

    if not valid_ids:
        print(
            "[archive] no node identifiers match the selected method; aborting.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[archive] {len(valid_ids)} identifier(s) matched method '{spec.name}'."
    )

    if args.dry_run:
        for node_id in valid_ids:
            print(f"  - id={node_id}")
        return 0

    from aiida.orm import load_node

    nodes = [load_node(node_id) for node_id in valid_ids]

    output_file = Path(args.output).resolve()
    create_archive(
        entities=nodes,
        filename=str(output_file),
        include_comments=args.include_comments,
        include_logs=args.include_logs,
        include_authinfos=False,
        overwrite=True,
        call_calc_backward=True,
        call_work_backward=True,
        create_backward=True,
        input_calc_forward=False,
        input_work_forward=False,
        return_backward=False,
    )

    print(f"[archive] wrote archive: {output_file}")
    return 0


def _copy(args) -> int:
    resolved = _resolve(args, log_tag=f"{getattr(args, 'method', None) or 'auto'}-copy")
    if resolved is None:
        return 1
    method, spec = resolved

    if args.profile:
        load_profile(args.profile)

    try:
        plan = load_copy_plan(
            input_json=args.input_json,
            method=spec.name,
            class_to_backend=spec.class_to_backend,
            base_dir=args.output,
        )
    except ValueError as exc:
        print(f"[{spec.name}-copy] {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[{spec.name}-copy] {exc}", file=sys.stderr)
        return 1

    if plan.skipped:
        print(
            f"[{spec.name}-copy] skipped {len(plan.skipped)} target(s):",
            file=sys.stderr,
        )
        for target, reason in plan.skipped:
            print(
                f"  - calcjob pk={target.calcjob_pk} ({reason})",
                file=sys.stderr,
            )

    if not plan.entries:
        print(
            f"[{spec.name}-copy] no copyable remote_folder found "
            f"under {args.output}; aborting.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[{spec.name}-copy] {len(plan.entries)} target(s) resolved "
        f"to {args.output}."
    )

    if args.dry_run:
        for entry in plan.entries:
            print(
                f"  - {entry.target.remote_path} -> {entry.local_path}"
            )
        return 0

    success, failures = _execute_copy_plan(plan.entries)
    for entry, message in failures:
        print(
            f"  - failed: calcjob pk={entry.target.calcjob_pk} -> {message}",
            file=sys.stderr,
        )
    print(
        f"[{spec.name}-copy] {success}/{len(plan.entries)} copied "
        f"successfully; {len(failures)} failed."
    )
    return 1 if failures else 0


_DISPATCH = {
    "run": _run,
    "report": _report,
    "archive": _archive,
    "copy": _copy,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point registered as the ``aiida-uranium`` console script."""
    parser = build_unified_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
