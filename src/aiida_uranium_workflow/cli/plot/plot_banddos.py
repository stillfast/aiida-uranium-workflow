#!/usr/bin/env python3
"""Plot band / DOS / PDOS from a JSON spec.

Usage:
    aiida-uranium-plot-banddos -i band.json            # combined figure
    aiida-uranium-plot-banddos -i dos.json  -o /tmp/fig
    aiida-uranium-plot-banddos -i pdos.json --no-combined
    aiida-uranium-plot-banddos -i band_compare.json    # pairwise η_v / max η tables

The JSON spec is the same shape produced by hand-editing the
``band.json`` / ``dos.json`` / ``pdos.json`` files in
``aiida-uranium-scripts/banddos/<case>/``::

    {
      "mode": "band" | "dos" | "pdos" | "band_compare",
      "is_combined": true,
      "data": {
        "<backend>": {"pks": [...], "labels": [...]},
        ...
      },
      "figure": {
        "title": "...", "xlabel": "...", "ylabel": "...",
        "energy_range": [emin, emax],       # band mode
        "xlim": [...], "ylim": [...],       # dos / pdos modes
        "zero_to_efermi": true,
        "legend_loc": "best",
        "fig_name": "band.png"
      }
    }

``band_compare`` mode reuses the same ``data`` block (pks + labels
from a ``band.json``) and writes a Markdown report plus heatmap PNGs
of the pairwise η_v / max η / ω matrices (PRB 98, 085117). The
Fermi–Dirac smearing ``figure.sigma`` (default 0.1 eV) and an optional
energy window ``figure.e_min`` / ``figure.e_max`` (eV, relative to
each E_F) can be overridden on the command line with ``--sigma`` /
``--e-min`` / ``--e-max``.

Multiple spec files can be passed; each gets its own PNG.

When ``--which <backend1.key1.presetX,...>`` is supplied, the
presets are resolved against ``--output-json`` (the file produced by
``aiida-uranium run``) and the resolved pk/uuid is injected into the
spec's ``data[backend1].pks`` before rendering. This lets users write
only the scalar figure settings in the JSON and pick the actual pk
list at the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from aiida_uranium_workflow.cli.plot._loading import (
    load_output_json,
    load_spec,
    resolve_pks_from_output,
)
from aiida_uranium_workflow.cli.plot._rendering import (
    render_band_compare,
    render_spec,
)


def _inject_resolved_pks(spec, output_json: Path, selectors: List[str]) -> None:
    """Resolve ``backend.key.preset`` selectors and merge into ``spec``.

    Each selector is split into ``(backend, key, preset)`` triplets.
    For every matching ``(backend, key, preset)`` we append the
    resolved pk to ``spec.data[backend].pks`` (creating the
    :class:`BackendSeries` if missing).
    """
    if not selectors:
        return
    output = load_output_json(output_json)
    for sel in selectors:
        parts = sel.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Selector {sel!r} must be 'backend.key.preset' "
                f"(e.g. 'abacus.banddos.pw_r')."
            )
        backend, key, preset = parts
        pk = resolve_pks_from_output(output, backend, key, preset)
        if pk is None:
            raise ValueError(
                f"Selector {sel!r} did not resolve in {output_json}. "
                f"Available: {list(output)}"
            )
        if backend not in spec.data:
            from aiida_uranium_workflow.cli.plot._loading import BackendSeries
            spec.data[backend] = BackendSeries(backend=backend)
        spec.data[backend].pks.append(str(pk))
        if preset not in spec.data[backend].labels:
            spec.data[backend].labels.append(preset)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiida-uranium-plot-banddos",
        description=(
            "Render band / DOS / pdos figures from JSON specs. "
            "Each spec file produces one PNG (or one per backend / pk "
            "when ``is_combined`` is false)."
        ),
    )
    parser.add_argument(
        "-i", "--input", dest="specs", required=True, nargs="+",
        help="One or more JSON spec files (band.json / dos.json / pdos.json).",
    )
    parser.add_argument(
        "-o", "--output-dir", dest="output_dir", default=".",
        help="Directory to write PNGs into (default: current directory).",
    )
    parser.add_argument(
        "--output-json", dest="output_json", default=None,
        help=(
            "Optional aiida-uranium output.json. When --which is "
            "supplied, presets are resolved against this file."
        ),
    )
    parser.add_argument(
        "--which", dest="which", action="append", default=[],
        help=(
            "Inject a preset into every spec. Format: "
            "'backend.key.preset'. May be repeated."
        ),
    )
    parser.add_argument(
        "--no-combined", action="store_true",
        help="Force ``is_combined=False`` on every spec (overrides JSON).",
    )
    parser.add_argument(
        "--sigma", type=float, default=None,
        help=(
            "band_compare only: Fermi–Dirac smearing width in eV "
            "(default: 0.1, or the spec's figure.sigma)."
        ),
    )
    parser.add_argument(
        "--e-min", type=float, default=None,
        help="band_compare only: lower bound of the energy window (eV, "
             "relative to each E_F; default: all bands).",
    )
    parser.add_argument(
        "--e-max", type=float, default=None,
        help="band_compare only: upper bound of the energy window (eV, "
             "relative to each E_F; default: all bands).",
    )
    parser.add_argument(
        "--align", choices=["auto", "window", "index"],
        default=None,
        help="band_compare only: how states are paired. 'window' (per-k "
             "energy-sorted pairing; default when a window is set) or "
             "'index' (by band index).",
    )
    parser.add_argument(
        "--occupied-only", action="store_true",
        help="band_compare only: restrict max η to states occupied in "
             "both structures (Fermi–Dirac weight > 0.5; SSSP max_diff "
             "semantics).",
    )
    args = parser.parse_args(argv)

    if (args.which or []) and not args.output_json:
        parser.error(
            "--which requires --output-json (the file produced by "
            "`aiida-uranium run`)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load output.json once when --which is used (cheap; reused
    # across every spec).
    output_json_path = Path(args.output_json) if args.output_json else None

    failures = 0
    for spec_path in args.specs:
        spec_path = Path(spec_path)
        try:
            spec = load_spec(spec_path)
        except Exception as exc:
            print(f"[plot] failed to parse {spec_path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if args.no_combined:
            spec.is_combined = False

        if args.which:
            try:
                _inject_resolved_pks(spec, output_json_path, args.which)
            except Exception as exc:
                print(f"[plot] {spec_path}: {exc}", file=sys.stderr)
                failures += 1
                continue

        # band_compare: CLI overrides for the Fermi–Dirac smearing, the
        # energy window, the alignment mode and the max-η mask.
        if spec.mode == "band_compare":
            if args.sigma is not None:
                spec.figure.sigma = args.sigma
            if args.e_min is not None:
                spec.figure.e_min = args.e_min
            if args.e_max is not None:
                spec.figure.e_max = args.e_max
            if args.align is not None:
                spec.figure.align = args.align
            if args.occupied_only:
                spec.figure.occupied_only = True

        try:
            if spec.mode == "band_compare":
                paths = render_band_compare(spec, out_dir)
            else:
                paths = render_spec(spec, out_dir)
        except Exception as exc:
            print(f"[plot] {spec_path}: render failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        for path in paths:
            print(f"[plot] {spec_path.name} -> {path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())