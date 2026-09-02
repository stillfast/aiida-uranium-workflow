#!/usr/bin/env python3
"""Render phonon band-structure (+DOS) figures.

Two modes:

* JSON-driven (band-style spec)::

      aiida-uranium-plot-phonon -i phonon.json -o ./out

  with a spec like::

      {
        "mode": "phonon",
        "is_combined": true,
        "data": {
          "abacus": {"pks": ["<uuid>", ...], "labels": ["pw", "lcao"]},
          "fleur":  {"pks": ["<uuid>"], "labels": ["lapw"]}
        },
        "figure": {
          "title": "", "xlabel": "k-point index", "ylabel": "Frequency (THz)",
          "ylim": [-2, 2], "legend_loc": "best", "fig_name": "phonon.png"
        }
      }

  Every series is drawn on the shared band axis (its own colour, legend
  ``backend/label``) with its DOS on the right axis. ``figure.ylim`` is
  the frequency window (THz).

* Single-node (legacy)::

      aiida-uranium-plot-phonon -i <pk-or-uuid> [-o out.png] [--labels ...]

  The ``-i`` argument is treated as a spec file when it names an existing
  file, otherwise as a node pk / UUID (anything exposing ``phonon_bands``).
"""

from __future__ import annotations

from pathlib import Path

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiida-uranium-plot-phonon",
        description=(
            "Render phonon band + DOS figures from a JSON spec "
            "(band-style, multi-node) or a single node identifier "
            "(pk/UUID) exposing 'phonon_bands'."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="JSON plot spec (e.g. phonon.json) OR a node pk/UUID.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        help="Output directory (spec mode) or PNG path (single-node mode).",
    )
    parser.add_argument(
        "-p",
        "--profile",
        dest="profile",
        default=None,
        help="AiiDA profile name.",
    )
    # Legacy single-node options (ignored in spec mode).
    parser.add_argument(
        "--labels",
        dest="labels",
        default=None,
        help="Comma-separated high-symmetry labels (single-node mode only).",
    )
    parser.add_argument(
        "--title",
        dest="title",
        default=None,
        help="Optional figure title (single-node mode only).",
    )
    parser.add_argument(
        "--fmin",
        dest="fmin",
        type=float,
        default=None,
        help="Frequency window lower bound, THz (single-node mode only).",
    )
    parser.add_argument(
        "--fmax",
        dest="fmax",
        type=float,
        default=None,
        help="Frequency window upper bound, THz (single-node mode only).",
    )
    args = parser.parse_args(argv)

    from aiida import load_profile

    load_profile(args.profile)

    input_path = Path(args.input)
    if input_path.is_file():
        # JSON-driven multi-node mode (band-style spec).
        from aiida_uranium_workflow.cli.plot._loading import load_spec
        from aiida_uranium_workflow.utils.plot.phonon import render_phonon_spec

        spec = load_spec(input_path)
        out_dir = Path(args.output) if args.output else input_path.parent
        try:
            paths = render_phonon_spec(spec, out_dir)
        except Exception as exc:
            print(f"[plot-phonon] failed: {exc}", file=sys.stderr)
            return 1
        for path in paths:
            print(f"[plot-phonon] saved figure to {path}")
        return 0

    # Legacy single-node mode.
    from aiida_uranium_workflow.utils.plot.phonon import render_phonon_figure

    short = args.input
    if len(short) >= 8:
        short = short[:8]
    output = Path(args.output) if args.output else Path(f"phonon_bands_dos_{short}.png")

    band_labels = None
    if args.labels:
        band_labels = [lb.strip() for lb in args.labels.split(",") if lb.strip()]

    freq_range = None
    if args.fmin is not None or args.fmax is not None:
        freq_range = (
            args.fmin if args.fmin is not None else -10.0,
            args.fmax if args.fmax is not None else 10.0,
        )

    try:
        path = render_phonon_figure(
            args.input,
            output,
            band_labels=band_labels,
            title=args.title,
            freq_range=freq_range,
        )
    except Exception as exc:
        print(f"[plot-phonon] failed: {exc}", file=sys.stderr)
        return 1

    print(f"[plot-phonon] saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
