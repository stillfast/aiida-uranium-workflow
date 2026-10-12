"""Tests for ``utils.parsers`` — the shared energy + wall-time parser.

These tests focus on the **pure helpers** (``parse_total_time``,
``parse_abacus_last_energy``) so they don't need a real AiiDA profile.
The AiiDA-aware helpers (``fetch_abacus`` / ``fetch_vasp``) are covered
indirectly by the workflow tests.
"""

from __future__ import annotations

import pytest

from aiida_uranium_workflow.utils.parsers import (
    count_abacus_elec_steps,
    parse_abacus_last_energy,
    parse_total_time,
)


# ---------------------------------------------------------------------------
# parse_total_time
# ---------------------------------------------------------------------------


class TestParseTotalTime:
    """Parse the ``Total  Time : H h M mins S secs`` ABACUS log line."""

    def test_basic(self) -> None:
        assert parse_total_time("Total  Time : 0 h 5 mins 12 secs") == 5 * 60 + 12

    def test_with_extra_whitespace(self) -> None:
        assert parse_total_time("Total    Time  :   1 h 2 mins 3 secs") == 3600 + 2 * 60 + 3

    def test_multiple_lines_last_wins(self) -> None:
        # Some logs print a per-stage summary and then a run-level one.
        text = "\n".join(
            [
                "Total  Time : 0 h 0 mins 5 secs",
                "Some other log line",
                "Total  Time : 0 h 1 mins 30 secs",
            ]
        )
        assert parse_total_time(text) == 1 * 60 + 30

    def test_missing_line_returns_none(self) -> None:
        assert parse_total_time("nothing here") is None

    def test_empty_string(self) -> None:
        assert parse_total_time("") is None

    def test_malformed_minutes_returns_none(self) -> None:
        # ``mins`` is a literal token in the regex; deviations don't match.
        assert parse_total_time("Total  Time : 0 h 5 minutes 12 secs") is None


# ---------------------------------------------------------------------------
# parse_abacus_last_energy
# ---------------------------------------------------------------------------


class TestParseAbacusLastEnergy:
    """Pull the last total-energy footer from an ABACUS log."""

    def test_modern_footer(self) -> None:
        log = "Some output\n!FINAL_ETOT_IS  -1234.567890 eV\n"
        assert parse_abacus_last_energy(log) == pytest.approx(-1234.56789)

    def test_legacy_footer(self) -> None:
        log = "...\nfinal etot is -42.5 eV\n"
        assert parse_abacus_last_energy(log) == pytest.approx(-42.5)

    def test_multiple_footers_last_wins(self) -> None:
        log = "\n".join(
            [
                "stage 1",
                "final etot is -10.0 eV",
                "stage 2",
                "final etot is -20.0 eV",
            ]
        )
        assert parse_abacus_last_energy(log) == pytest.approx(-20.0)

    def test_no_footer_returns_none(self) -> None:
        assert parse_abacus_last_energy("just a log\nwithout any total energy") is None

    def test_empty(self) -> None:
        assert parse_abacus_last_energy("") is None


# ---------------------------------------------------------------------------
# count_abacus_elec_steps
# ---------------------------------------------------------------------------


class TestCountAbacusElecSteps:
    """Both ABACUS electronic-step log formats must be counted.

    The ``abacus_lts`` build prints ``ION=   1  ELEC=   n`` while the
    3.9.x / PEXSI build prints ``--> #ION MOVE#   1  #ELEC ITER#   n``.
    Counting only the legacy format silently reported ``None`` for every
    run of the newer binary (found on the PEXSI smoke run pk 380925).
    """

    def test_legacy_format(self) -> None:
        log = "\n".join(
            f" ION=   1  ELEC={index:4d}" for index in range(1, 6)
        )
        assert count_abacus_elec_steps(log) == 5

    def test_modern_ion_move_format(self) -> None:
        log = "\n".join(
            f" --> #ION MOVE#         1  #ELEC ITER#  {index:8d}"
            for index in range(1, 41)
        )
        assert count_abacus_elec_steps(log) == 40

    def test_mixed_formats_takes_the_larger_count(self) -> None:
        """Each format is counted on its own and the larger count wins,
        so a log carrying both markers is never double-counted."""
        legacy = "\n".join(f" ION=   1  ELEC={index:4d}" for index in range(1, 4))
        modern = " --> #ION MOVE#         1  #ELEC ITER#         1"
        assert count_abacus_elec_steps(f"{legacy}\n{modern}") == 3

    def test_unrelated_log_returns_none(self) -> None:
        assert count_abacus_elec_steps("DONE : INIT SCF Time : 0.27 (SEC)") is None

    def test_empty_and_missing_text(self) -> None:
        assert count_abacus_elec_steps("") is None
        assert count_abacus_elec_steps(None) is None


# ---------------------------------------------------------------------------
# Smoke check: helpers can be imported without AiiDA profile loaded
# ---------------------------------------------------------------------------


def test_import_without_aiida_profile():
    """The pure helpers must not require a configured AiiDA profile.

    This is what allows the workflow tests' ``smoke_inputs`` style
    fixtures to import the module without profile setup.
    """
    # No exception ⇒ the module is import-safe.
    from aiida_uranium_workflow.utils import parsers as parser_energy_time  # noqa: F401


class TestFailedNodeTolerance:
    """``fetch_abacus`` must survive a node without ``misc``/``retrieved``.

    A partly failed parameter scan (see the PEXSI verification steps) keeps
    the successful cells and records the failed ones; collecting the failed
    ones used to raise ``NotExistentAttributeError`` and kill the whole
    WorkChain instead of tabulating them.
    """

    def test_missing_outputs_return_none(self):
        from aiida_uranium_workflow.utils.parsers import fetch_summary

        class _Bare:
            """Node stand-in with neither outputs nor inputs."""

            class outputs:  # noqa: D106
                pass

            class inputs:  # noqa: D106
                pass

        summary = fetch_summary(_Bare(), "abacus")
        assert summary["energy_ev"] is None
        assert summary["time_s"] is None
        assert summary["scf_steps"] is None
        assert summary["natoms"] is None
