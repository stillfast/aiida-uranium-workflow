"""Tests for ``utils.parsers`` — the shared energy + wall-time parser.

These tests focus on the **pure helpers** (``parse_total_time``,
``parse_abacus_last_energy``) so they don't need a real AiiDA profile.
The AiiDA-aware helpers (``fetch_abacus`` / ``fetch_vasp``) are covered
indirectly by the workflow tests.
"""

from __future__ import annotations

import pytest

from aiida_uranium_workflow.utils.parsers import (
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
# Smoke check: helpers can be imported without AiiDA profile loaded
# ---------------------------------------------------------------------------


def test_import_without_aiida_profile():
    """The pure helpers must not require a configured AiiDA profile.

    This is what allows the workflow tests' ``smoke_inputs`` style
    fixtures to import the module without profile setup.
    """
    # No exception ⇒ the module is import-safe.
    from aiida_uranium_workflow.utils import parsers as parser_energy_time  # noqa: F401
