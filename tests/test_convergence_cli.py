"""Tests for convergence workflow CLI."""

from __future__ import annotations

from aiida_uranium_workflow.cli.main import main

import pytest


class TestConvergenceCLI:
    """Tests for convergence workflow CLI."""

    def test_main_help(self, capsys):
        """Test that main() with --help exits with 0 and shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "aiida-uranium run" in captured.out
        assert "-h, --help" in captured.out

    def test_main_no_args(self, capsys):
        """Test that main() without args exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
