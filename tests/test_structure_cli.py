"""Tests for :mod:`aiida_uranium_workflow.utils.structure` CIF CLI.

The ``write_cif`` / ``main`` helpers turn a structure name in
``static/structure.yml`` into a CIF file on disk.  These tests cover
the file-writing logic and the CLI behaviour using the package's
bundled YAML registry (no fake fixtures required) so a regression in
pyxtal or ase.io would surface here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


# ---------------------------------------------------------------------------
# write_cif
# ---------------------------------------------------------------------------


class TestWriteCif:
    """``write_cif`` writes a CIF file for each registered structure."""

    def test_writes_cif_to_specified_directory(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import write_cif

        out = write_cif("bcc-uranium", output_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".cif"
        assert out.name == "bcc-uranium.cif"
        # CIF must contain the data block + atom loop header.
        text = out.read_text(encoding="utf-8")
        assert "data_" in text
        assert "_atom_site" in text

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import write_cif

        nested = tmp_path / "deep" / "nested" / "dir"
        out = write_cif("bcc-uranium-0K", output_dir=nested)
        assert nested.is_dir()
        assert out.exists()
        assert out.name == "bcc-uranium-0K.cif"

    def test_unknown_structure_raises_key_error(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import write_cif

        with pytest.raises(KeyError, match="no-such-structure"):
            write_cif("no-such-structure", output_dir=tmp_path)

    def test_slash_in_name_is_flattened(self, tmp_path: Path) -> None:
        """Keys like ``U-X/FCC`` would otherwise be treated as sub-dirs."""
        from aiida_uranium_workflow.utils.structure import write_cif

        out = write_cif("U-X/FCC", output_dir=tmp_path)
        assert out.exists()
        # ``/`` is rewritten to ``__`` so the file lands directly under
        # ``output_dir`` instead of a nested sub-directory.
        assert out.parent == tmp_path.resolve()
        assert out.name == "U-X__FCC.cif"


# ---------------------------------------------------------------------------
# main — CLI entry point
# ---------------------------------------------------------------------------


class TestMainList:
    """``--list`` prints registered structures and exits cleanly."""

    def test_list_prints_all_structures(self, capsys) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main(["--list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "bcc-uranium" in out
        assert "bcc-uranium-0K" in out
        assert "Available structures:" in out

    def test_list_does_not_write_files(self, tmp_path: Path, capsys) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main(["--list"])
        assert rc == 0
        # No .cif files anywhere under tmp_path.
        assert list(tmp_path.rglob("*.cif")) == []


class TestMainGenerate:
    """``main`` writes one CIF file per requested structure."""

    def test_single_structure(self, tmp_path: Path, capsys) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main(["bcc-uranium", "-o", str(tmp_path)])
        assert rc == 0
        cif = tmp_path / "bcc-uranium.cif"
        assert cif.exists()
        assert cif.stat().st_size > 0
        out = capsys.readouterr().out
        assert "wrote" in out
        assert "1 CIF file(s) generated" in out

    def test_multiple_structures(self, tmp_path: Path, capsys) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main(
            ["bcc-uranium", "bcc-uranium-0K", "-o", str(tmp_path)]
        )
        assert rc == 0
        assert (tmp_path / "bcc-uranium.cif").exists()
        assert (tmp_path / "bcc-uranium-0K.cif").exists()
        out = capsys.readouterr().out
        assert "2 CIF file(s) generated" in out

    def test_unknown_structure_returns_error_code(
        self, tmp_path: Path, capsys
    ) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main(["bcc-uranium", "no-such", "-o", str(tmp_path)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "no-such" in captured.err
        # The valid structure is still written before the failure.
        assert (tmp_path / "bcc-uranium.cif").exists()

    def test_no_names_returns_error_code(self, capsys) -> None:
        from aiida_uranium_workflow.utils.structure import main

        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "structure name is required" in captured.err

    def test_output_dir_is_created(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import main

        target = tmp_path / "fresh" / "cifs"
        assert not target.exists()
        rc = main(["bcc-uranium", "-o", str(target)])
        assert rc == 0
        assert target.is_dir()
        assert (target / "bcc-uranium.cif").exists()


class TestCifFileContents:
    """Light sanity-check on the actual CIF content produced."""

    def test_cif_contains_expected_element(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import write_cif

        cif = write_cif("bcc-uranium", output_dir=tmp_path)
        text = cif.read_text(encoding="utf-8")
        # ``U`` must appear in the atom_site loop of a uranium structure.
        assert "U" in text

    def test_multi_element_structure_contains_both_species(
        self, tmp_path: Path
    ) -> None:
        """Multi-element YAML entries (e.g. ``U-X2O``) must round-trip
        both species through the CIF writer."""
        from aiida_uranium_workflow.utils.structure import write_cif

        cif = write_cif("U-X2O", output_dir=tmp_path)
        text = cif.read_text(encoding="utf-8")
        assert "U" in text
        assert "O" in text
        # Lattice constant declared in the YAML must appear in the file.
        assert "5.8533" in text


class TestBuildStructureValidation:
    """Regression guards for the YAML shape checks in ``build_structure``."""

    def test_elements_length_must_match_wps(self, tmp_path: Path) -> None:
        from aiida_uranium_workflow.utils.structure import build_structure

        registry = {
            "bad": {
                "spacegroup": 225,
                "elements": ["U"],          # 1 species
                "wickoff_position": ["a", "b"],  # 2 sites — mismatch
                "x": [4.84],
            }
        }
        with pytest.raises(ValueError, match="'elements'"):
            build_structure("bad", registry=registry)

    def test_x_must_contain_lattice_param(self, tmp_path: Path) -> None:
        """``x`` must be a non-empty list — it carries the lattice
        parameter(s) (plus optional free Wyckoff coordinates).
        """
        from aiida_uranium_workflow.utils.structure import build_structure

        registry = {
            "bad": {
                "spacegroup": 225,
                "elements": ["U", "O"],
                "wickoff_position": ["a", "b"],
                "x": [],
            }
        }
        with pytest.raises(ValueError, match="'x' must be a non-empty list"):
            build_structure("bad", registry=registry)

    def test_x_lattice_param_only_is_valid(self, tmp_path: Path) -> None:
        """``x = [a]`` (single cubic lattice parameter) is valid when the
        Wyckoff positions are fixed — no free coordinates needed.
        Matches the real ``structure.yml`` entries (e.g. ``U-XO``).
        """
        from aiida_uranium_workflow.utils.structure import build_structure

        registry = {
            "ok": {
                "spacegroup": 225,
                "elements": ["U", "O"],
                "wickoff_position": ["a", "b"],
                "x": [4.8408],
            }
        }
        atoms = build_structure("ok", registry=registry)
        # SG 225 a/b positions each have multiplicity 4.
        assert len(atoms) == 8
        assert "U" in atoms.get_chemical_symbols()
        assert "O" in atoms.get_chemical_symbols()