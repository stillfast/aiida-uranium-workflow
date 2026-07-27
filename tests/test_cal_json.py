"""Tests for :mod:`aiida_uranium_workflow.utils.cal_json`.

These tests exercise:

1. ``build_cal_json`` writes **UUID strings** (not integer pks) in the
   leaves when ``SubmittedJob.uuid`` is populated.
2. ``build_cal_json`` falls back to the integer pk when ``uuid`` is empty
   (legacy / test-stub ``SubmittedJob`` records).
3. ``write_cal_json`` round-trips through the file system.
4. ``collect_pks_from_json`` / ``collect_node_ids_from_json`` /
   ``collect_identifiers_from_json`` correctly walk both legacy pk-style
   and modern UUID-style ``output.json`` payloads.
5. ``_short_id`` produces filesystem-friendly short identifiers (8
   characters for UUIDs).
"""

from __future__ import annotations

import json

from aiida_uranium_workflow.schedulers.base import SubmittedJob
from aiida_uranium_workflow.utils.cal_json import (
    _node_identifier,
    build_cal_json,
    write_cal_json,
)
from aiida_uranium_workflow.utils.json_collect import (
    _looks_like_uuid,
    collect_identifiers_from_json,
    collect_node_ids_from_json,
    collect_pks_from_json,
)
from aiida_uranium_workflow.cli._common import _short_id


# Two valid AiiDA UUIDs plus one synthetic pk to exercise both shapes.
UUID_PW = "8c0fe1a9-1234-4abc-9def-0a1b2c3d4e5f"
UUID_PW_R = "33e15b7c-9f8a-4b6c-8123-0f9e8d7c6b5a"
UUID_VASP = "5f4a3b2c-1d2e-3f4a-5b6c-7d8e9f0a1b2c"


class TestBuildCalJsonUUIDs:
    """``build_cal_json`` records UUIDs (preferred) or pks (fallback)."""

    def test_uuid_written_when_set(self):
        """When ``SubmittedJob.uuid`` is non-empty, the JSON leaf is the UUID."""
        jobs = [
            SubmittedJob(
                backend="abacus",
                preset_name="pw",
                pk=42,
                structure_name="bcc-uranium",
                uuid=UUID_PW,
            ),
            SubmittedJob(
                backend="abacus",
                preset_name="pw_r",
                pk=43,
                structure_name="bcc-uranium",
                uuid=UUID_PW_R,
            ),
            SubmittedJob(
                backend="vasp",
                preset_name="test",
                pk=44,
                structure_name="bcc-uranium",
                uuid=UUID_VASP,
            ),
        ]

        data = build_cal_json(jobs, workflow="convergence")
        # Convergence writes ``"convergence"`` as the inner key for both
        # backends (cf. ``DEFAULT_BACKEND_TO_KEY``).
        assert data["abacus"]["convergence"]["pw"] == UUID_PW
        assert data["abacus"]["convergence"]["pw_r"] == UUID_PW_R
        assert data["vasp"]["convergence"]["test"] == UUID_VASP

    def test_pk_fallback_when_uuid_empty(self):
        """Without a UUID the integer pk is preserved on the leaf."""
        jobs = [
            SubmittedJob(
                backend="abacus",
                preset_name="pw",
                pk=42,
                structure_name="bcc-uranium",
                uuid="",  # legacy / stub
            ),
        ]
        data = build_cal_json(jobs, workflow="convergence")
        assert data["abacus"]["convergence"]["pw"] == 42

    def test_uuid_preferred_over_pk(self):
        """When a UUID is present, the pk is shadowed."""
        jobs = [
            SubmittedJob(
                backend="abacus",
                preset_name="pw",
                pk=42,
                structure_name="bcc-uranium",
                uuid=UUID_PW,
            ),
        ]
        data = build_cal_json(jobs, workflow="convergence")
        assert data["abacus"]["convergence"]["pw"] == UUID_PW
        assert data["abacus"]["convergence"]["pw"] != 42

    def test_serialization_is_json_compatible(self):
        """The resulting object must JSON-serialise via ``json.dumps``."""
        jobs = [
            SubmittedJob(
                backend="abacus",
                preset_name="pw",
                pk=42,
                structure_name="bcc-uranium",
                uuid=UUID_PW,
            ),
        ]
        data = build_cal_json(jobs, workflow="convergence")
        rendered = json.dumps(data)
        # UUID string must appear verbatim.
        assert UUID_PW in rendered
        # And an integer pk (42) must NOT show up in a UUID-file.
        assert '"pw": 42' not in rendered


class TestWriteCalJsonRoundTrip:
    """``write_cal_json`` actually writes a JSON file."""

    def test_write_and_readback(self, tmp_path):
        out_path = tmp_path / "output.json"
        jobs = [
            SubmittedJob(
                backend="abacus",
                preset_name="pw",
                pk=1,
                structure_name="bcc-uranium",
                uuid=UUID_PW,
            ),
            SubmittedJob(
                backend="vasp",
                preset_name="test",
                pk=2,
                structure_name="bcc-uranium",
                uuid=UUID_VASP,
            ),
        ]

        write_cal_json(jobs, output_path=out_path, workflow="convergence")

        assert out_path.is_file()
        loaded = json.loads(out_path.read_text())
        assert loaded["abacus"]["convergence"]["pw"] == UUID_PW
        assert loaded["vasp"]["convergence"]["test"] == UUID_VASP


class TestNodeIdentifierHelper:
    """``_node_identifier`` returns UUID when present, else pk."""

    def test_uuid_path(self):
        job = SubmittedJob(
            backend="abacus", preset_name="pw", pk=42,
            structure_name="bcc-uranium", uuid=UUID_PW,
        )
        assert _node_identifier(job) == UUID_PW

    def test_pk_fallback(self):
        job = SubmittedJob(
            backend="abacus", preset_name="pw", pk=42,
            structure_name="bcc-uranium", uuid="",
        )
        assert _node_identifier(job) == 42


# ---------------------------------------------------------------------------
# JSON walker tests — old PK-style and new UUID-style payloads
# ---------------------------------------------------------------------------


class TestCollectPksFromJson:
    """``collect_pks_from_json`` only walks integer pk leaves."""

    def test_old_pk_payload(self):
        data = {
            "abacus": {"convergence": {"pw": 1, "pw_r": 2}},
            "vasp": {"convergence": {"test": 3}},
        }
        pks = sorted(collect_pks_from_json(data))
        assert pks == [1, 2, 3]

    def test_uuid_payload_returns_no_pks(self):
        data = {
            "abacus": {"convergence": {"pw": UUID_PW}},
        }
        # ``UUID_PW`` is a string, not an int; the legacy helper
        # leaves it untouched.
        assert collect_pks_from_json(data) == []

    def test_mixed_payload(self):
        data = {
            "abacus": {"convergence": {"pw": 1, "pw_r": UUID_PW_R}},
        }
        assert collect_pks_from_json(data) == [1]


class TestCollectNodeIdsFromJson:
    """``collect_node_ids_from_json`` only walks UUID leaves."""

    def test_uuid_payload(self):
        data = {
            "abacus": {"convergence": {"pw": UUID_PW, "pw_r": UUID_PW_R}},
        }
        ids = sorted(collect_node_ids_from_json(data))
        assert ids == sorted([UUID_PW, UUID_PW_R])

    def test_old_pk_payload_returns_empty(self):
        data = {"abacus": {"convergence": {"pw": 1, "pw_r": 2}}}
        assert collect_node_ids_from_json(data) == []

    def test_non_uuid_strings_are_skipped(self):
        # Preset names ("pw_r") happen to look similar to UUIDs at first
        # glance; ``_looks_like_uuid`` rejects them.
        data = {"abacus": {"convergence": {"pw": "pw_r"}}}
        assert collect_node_ids_from_json(data) == []


class TestCollectIdentifiersFromJson:
    """``collect_identifiers_from_json`` handles both pk and UUID leaves."""

    def test_legacy_payload_only_pks(self):
        data = {
            "abacus": {"smear": {"lcao": 1, "pw": 2}},
            "vasp": {"vasp": {"test": 3}},
        }
        ids = sorted(collect_identifiers_from_json(data))
        assert ids == ["1", "2", "3"]

    def test_modern_payload_only_uuids(self):
        data = {
            "abacus": {"smear": {"lcao": UUID_PW, "pw": UUID_PW_R}},
            "vasp": {"vasp": {"test": UUID_VASP}},
        }
        ids = sorted(collect_identifiers_from_json(data))
        assert ids == sorted([UUID_PW, UUID_PW_R, UUID_VASP])

    def test_mixed_payload_yields_all(self):
        data = {
            "abacus": {"smear": {"lcao": 42, "pw": UUID_PW}},
            "vasp": {"vasp": {"test": UUID_VASP}},
        }
        ids = sorted(collect_identifiers_from_json(data))
        assert ids == sorted(["42", UUID_PW, UUID_VASP])

    def test_bool_is_not_treated_as_pk(self):
        """``True``/``False`` aren't mistakenly collected as integer pks."""
        data = {"abacus": {"smear": {"x": True, "y": False}}}
        assert collect_identifiers_from_json(data) == []

    def test_preset_names_are_not_collected(self):
        """Non-UUID strings are silently skipped."""
        data = {"abacus": {"smear": {"lcao": 1, "pw": 2}}, "smear": "test"}
        # Top-level workflow-protocol slot is a string ("test") but
        # it's not a UUID, so it's skipped.
        ids = sorted(collect_identifiers_from_json(data))
        assert ids == ["1", "2"]


class TestLooksLikeUUID:
    """``_looks_like_uuid`` matches the canonical 8-4-4-4-12 form."""

    def test_valid_uuids(self):
        assert _looks_like_uuid(UUID_PW)
        assert _looks_like_uuid(UUID_PW_R)
        assert _looks_like_uuid(UUID_VASP)

    def test_invalid_strings(self):
        assert not _looks_like_uuid("not-a-uuid")
        assert not _looks_like_uuid("8c0fe1a9")  # too short
        assert not _looks_like_uuid("")
        # 36 chars but wrong layout
        assert not _looks_like_uuid("8c0fe1a9x1234y4abc-zdef-0a1b2c3d4e5f")
        assert not _looks_like_uuid("8c0fe1a9-1234-4abc-9def")  # too short

    def test_non_string_input(self):
        assert not _looks_like_uuid(42)
        assert not _looks_like_uuid(None)
        assert not _looks_like_uuid(["list"])


class TestShortIdHelper:
    """``_short_id`` returns filesystem-friendly IDs (8 chars for UUIDs)."""

    def test_int_passthrough(self):
        assert _short_id(123) == "123"

    def test_uuid_first_8(self):
        # The 8-hex prefix is taken verbatim (including any leading digits).
        assert _short_id(UUID_PW) == UUID_PW[:8]
        assert _short_id(UUID_PW_R) == UUID_PW_R[:8]

    def test_short_string_passthrough(self):
        assert _short_id("x") == "x"
