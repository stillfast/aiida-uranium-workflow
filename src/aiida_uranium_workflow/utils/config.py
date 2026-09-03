"""ConfigLoader: input-JSON + associated YAML, no AiiDA involved.

Used by every workchain's CLI entry point: the loader doesn't know what
``smear`` is, it just knows there is a ``protocol`` YAML and one
``<backend>.yml`` per backend.

The workflow-specific logic is delegated to the registry in
``schedulers/__init__.py``.
"""

from __future__ import annotations

from .common import ParamBundle, PARAMETERS_DIR, PROTOCOL_DIR, STATIC_DIR
from .yamlio import read_json, read_yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    """Read an input JSON and the YAML files it references.

    Knows nothing about AiiDA, ORMs, codes, or smearing — it only parses
    files. Output is a ``ParamBundle`` so callers can ignore the
    intermediate JSON structure if they wish.

    The loader uses the workflow registry to discover which protocol YAML
    to load based on ``input.json["workflow"]``.
    """

    REQUIRED_KEYS = ("workflow", "parameters", "static", "profile", "code")

    #: Backend keys that map to a ``parameters/<backend>/<backend>.yml``
    #: directory. Anything else appearing under ``parameters`` in
    #: ``input.json`` (e.g. ``"smear"``, ``"magmom"``, ``"convergence"``)
    #: is treated as a workflow-protocol slot and handled by
    #: :meth:`_load_protocol` instead.
    BACKEND_DIRS = ("abacus", "vasp", "fleur", "qe")

    def __init__(self, input_json_path: str | Path) -> None:
        self.input_json_path = input_json_path
        self.input_params: dict[str, Any] = read_json(input_json_path)

    def load_all(self) -> ParamBundle:
        """Populate every section of the bundle. Returns the bundle."""
        self._validate()

        # Get workflow-specific info from registry
        from aiida_uranium_workflow.schedulers import get_workflow_entry

        entry = get_workflow_entry(self.input_params["workflow"])

        # Load protocol YAML and parse with workflow-specific hook
        protocol = self._load_protocol(entry)
        workflow_data = self._parse_protocol(protocol, entry)

        # Multi-preset protocol (``parameters[<workflow_key>]`` as a list):
        # one WorkChain is submitted per requested preset, each with its own
        # parsed workflow_data. Single-name inputs leave these empty and the
        # orchestrator behaves exactly as before.
        raw_wf = self.input_params["parameters"].get(entry.workflow_key)
        if isinstance(raw_wf, (list, tuple)):
            preset_entries = self._load_protocol_entries(entry)
            if not preset_entries:
                raise KeyError(
                    f"None of the requested protocol presets {list(raw_wf)!r} "
                    f"exist in parameters/{entry.protocol_file}; check the "
                    f"'parameters' section of {self.input_json_path}."
                )
            workflow_presets = [name for name, _ in preset_entries]
            workflow_data_map = {
                name: self._parse_protocol(raw, entry)
                for name, raw in preset_entries
            }
        else:
            workflow_presets = []
            workflow_data_map = {}

        # Load backend-specific parameters
        software_params, software_preset_names = self._load_all_software_params()

        # Load metadata
        metadata = self._load_metadata()

        # An empty protocol section is legitimate when the workflow's
        # per-backend presets carry everything (e.g. FLEUR's banddos /
        # magmom presets under ``parameters/<backend>/``). But when
        # BOTH the protocol name is missing AND no backend preset was
        # requested, the input.json is effectively empty — surface a
        # descriptive error instead of silently submitting nothing.
        if not protocol and not software_params:
            name = self.input_params["parameters"].get(entry.workflow_key)
            raise KeyError(
                f"Protocol '{name}' not found for workflow "
                f"'{self.input_params['workflow']}' and no backend preset "
                f"was requested; check parameters/ and the 'parameters' "
                f"section of {self.input_json_path}."
            )

        return ParamBundle(
            input_params=self.input_params,
            protocol=protocol,
            workflow_data=workflow_data,
            software_params=software_params,
            software_preset_names=software_preset_names,
            metadata=metadata,
            workflow_presets=workflow_presets,
            workflow_data_map=workflow_data_map,
        )

    def _validate(self) -> None:
        for key in self.REQUIRED_KEYS:
            if key not in self.input_params:
                raise KeyError(f"input.json is missing required key '{key}'")
        self._validate_static()

    def _validate_static(self) -> None:
        """Validate ``static.structure`` accepts a string or a list of strings.

        Strings are the legacy single-structure form; the list form
        causes the orchestrator to submit one WorkChain per structure.
        Anything else is rejected up-front so a malformed input fails at
        load time rather than after profile / AiiDA setup.
        """
        static = self.input_params.get("static", {})
        if not isinstance(static, dict) or "structure" not in static:
            raise KeyError("input.json is missing 'static.structure'")
        struct = static["structure"]
        if isinstance(struct, str):
            return
        if isinstance(struct, (list, tuple)):
            if not struct:
                raise ValueError("'static.structure' list must be non-empty")
            if not all(isinstance(s, str) for s in struct):
                raise TypeError(
                    "'static.structure' must be a string or a list of strings, "
                    f"got {struct!r}"
                )
            return
        raise TypeError(
            "'static.structure' must be a string or a list of strings, "
            f"got type {type(struct).__name__}"
        )

    def _protocol_names(self, entry) -> list[str]:
        """Resolve ``parameters[<workflow_key>]`` to a list of preset names.

        Accepts a single preset name (``"defects": "test"``) or a list of
        preset names (``"defects": ["vacancy_scf", "vacancy_relax"]`` —
        one WorkChain is submitted per preset).
        """
        if entry.workflow_key not in self.input_params["parameters"]:
            return []
        raw = self.input_params["parameters"][entry.workflow_key]
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)) and all(isinstance(n, str) for n in raw):
            return list(raw)
        raise TypeError(
            f"parameters['{entry.workflow_key}'] must be a preset name or a "
            f"list of preset names, got {type(raw).__name__}"
        )

    def _load_protocol(self, entry) -> dict[str, Any]:
        """Load the workflow protocol, or return empty data when absent.

        The protocol name lives in ``parameters[<workflow>]`` (e.g.
        ``parameters["smear"]`` for the smear workflow) and points to a
        name inside the protocol YAML referenced by ``entry.protocol_file``.
        Direct base workflows deliberately have no protocol YAML or slot.

        For workflows with multiple backends (e.g. ``banddos`` runs on
        ABACUS *and* FLEUR), the protocol name may only exist in the
        backend-specific sub-YAML (``parameters/<backend>/<file>.yml``).
        When the top-level ``parameters/<file>.yml`` does not contain
        the protocol name, this method silently returns ``{}`` so the
        backend-specific presets can carry everything the workflow
        needs (FLEUR's ``scf.yml`` puts the SCF base while the protocol
        carries band_wf + dos_wf
        inside each preset).

        A **list** of preset names is allowed; the first one found in the
        protocol YAML becomes the default ``workflow_data`` (all requested
        presets are submitted via :meth:`_load_protocol_entries`).
        """
        if entry.protocol_file is None:
            return {}
        entries = self._load_protocol_entries(entry)
        return dict(entries[0][1]) if entries else {}

    def _load_protocol_entries(self, entry) -> list[tuple[str, dict[str, Any]]]:
        """Return ``[(preset_name, raw_preset_table), ...]`` for every
        requested protocol preset (single name or list)."""
        if entry.protocol_file is None:
            return []
        names = self._protocol_names(entry)
        protocol_path = PROTOCOL_DIR / entry.protocol_file
        if not protocol_path.is_file():
            return []
        table = read_yaml(protocol_path)
        return [(name, table[name]) for name in names if name in table]

    def _parse_protocol(self, protocol: dict[str, Any], entry) -> dict[str, Any]:
        """Apply workflow-specific parser hook if available."""
        if entry.parser_hook is not None:
            return entry.parser_hook(protocol)
        return {}

    def _load_all_software_params(
        self,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        """Load backend-specific parameters from their YAML files.

        Returns ``(software_params, preset_names)`` where
        ``preset_names[backend]`` holds the requested preset names in the
        exact same order as ``software_params[backend]`` (the resolved
        YAML sections do not carry their own name, so callers that need
        to label a submission — e.g. the ``check`` CLI — use the parallel
        name list).

        Each backend has its own subdirectory ``parameters/<backend>/``.
        The default preset file is ``parameters/<backend>/<backend>.yml``,
        but a backend slot may also pick a workflow-specific preset
        table by using the dict form::

            "parameters": {
                "abacus": "test",                       # → parameters/abacus/abacus.yml
                "abacus": ["test", "test_soc"],         # → same file, two presets
                "abacus": {"magmom": ["test_magmom"]},  # → parameters/abacus/magmom.yml
            }

        The dict form dispatches to ``parameters/<backend>/<category>.yml``
        and is used by the magmom workflow, whose per-backend preset
        tables live in ``parameters/abacus/magmom.yml`` and
        ``parameters/vasp/magmom.yml``. The category key typically
        matches the workflow name.

        Workflow-protocol slots (``smear`` / ``magmom`` / ``convergence``)
        live in ``parameters/<protocol>.yml`` at the top of
        ``parameters/`` and are handled by :meth:`_load_protocol`; they
        are skipped here.

        A preset may be written in two equivalent layouts:

        * flat — ``preset = {parameters: ..., ...}``
        * nested — ``preset = {<backend>: {parameters: ..., ...}}``

        In both cases the adapter receives the same backend-native
        section (``{parameters: ..., <backend-specific keys>: ...}``).
        The nested form is convenient when several backends share a
        single preset file.

        In every case ``out[backend]`` is a list of presets and the
        orchestrator submits one WorkChain per preset, in order.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        names_out: dict[str, list[str]] = {}
        parameters = self.input_params.get("parameters", {})
        for backend, param_name in parameters.items():
            # Workflow-protocol slots (``smear`` / ``magmom`` / ...) live
            # in ``parameters/<protocol>.yml`` and are handled by
            # ``_load_protocol``; skip them here.
            if backend not in self.BACKEND_DIRS:
                continue

            # Dict form: ``{"<category>": preset_or_list}`` — picks a
            # category-specific preset table under ``parameters/<backend>/``.
            if isinstance(param_name, dict):
                for category, category_value in param_name.items():
                    if not isinstance(category, str) or not category:
                        raise ValueError(
                            f"parameters['{backend}'] dict keys must be "
                            f"non-empty strings, got {category!r}"
                        )
                    table = read_yaml(
                        PARAMETERS_DIR / backend / f"{category}.yml"
                    )
                    preset_names = self._coerce_preset_names(
                        backend, category, category_value
                    )
                    presets = out.setdefault(backend, [])
                    presets.extend(
                        self._resolve_presets(
                            backend,
                            table,
                            preset_names,
                            source=f"parameters/{backend}/{category}.yml",
                        )
                    )
                    names_out.setdefault(backend, []).extend(preset_names)
                continue

            preset_names = self._coerce_preset_names(backend, backend, param_name)
            if not preset_names:
                continue

            table = read_yaml(PARAMETERS_DIR / backend / f"{backend}.yml")
            out[backend] = self._resolve_presets(
                backend,
                table,
                preset_names,
                source=f"parameters/{backend}/{backend}.yml",
            )
            names_out[backend] = list(preset_names)
        return out, names_out

    @staticmethod
    def _coerce_preset_names(
        backend: str, slot: str, value: Any
    ) -> list[str]:
        """Coerce a backend-slot value into a flat list of preset names.

        Accepts either a single string or a list/tuple of strings.
        Anything else raises a :class:`TypeError` referencing the
        ``slot`` (which may differ from ``backend`` when the dict
        sub-category form is used).
        """
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            if not all(isinstance(p, str) for p in value):
                raise TypeError(
                    f"parameters['{backend}']['{slot}'] must be a string or "
                    f"list of strings, got {value!r}"
                )
            return list(value)
        raise TypeError(
            f"parameters['{backend}']['{slot}'] must be a string or list of "
            f"strings, got {value!r}"
        )

    @staticmethod
    def _resolve_presets(
        backend: str,
        table: dict[str, Any],
        preset_names: list[str],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        """Resolve preset names into loaded YAML sections.

        Each preset can be either ``{parameters: ..., ...}`` (flat) or
        ``{<backend>: {parameters: ..., ...}, sibling: ...}`` (nested
        layout — sibling keys survive the unwrap). This convention is
        shared by every backend preset file.
        """
        presets: list[dict[str, Any]] = []
        for name in preset_names:
            if name not in table:
                raise KeyError(
                    f"Parameter set '{name}' not found in {source}; "
                    f"available: {list(table)}"
                )
            preset = table[name]
            # When the preset nests the backend key (``<backend>:``),
            # unwrap it but preserve sibling keys like kpoints_distance,
            # pseudo_family that are at the same level as the backend key.
            if isinstance(preset, dict) and backend in preset:
                unwrapped = dict(preset[backend])
                for key, value in preset.items():
                    if key != backend:
                        unwrapped[key] = value
                preset = unwrapped
            presets.append(preset)
        return presets

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from static YAML."""
        name = self.input_params["static"]["metadata"]
        table = read_yaml(STATIC_DIR / "metadata.yml")
        if name not in table:
            raise KeyError(f"Metadata '{name}' not found in metadata.yml")
        return table[name]
