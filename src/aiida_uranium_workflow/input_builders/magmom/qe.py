"""QE (pw.x) input builder for the magmom workflow.

Reads the per-backend SCF base from ``parameters/qe/magmom.yml`` (pw
parameters / k-points / SSSP pseudo family) and assembles inputs for
``QeMagmomWorkChain``. Pseudopotentials and the recommended cutoffs come
from the SSSP pseudo family (``pseudo_family`` in the preset):
``get_pseudos(structure=…)`` for the {element: UpfData} mapping and
``get_recommended_cutoffs(structure=…, unit='Ry')`` for ecutwfc/ecutrho.

QE sets the initial magnetisation **per species** (SYSTEM.
starting_magnetization), not per atom. Anti-ferromagnetic seeds are
therefore expressed with split species labels: if a ``magmom_list``
entry uses keys like ``U1`` / ``U2`` (same element, different label),
the single-kind structure is relabelled alternately (site i → species
``i % n``) and the pseudopotential of the common element is mapped to
every split species.
"""

from __future__ import annotations

from aiida.orm import load_group
from ..base import SoftwareAdapter
from typing import Any, Dict

import copy


class QeMagmomAdapter(SoftwareAdapter):
    """Translate a ParamBundle into QeMagmomWorkChain inputs."""

    name = "qe"

    def _workchain_entry_point(self) -> str:
        return "uranium.magmom.qe"

    # ------------------------------------------------------------------
    # Species splitting (AFM support)
    # ------------------------------------------------------------------

    def _magmom_entries(self) -> list[dict]:
        """Raw ``magmom_list`` entries from the protocol."""
        lists = self.workflow_data.get("magmom_lists", {}).get("qe", {})
        return list(lists.get("magmom_list", []))

    def _split_structure(self, structure, species: list[str]):
        """Relabel the sites alternately with the given species names.

        Returns a new StructureData where site ``i`` gets kind
        ``species[i % len(species)]``, keeping each site's cartesian
        position and element symbol. The split species all map to the
        same pseudopotential (their common element).
        """
        from aiida.orm import StructureData

        elements = {k.symbols[0] for k in structure.kinds}
        if len(elements) != 1:
            raise ValueError(
                "split-species magmom entries (e.g. U1/U2) require a "
                f"single-element structure; found {sorted(elements)}"
            )
        element = next(iter(elements))
        sd = StructureData(cell=structure.cell, pbc=True)
        for i, site in enumerate(structure.sites):
            # ``name`` creates a kind with the given label (e.g. U1 / U2);
            # all split species share the element's pseudopotential.
            sd.append_atom(
                position=site.position,
                symbols=[element],
                name=species[i % len(species)],
            )
        return sd

    @staticmethod
    def _expand_entries(entries: list[dict], species: list[str]) -> list[dict]:
        """Expand element-level entries to all split species.

        ``{"U": 2.0}`` → ``{"U1": 2.0, "U2": 2.0}`` (all sites get the
        same initial moment); species-level entries are passed through
        with missing species defaulting to 0.0.
        """
        if not species:
            return entries
        element = species[0].rstrip("0123456789")
        expanded = []
        for entry in entries:
            if set(entry) <= {element}:
                value = entry.get(element, 0.0)
                expanded.append({s: value for s in species})
            else:
                expanded.append({s: entry.get(s, 0.0) for s in species})
        return expanded

    def _prepare_structure_and_entries(self, structure):
        """Return ``(structure, magmom_entries)`` with species splitting
        applied when any entry uses split-species keys (e.g. U1/U2)."""
        entries = self._magmom_entries()
        if not entries:
            return structure, entries

        elements = {k.symbols[0] for k in structure.kinds}
        species_keys = sorted({
            k for entry in entries for k in entry if k not in elements
        })
        if not species_keys:
            return structure, entries  # element-level (NM / FM)

        # Derive the split species list (U1..Un) from the keys.
        base = species_keys[0].rstrip("0123456789")
        if any(k.rstrip("0123456789") != base for k in species_keys):
            raise ValueError(
                f"split-species keys must share one element prefix: {species_keys}"
            )
        indices = sorted(int(k[len(base):] or "1") for k in species_keys)
        species = [f"{base}{i}" for i in indices]
        structure = self._split_structure(structure, species)
        entries = self._expand_entries(entries, species)
        return structure, entries

    # ------------------------------------------------------------------

    def _build_workchain_inputs(self, structure, magmom_entries) -> dict[str, Any]:
        from aiida import orm

        if not magmom_entries:
            raise ValueError(
                "qe magmom protocol has an empty magmom_list; check the "
                "'qe' block of parameters/magmom.yml (e.g. test_u_afm_qe)."
            )

        # The preset's pw parameters live under the ``pw`` block
        # (k-points / pseudo family are top-level keys).
        pw_block = self.software_params.get("pw", self.software_params)
        params = copy.deepcopy(pw_block.get("parameters", {}))
        options = self.metadata.get("options", {})

        group = self._load_family()

        # Pseudos keyed by the structure's kind names; split species
        # (U1/U2) all map to the pseudopotential of their element (U).
        # Fetch by element symbols so the family lookup is unambiguous.
        elements = {element for kind in structure.kinds for element in kind.symbols}
        by_element = group.get_pseudos(elements=sorted(elements))
        pseudos = {}
        for kind in structure.kinds:
            element = kind.symbols[0]
            try:
                pseudos[kind.name] = by_element[element]
            except KeyError:
                raise ValueError(
                    f"pseudo family {self.software_params.get('pseudo_family')!r} "
                    f"has no pseudopotential for element {element!r} "
                    f"(structure kinds: {[k.name for k in structure.kinds]})"
                ) from None

        # SSSP recommended cutoffs (Ry) — override the preset reference
        # values so the run always uses the family's own recommendation.
        system = params.setdefault("SYSTEM", {})
        try:
            ecutwfc, ecutrho = group.get_recommended_cutoffs(
                structure=structure, unit="Ry"
            )
            system["ecutwfc"] = float(ecutwfc)
            system["ecutrho"] = float(ecutrho)
        except (AttributeError, TypeError, ValueError):
            pass  # keep the preset cutoffs when the family has none

        inputs: dict[str, Any] = {
            "pw": {
                "code": orm.load_code(self.code_label),
                "parameters": orm.Dict(params),
                "pseudos": pseudos,
                "structure": structure,
                "metadata": {"options": options} if options else {},
            },
        }
        if "kpoints_distance" in self.software_params:
            inputs["kpoints_distance"] = orm.Float(
                float(self.software_params["kpoints_distance"])
            )
        elif "kpoints_mesh" in self.software_params:
            from aiida.plugins import DataFactory

            KpointsData = DataFactory("core.array.kpoints")
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_kpoints_mesh(list(self.software_params["kpoints_mesh"]))
            inputs["kpoints"] = kpoints_mesh
        return inputs

    def _load_family(self):
        """Return the pseudo family group named in the preset."""
        family = str(self.software_params.get("pseudo_family", ""))
        if not family:
            raise ValueError(
                "qe magmom preset is missing 'pseudo_family' "
                "(e.g. SSSP/1.3/PBE/efficiency)."
            )
        return load_group(family)

    def _prepare_workflow_inputs(self) -> Dict[str, list]:
        """Extract the ``magmom_list`` from workflow_data."""
        lists = self.workflow_data.get("magmom_lists", {}).get("qe", {})
        return {
            "magmom_list": list(lists.get("magmom_list", [])),
        }

    def adapt(self, structure):
        """Compose the final AiiDA inputs + workchain class."""
        from aiida import orm
        from aiida.plugins import WorkflowFactory

        options = self.metadata.get("options", {})
        structure, magmom_entries = self._prepare_structure_and_entries(structure)
        if not magmom_entries:
            raise ValueError(
                "qe magmom protocol has an empty magmom_list; check the "
                "'qe' block of parameters/magmom.yml."
            )

        inputs = self._build_workchain_inputs(structure, magmom_entries)
        inputs["magmom_list"] = orm.List(list=magmom_entries)

        self._inject_options(inputs, options)

        return self.AdaptedInputs(
            workchain_cls=WorkflowFactory(self._workchain_entry_point()),
            inputs=inputs,
        )

    AdaptedInputs = __import__(
        "aiida_uranium_workflow.input_builders.base", fromlist=["AdaptedInputs"]
    ).AdaptedInputs
