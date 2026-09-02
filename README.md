# aiida-uranium-workflow

AiiDA workflow orchestration for uranium (and other) DFT calculations across
**ABACUS / VASP / FLEUR**. One JSON input drives parameter sweeps, property
calculations and cross-code comparisons; one CLI produces reports, figures,
archives and copied remote folders.

```
input.json  ──►  aiida-uranium run     ──►  WorkChains in AiiDA
                     │
                     └──►  output.json  ──►  aiida-uranium report / plot / archive / copy
```

## Features

* **8 methods × 3 backends** — smear, convergence, magmom, banddos, relax,
  elastic, EOS, phonopy (see the support matrix below).
* **JSON-driven orchestration** — `input.json` (workflow, per-backend presets,
  structure, profile, codes) → `output.json` (WorkChain UUID map). No Python
  scripting needed for routine runs.
* **Unified CLI** — `aiida-uranium {run, report, archive, copy}` plus
  `aiida-uranium-plot-banddos` / `aiida-uranium-plot-phonon`.
* **Markdown reports** per method (`utils/report/`), **matplotlib figures**
  (`utils/plot/`), **AiiDA archive export** and **remote-folder copy**.
* **Band comparison** — pairwise η_v / max η / ω metrics (PRB 98, 085117,
  BFC) for cross-code band-structure agreement, with k-point weights,
  energy-window state pairing and optional occupied-only max η.
* **AiiDA-free pure functions** — comparison, elastic fitting and extraction
  layers are plain numpy/pymatgen code, unit-testable without a profile.

## Support matrix

| Method            | ABACUS | VASP | FLEUR |
|-------------------|:------:|:----:|:-----:|
| `smear`           |   ✓    |  ✓   |       |
| `convergence`     |   ✓    |  ✓   |       |
| `magmom`          |   ✓    |  ✓   |   ✓   |
| `banddos`         |        |      |   ✓   |
| `relax`           |   ✓*   |      |   ✓*  |
| `elastic`         |   ✓    |      |   ✓   |
| `eos`             |   ✓    |      |   ✓   |
| `phonopy`         |   ✓    |      |       |

\* `relax` has no custom WorkChain — it calls the plugin relax WorkChains
(`abacus.relax` / `fleur.relax`) directly.

## Quick start

### 1. Install

```bash
pip install -e .          # from the repo root (src layout)
verdi profile setup ...   # or reuse an existing profile
verdi code setup ...      # abacus / vasp / pw.x codes
```

> **Entry points**: after any change to `[project.entry-points."aiida.workflows"]`
> re-run `pip install -e .` and, if AiiDA cached the old list,
> `verdi devel refresh-entry-point-cache`.

### 2. Write an input JSON

```json
{
  "workflow": "magmom",
  "parameters": {
    "abacus": {"magmom": "test"},
    "vasp":   {"magmom": "test"},
    "fleur":  {"magmom": "test"}
  },
  "static": {"structure": "bcc-uranium", "metadata": "yeesuan"},
  "profile": "aiida_profile",
  "code": {
    "abacus": "abacus@yeesuan",
    "vasp":   "vaspstd@yeesuan",
    "fleur":  "fleur@yeesuan"
  }
}
```

Per-method JSON layouts live under `src/aiida_uranium_workflow/example/`;
real case files (banddos / relax_test / elastic_test / magmom_test / …) live
in the companion `aiida-uranium-scripts/` repository.

### 3. Run, report, plot

```bash
aiida-uranium run -i input.json --method magmom
aiida-uranium report -i output.json --method magmom     # Markdown report
aiida-uranium-plot-banddos -i band_compare.json         # band / DOS / compare figures
```

## Architecture

```
aiida_uranium_workflow/
├── workflows/        custom WorkChains (wrap plugin WorkChains)
│   └── <method>/<backend>.py
├── input_builders/   SoftwareAdapter: parameters/*.yml  →  AiiDA inputs dict
│   └── <method>/<backend>.py          (_workchain_entry_point, _build_workchain_inputs)
├── schedulers/       WorkflowOrchestrator + register_workflow() registry
│   └── <method>.py                    (ADAPTERS / BACKENDS / PRESET_SUBKEYS)
├── parameters/       per-method, per-backend preset YAML
│   ├── <method>.yml                   top-level protocol blocks
│   └── <backend>/<subkey>.yml         shared SCF base (scf.yml), magmom, …
├── utils/
│   ├── config.py     ConfigLoader (presets, no silent fallback)
│   ├── plot/         extract.py (backend-agnostic BandData/DosData),
│   │                 compare.py (η_v / max η / ω), plot.py, phonon.py
│   ├── report/       Markdown report generators per method
│   └── elastic.py    pymatgen-based deformation / fitting (stress & energy)
├── cli/              aiida-uranium {run,report,archive,copy} + plot CLIs
└── static/           structure / metadata presets
```

Key conventions (see `AGENT.md` for the full spec):

* **Entry points** are `uranium.<method>.<backend>` — never bare
  `abacus.*` / `vasp.*` / `fleur.*` (those belong to the plugins).
* **Single source of truth for parameters** — presets are read from
  `parameters/`, missing presets raise instead of silently falling back.
* **Adding a method** = one WorkChain (optional), one adapter per backend,
  one scheduler (self-registering), one report generator, one `MethodSpec`
  entry in `cli/_common.py`.

## Development

```bash
pip install -e '.[dev]'
pytest tests/            # 400+ tests, pure-function tests need no AiiDA profile
```

See `AGENT.md` for repository conventions (naming, entry points, parameters,
testing, how to add a method) and `docs/` for the design notes.

## License

MIT
