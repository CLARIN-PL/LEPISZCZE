# LEPISZCZE LLM Evaluation Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible LLM evaluation track to LEPISZCZE — prompting, QLoRA fine-tuning and a refreshed encoder baseline over the existing 17 dataset variants — and publish the results on the existing leaderboard with quality *and* inference-cost columns.

**Architecture:** A standalone `llm-eval/` sub-project (uv, Python 3.11) that never imports the legacy `lepiszcze` package. Tasks declare a JSON output schema; a vLLM backend generates under guided decoding; an append-only per-example prediction store makes every run resumable on shared, preemptible GPUs; a scorer aggregates predictions into the same result JSON shape the historical `embeddings` repo produced, which `scripts/build_leaderboard.py` then merges into `docs/data/leaderboard.json` under schema v3.

**Tech Stack:** Python 3.11, uv, vLLM (+ xgrammar guided decoding), transformers, peft/trl, bitsandbytes, datasets, seqeval, pynvml, pydantic, typer, pytest.

**Spec:** `planning/specs/2026-09-01-llm-eval-track.md`

## Global Constraints

- Python `>=3.11,<3.13` for `llm-eval/`. The legacy root `pyproject.toml` (Python 3.9, torch 1.11, transformers 4.18) is **never modified and never imported**.
- Dependency manager for the sub-project is `uv`. All commands below run from `llm-eval/` unless a path says otherwise.
- Random seed is `441` everywhere — it is the seed already used in `experiments/configs/datasets/*.yaml`.
- Subsample cap for large test splits is exactly `2000` examples, stratified, seed `441`.
- Prompt variants per task for track T1: exactly `5`, indexed `0..4`.
- Every Hugging Face model and dataset is pinned by an explicit `revision` (40-hex commit SHA). Unpinned means broken reproduction.
- Ampere GPUs only (RTX 3090 / A5000 / A4000): **no FP8**. Quantization is `awq-int4` or `gptq-int4` with Marlin kernels, or plain `bf16`.
- One model = one GPU. No cross-host or cross-GPU tensor parallelism anywhere in this plan.
- Every test in `llm-eval/tests/` must pass on a machine with **no GPU**. GPU-only tests carry `@pytest.mark.gpu` and are deselected by default via `addopts = "-m 'not gpu'"`.
- Metric names must match the historical set exactly, or the leaderboard UI will show a new column: `accuracy`, `f1_macro`, `f1_micro`, `f1_weighted`, `precision_macro`, `precision_micro`, `precision_weighted`, `recall_macro`, `recall_micro`, `recall_weighted`.
- Commit style: Conventional Commits, imperative, lowercase subject, ≤72 chars, `Refs #5` in the footer.
- Never write `Co-Authored-By` or any tool attribution into a commit message.

## Working Environment

Create the implementation worktree before Task 1:

```bash
cd /home/laugustyniak/github/pwr/lepiszcze-benchmark/LEPISZCZE
git fetch origin
git worktree add .worktrees/feat-5-llm-eval-track -b feat/5-llm-eval-track origin/main
cd .worktrees/feat-5-llm-eval-track
```

All paths in this plan are relative to that worktree root.

## File Structure

```
llm-eval/
  pyproject.toml                    uv project, deps, pytest config
  README.md                         one-command reproduction path
  llm_eval/
    __init__.py
    registry.py                     ModelSpec + MODELS dict, get_model()
    tasks/
      __init__.py                   TASKS dict, get_task()
      base.py                       Example, ParseResult, Task protocol
      classification.py             8 classification dataset variants
      sequence_labeling.py          5 token-level datasets, span<->BIO alignment
      qa.py                         4 extractive QA variants
    subsets.py                      frozen stratified subsampling + manifests
    prompts/
      __init__.py                   PromptTemplate, render(), demo selection
      classification.yaml           5 variants x 8 datasets
      sequence_labeling.yaml        5 variants x 5 datasets
      qa.yaml                       5 variants x 4 datasets
    backends/
      __init__.py                   GenerationRequest/Result, Backend protocol
      dummy.py                      deterministic CPU backend for tests
      vllm_backend.py               vLLM offline engine + guided decoding
    cost.py                         NVML energy/VRAM/GPU-seconds meter
    runner.py                       resumable append-only generation loop
    score.py                        preds -> per-variant metrics
    aggregate.py                    variants -> metrics_avg/metrics_std result JSON
    finetune/
      __init__.py
      lora.py                       T2 QLoRA SFT
      encoder.py                    T3 encoder fine-tuning
    contamination.py                n-gram overlap + no-input probe
    cli.py                          typer app: plan/run/score/export/status/audit
  tests/
    conftest.py                     fixtures: tiny synthetic tasks, DummyBackend
    test_registry.py
    test_subsets.py
    test_classification.py
    test_prompts.py
    test_backends.py
    test_runner.py
    test_cost.py
    test_sequence_labeling.py
    test_qa.py
    test_aggregate.py
    test_contamination.py
    test_cli.py
scripts/
  build_leaderboard.py              MODIFIED: schema v3, --llm-results-dir, --base
  queue_agenda.sh                   NEW: task-spooler job agenda for shared GPUs
docs/assets/
  leaderboard.js                    MODIFIED: protocol badge + cost column
results/
  preds/<model_id>/<task>/<variant>.jsonl        append-only predictions
  preds/<model_id>/<task>/<variant>.meta.json    cost metadata
  llm/<model_id>__<task>.json                    aggregated, leaderboard-ready
data/subsets/<task>.json                         frozen subsample manifests
.github/workflows/llm_eval.yml     NEW: CPU-only test job
```

---

### Task 1: Sub-project scaffold and CI

**Files:**
- Create: `llm-eval/pyproject.toml`
- Create: `llm-eval/llm_eval/__init__.py`
- Create: `llm-eval/tests/test_smoke.py`
- Create: `.github/workflows/llm_eval.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `llm_eval` with `llm_eval.__version__: str`; a `uv run pytest` entry point that later tasks extend.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_smoke.py`:

```python
import llm_eval


def test_package_exposes_version():
    assert llm_eval.__version__ == "0.1.0"


def test_package_declares_seed():
    assert llm_eval.SEED == 441
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval'`

- [ ] **Step 3: Write the project definition**

Create `llm-eval/pyproject.toml`:

```toml
[project]
name = "lepiszcze-llm-eval"
version = "0.1.0"
description = "LLM evaluation track for the LEPISZCZE benchmark"
requires-python = ">=3.11,<3.13"
dependencies = [
    "datasets>=3.0",
    "huggingface-hub>=0.26",
    "numpy>=1.26",
    "nvidia-ml-py>=12.560",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "scikit-learn>=1.5",
    "seqeval>=1.2.2",
    "typer>=0.12",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.5",
    "transformers>=4.46",
    "vllm>=0.7",
    "peft>=0.13",
    "trl>=0.12",
    "bitsandbytes>=0.44",
    "accelerate>=1.0",
]

[project.scripts]
llm-eval = "llm_eval.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["llm_eval"]

[tool.pytest.ini_options]
addopts = "-m 'not gpu' -q"
markers = ["gpu: requires a CUDA device and the [gpu] extra"]
testpaths = ["tests"]
```

Create `llm-eval/llm_eval/__init__.py`:

```python
"""LLM evaluation track for the LEPISZCZE benchmark."""

__version__ = "0.1.0"

# Seed used by every dataset config in experiments/configs/datasets/*.yaml.
SEED = 441
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd llm-eval && uv run pytest tests/test_smoke.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Add the CI job**

Create `.github/workflows/llm_eval.yml`:

```yaml
name: LLM eval track

on:
  pull_request:
  push:
    branches: [main]
    paths:
      - "llm-eval/**"
      - ".github/workflows/llm_eval.yml"
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: llm-eval-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: CPU test suite
    runs-on: ubuntu-24.04
    defaults:
      run:
        working-directory: llm-eval
    steps:
      - name: Check out repository
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Run tests
        run: uv run --frozen pytest -v
```

- [ ] **Step 6: Verify the workflow is valid YAML and the suite runs clean**

Run: `cd llm-eval && uv lock && uv run pytest -v && python3 -c "import yaml,sys; yaml.safe_load(open('../.github/workflows/llm_eval.yml'))"`
Expected: tests pass, no YAML exception

- [ ] **Step 7: Commit**

```bash
git add llm-eval/pyproject.toml llm-eval/uv.lock llm-eval/llm_eval/__init__.py llm-eval/tests/test_smoke.py .github/workflows/llm_eval.yml
git commit -m "feat(llm-eval): scaffold llm evaluation sub-project

Refs #5"
```

---

### Task 2: Model registry with pinned Hub revisions

**Files:**
- Create: `llm-eval/llm_eval/registry.py`
- Create: `llm-eval/tests/test_registry.py`

**Interfaces:**
- Consumes: `llm_eval.SEED` (not used here, but the module is importable).
- Produces:
  - `ModelSpec` frozen dataclass with fields `id: str`, `hf_id: str`, `revision: str`, `family: str`, `base_model_id: str | None`, `params_b: float`, `quantization: str`, `license: str`, `max_model_len: int`, `is_instruct: bool`, `track: tuple[str, ...]`.
  - `MODELS: dict[str, ModelSpec]` keyed by `ModelSpec.id`.
  - `get_model(model_id: str) -> ModelSpec` raising `KeyError` with the sorted list of known ids.
  - `paired_models() -> list[tuple[ModelSpec, ModelSpec]]` returning `(adapted, base)` pairs for RQ1.
  - `resolve_revision(hf_id: str) -> str` hitting the Hub API (network, `@pytest.mark.gpu`-free but marked `network`).

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_registry.py`:

```python
import re

import pytest

from llm_eval.registry import MODELS, ModelSpec, get_model, paired_models

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_ids_are_filesystem_safe_and_unique():
    assert MODELS
    for model_id, spec in MODELS.items():
        assert model_id == spec.id
        assert re.fullmatch(r"[a-z0-9._-]+", model_id), model_id


def test_every_model_pins_a_full_sha_revision():
    for spec in MODELS.values():
        assert SHA_RE.match(spec.revision), f"{spec.id} revision must be a 40-hex SHA"


def test_quantization_never_uses_fp8_on_ampere():
    for spec in MODELS.values():
        assert spec.quantization in {"bf16", "awq-int4", "gptq-int4"}, spec.id


def test_models_fit_a_single_24gb_gpu():
    for spec in MODELS.values():
        budget_gb = spec.params_b * (2.0 if spec.quantization == "bf16" else 0.6)
        assert budget_gb <= 20.0, f"{spec.id} needs ~{budget_gb:.1f} GB, over the 24 GB budget"


def test_paired_models_cover_the_rq1_comparisons():
    pairs = {(a.id, b.id) for a, b in paired_models()}
    assert ("llama-pllum-8b-instruct", "llama-3.1-8b-instruct") in pairs
    assert ("pllum-12b-instruct", "mistral-nemo-12b-instruct") in pairs


def test_base_model_ids_resolve_inside_the_registry():
    for spec in MODELS.values():
        if spec.base_model_id is not None:
            assert spec.base_model_id in MODELS, f"{spec.id} -> {spec.base_model_id}"


def test_get_model_raises_with_known_ids_listed():
    with pytest.raises(KeyError) as excinfo:
        get_model("does-not-exist")
    assert "bielik-11b-v3-instruct" in str(excinfo.value)


def test_get_model_returns_the_spec():
    spec = get_model("bielik-11b-v3-instruct")
    assert isinstance(spec, ModelSpec)
    assert spec.hf_id == "speakleash/Bielik-11B-v3.0-Instruct"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.registry'`

- [ ] **Step 3: Write the registry**

Create `llm-eval/llm_eval/registry.py`. Revisions below are placeholders of the literal
string `"0" * 40` **only until Step 5 fills them in** — Step 5 is not optional, and Step 4
will fail until it runs, which is intentional.

```python
"""Registry of every model evaluated in the LEPISZCZE LLM track.

Each entry is pinned by (hf_id, revision). An unpinned model makes the
benchmark irreproducible, so `resolve_revisions.py` regenerates the SHAs and
`tests/test_registry.py` fails when any of them is not a 40-hex commit.
"""

from __future__ import annotations

from dataclasses import dataclass

TRACK_PROMPTING = "T1"
TRACK_LORA = "T2"
TRACK_ENCODER = "T3"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_id: str
    revision: str
    family: str
    base_model_id: str | None
    params_b: float
    quantization: str
    license: str
    max_model_len: int
    is_instruct: bool
    track: tuple[str, ...]


def _spec(**kwargs) -> ModelSpec:
    return ModelSpec(**kwargs)


_ENTRIES: list[ModelSpec] = [
    _spec(
        id="bielik-11b-v3-instruct",
        hf_id="speakleash/Bielik-11B-v3.0-Instruct",
        revision="0" * 40,
        family="bielik",
        base_model_id=None,
        params_b=11.0,
        quantization="awq-int4",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="bielik-minitron-7b-v3-instruct",
        hf_id="speakleash/Bielik-Minitron-7B-v3.0-Instruct",
        revision="0" * 40,
        family="bielik",
        base_model_id=None,
        params_b=7.35,
        quantization="bf16",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="bielik-11b-v2.6-instruct",
        hf_id="speakleash/Bielik-11B-v2.6-Instruct",
        revision="0" * 40,
        family="bielik",
        base_model_id=None,
        params_b=11.0,
        quantization="awq-int4",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING,),
    ),
    _spec(
        id="mistral-nemo-12b-instruct",
        hf_id="mistralai/Mistral-Nemo-Instruct-2407",
        revision="0" * 40,
        family="mistral",
        base_model_id=None,
        params_b=12.2,
        quantization="awq-int4",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="pllum-12b-instruct",
        hf_id="CYFRAGOVPL/PLLuM-12B-instruct",
        revision="0" * 40,
        family="pllum",
        base_model_id="mistral-nemo-12b-instruct",
        params_b=12.2,
        quantization="awq-int4",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="llama-3.1-8b-instruct",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="0" * 40,
        family="llama",
        base_model_id=None,
        params_b=8.0,
        quantization="bf16",
        license="llama3.1",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="llama-pllum-8b-instruct",
        hf_id="CYFRAGOVPL/Llama-PLLuM-8B-instruct",
        revision="0" * 40,
        family="pllum",
        base_model_id="llama-3.1-8b-instruct",
        params_b=8.0,
        quantization="bf16",
        license="llama3.1",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING, TRACK_LORA),
    ),
    _spec(
        id="qra-7b",
        hf_id="OPI-PG/Qra-7b",
        revision="0" * 40,
        family="qra",
        base_model_id=None,
        params_b=7.0,
        quantization="bf16",
        license="llama2",
        max_model_len=4096,
        is_instruct=False,
        track=(TRACK_LORA,),
    ),
    _spec(
        id="eurollm-9b-instruct",
        hf_id="utter-project/EuroLLM-9B-Instruct",
        revision="0" * 40,
        family="eurollm",
        base_model_id=None,
        params_b=9.0,
        quantization="bf16",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING,),
    ),
    _spec(
        id="gpt-oss-20b",
        hf_id="openai/gpt-oss-20b",
        revision="0" * 40,
        family="gpt-oss",
        base_model_id=None,
        params_b=20.9,
        quantization="gptq-int4",
        license="apache-2.0",
        max_model_len=8192,
        is_instruct=True,
        track=(TRACK_PROMPTING,),
    ),
]

MODELS: dict[str, ModelSpec] = {spec.id: spec for spec in _ENTRIES}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as error:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model '{model_id}'. Known models: {known}") from error


def paired_models() -> list[tuple[ModelSpec, ModelSpec]]:
    """Return (Polish-adapted, base) pairs backing research question RQ1."""
    return [
        (spec, MODELS[spec.base_model_id])
        for spec in _ENTRIES
        if spec.base_model_id is not None
    ]
```

- [ ] **Step 4: Run test to verify the revision test fails and the rest pass**

Run: `cd llm-eval && uv run pytest tests/test_registry.py -v`
Expected: 7 passed, 1 failed — `test_every_model_pins_a_full_sha_revision`, with
`bielik-11b-v3-instruct revision must be a 40-hex SHA`

- [ ] **Step 5: Resolve and write real revisions**

Create `llm-eval/scripts/resolve_revisions.py`:

```python
"""Rewrite registry revision placeholders with the current Hub commit SHAs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

REGISTRY = Path(__file__).resolve().parents[1] / "llm_eval" / "registry.py"
ENTRY_RE = re.compile(
    r'hf_id="(?P<hf_id>[^"]+)",\n(?P<indent>\s*)revision="(?P<revision>[0-9a-f]{40})"'
)


def main() -> int:
    api = HfApi()
    source = REGISTRY.read_text(encoding="utf-8")
    failures: list[str] = []

    def replace(match: re.Match[str]) -> str:
        hf_id = match["hf_id"]
        try:
            sha = api.model_info(hf_id).sha
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{hf_id}: {error}")
            return match.group(0)
        return f'hf_id="{hf_id}",\n{match["indent"]}revision="{sha}"'

    REGISTRY.write_text(ENTRY_RE.sub(replace, source), encoding="utf-8")
    for failure in failures:
        print(f"UNRESOLVED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `cd llm-eval && uv run --extra gpu python scripts/resolve_revisions.py`
Expected: exit 0 and no `UNRESOLVED` lines. **Any `UNRESOLVED` line means that `hf_id` is
wrong or gated.** Fix the id (or request access on the Hub for `meta-llama/*`) and re-run.
Do not proceed with a placeholder revision.

- [ ] **Step 6: Run tests to verify they all pass**

Run: `cd llm-eval && uv run pytest tests/test_registry.py -v`
Expected: PASS, 8 passed

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/registry.py llm-eval/scripts/resolve_revisions.py llm-eval/tests/test_registry.py
git commit -m "feat(llm-eval): add model registry with pinned hub revisions

Refs #5"
```

---

### Task 3: Task protocol, dataset loading and frozen stratified subsets

**Files:**
- Create: `llm-eval/llm_eval/tasks/__init__.py`
- Create: `llm-eval/llm_eval/tasks/base.py`
- Create: `llm-eval/llm_eval/subsets.py`
- Create: `llm-eval/tests/conftest.py`
- Create: `llm-eval/tests/test_subsets.py`

**Interfaces:**
- Consumes: `llm_eval.SEED`.
- Produces:
  - `Example(id: str, inputs: dict[str, str], gold: object)` frozen dataclass.
  - `ParseResult(prediction: object | None, ok: bool, error: str | None = None)` frozen dataclass.
  - `TaskSpec(name, leaderboard_task, dataset, dataset_revision, kind, labels, max_new_tokens)` frozen dataclass.
  - `Task` protocol: `spec: TaskSpec`, `load(split: str) -> list[Example]`, `output_schema() -> dict`, `parse(raw: str, example: Example) -> ParseResult`, `score(golds: list, preds: list) -> dict[str, float]`, `stratum(example: Example) -> str`.
  - `SubsetManifest(task: str, seed: int, cap: int, example_ids: tuple[str, ...], sha256: str)` frozen dataclass with `.to_json()` / `SubsetManifest.from_json()`.
  - `freeze_subset(task: Task, examples: list[Example], cap: int = 2000, seed: int = SEED) -> SubsetManifest`.
  - `apply_subset(examples: list[Example], manifest: SubsetManifest) -> list[Example]`.
  - `subset_path(task_name: str, root: Path) -> Path` = `root / "data" / "subsets" / f"{task_name}.json"`.
  - Test fixture `tiny_task` (a 3-label classification task over 12 in-memory examples) and `dummy_backend`, both in `tests/conftest.py`.

- [ ] **Step 1: Write the shared test fixtures**

Create `llm-eval/tests/conftest.py`:

```python
from __future__ import annotations

import json

import pytest

from llm_eval.tasks.base import Example, ParseResult, TaskSpec

LABELS = ("positive", "negative", "neutral")


class TinyTask:
    """A 12-example, 3-label classification task used across the CPU test suite."""

    spec = TaskSpec(
        name="tiny",
        leaderboard_task="Sentiment Analysis",
        dataset="test/tiny",
        dataset_revision="0" * 40,
        kind="classification",
        labels=LABELS,
        max_new_tokens=8,
    )

    def load(self, split: str) -> list[Example]:
        assert split in {"train", "validation", "test"}
        return [
            Example(id=f"{split}-{index}", inputs={"text": f"zdanie {index}"}, gold=LABELS[index % 3])
            for index in range(12)
        ]

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"label": {"enum": list(LABELS)}}, "required": ["label"]}

    def parse(self, raw: str, example: Example) -> ParseResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            return ParseResult(prediction=None, ok=False, error=f"invalid json: {error}")
        label = payload.get("label")
        if label not in LABELS:
            return ParseResult(prediction=None, ok=False, error=f"unknown label: {label!r}")
        return ParseResult(prediction=label, ok=True)

    def score(self, golds: list, preds: list) -> dict[str, float]:
        correct = sum(1 for gold, pred in zip(golds, preds) if gold == pred)
        return {"accuracy": correct / len(golds)}

    def stratum(self, example: Example) -> str:
        return str(example.gold)


@pytest.fixture
def tiny_task() -> TinyTask:
    return TinyTask()
```

- [ ] **Step 2: Write the failing subset test**

Create `llm-eval/tests/test_subsets.py`:

```python
import json

from llm_eval.subsets import SubsetManifest, apply_subset, freeze_subset, subset_path


def test_no_subsampling_when_under_the_cap(tiny_task):
    examples = tiny_task.load("test")
    manifest = freeze_subset(tiny_task, examples, cap=100)
    assert len(manifest.example_ids) == len(examples)


def test_subsample_is_stratified_and_capped(tiny_task):
    examples = tiny_task.load("test")
    manifest = freeze_subset(tiny_task, examples, cap=6)
    selected = apply_subset(examples, manifest)
    assert len(selected) == 6
    strata = [tiny_task.stratum(example) for example in selected]
    assert sorted(strata) == ["negative", "negative", "neutral", "neutral", "positive", "positive"]


def test_subsample_is_deterministic_across_calls(tiny_task):
    examples = tiny_task.load("test")
    first = freeze_subset(tiny_task, examples, cap=6)
    second = freeze_subset(tiny_task, examples, cap=6)
    assert first.example_ids == second.example_ids
    assert first.sha256 == second.sha256


def test_manifest_round_trips_through_json(tiny_task):
    manifest = freeze_subset(tiny_task, tiny_task.load("test"), cap=6)
    restored = SubsetManifest.from_json(json.loads(manifest.to_json()))
    assert restored == manifest


def test_apply_subset_rejects_a_tampered_manifest(tiny_task):
    examples = tiny_task.load("test")
    manifest = freeze_subset(tiny_task, examples, cap=6)
    tampered = SubsetManifest(
        task=manifest.task,
        seed=manifest.seed,
        cap=manifest.cap,
        example_ids=manifest.example_ids[:-1] + ("test-0",),
        sha256=manifest.sha256,
    )
    try:
        apply_subset(examples, tampered)
    except ValueError as error:
        assert "sha256" in str(error)
    else:
        raise AssertionError("apply_subset must reject a manifest whose digest does not match")


def test_apply_subset_preserves_manifest_order(tiny_task):
    examples = tiny_task.load("test")
    manifest = freeze_subset(tiny_task, examples, cap=6)
    selected = apply_subset(examples, manifest)
    assert [example.id for example in selected] == list(manifest.example_ids)


def test_subset_path_layout(tmp_path):
    assert subset_path("nkjp_pos", tmp_path) == tmp_path / "data" / "subsets" / "nkjp_pos.json"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_subsets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.subsets'`

- [ ] **Step 4: Write the task base module**

Create `llm-eval/llm_eval/tasks/base.py`:

```python
"""Core data types every LEPISZCZE LLM task implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

KIND_CLASSIFICATION = "classification"
KIND_SEQUENCE_LABELING = "sequence_labeling"
KIND_QA = "qa"


@dataclass(frozen=True)
class Example:
    id: str
    inputs: dict[str, str]
    gold: Any


@dataclass(frozen=True)
class ParseResult:
    prediction: Any | None
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    leaderboard_task: str
    dataset: str
    dataset_revision: str
    kind: str
    labels: tuple[str, ...]
    max_new_tokens: int


@runtime_checkable
class Task(Protocol):
    spec: TaskSpec

    def load(self, split: str) -> list[Example]: ...

    def output_schema(self) -> dict: ...

    def parse(self, raw: str, example: Example) -> ParseResult: ...

    def score(self, golds: list[Any], preds: list[Any]) -> dict[str, float]: ...

    def stratum(self, example: Example) -> str: ...
```

Create `llm-eval/llm_eval/tasks/__init__.py`:

```python
"""Task registry. Concrete tasks register themselves in later modules."""

from __future__ import annotations

from llm_eval.tasks.base import (
    KIND_CLASSIFICATION,
    KIND_QA,
    KIND_SEQUENCE_LABELING,
    Example,
    ParseResult,
    Task,
    TaskSpec,
)

TASKS: dict[str, Task] = {}


def register(task: Task) -> Task:
    if task.spec.name in TASKS:
        raise ValueError(f"Task already registered: {task.spec.name}")
    TASKS[task.spec.name] = task
    return task


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as error:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{name}'. Known tasks: {known}") from error


__all__ = [
    "KIND_CLASSIFICATION",
    "KIND_QA",
    "KIND_SEQUENCE_LABELING",
    "Example",
    "ParseResult",
    "TASKS",
    "Task",
    "TaskSpec",
    "get_task",
    "register",
]
```

- [ ] **Step 5: Write the subsets module**

Create `llm-eval/llm_eval/subsets.py`:

```python
"""Frozen, stratified test subsets.

Large test splits (NKJP-POS, punctuation restoration) are too expensive to
generate for every model x prompt variant. They are capped at 2000 examples
with a stratified draw at a fixed seed, and the drawn ids are committed so the
subset never silently changes.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from llm_eval import SEED
from llm_eval.tasks.base import Example, Task

DEFAULT_CAP = 2000


@dataclass(frozen=True)
class SubsetManifest:
    task: str
    seed: int
    cap: int
    example_ids: tuple[str, ...]
    sha256: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "task": self.task,
                "seed": self.seed,
                "cap": self.cap,
                "example_ids": list(self.example_ids),
                "sha256": self.sha256,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"

    @staticmethod
    def from_json(payload: dict) -> "SubsetManifest":
        return SubsetManifest(
            task=str(payload["task"]),
            seed=int(payload["seed"]),
            cap=int(payload["cap"]),
            example_ids=tuple(str(value) for value in payload["example_ids"]),
            sha256=str(payload["sha256"]),
        )


def digest(example_ids: tuple[str, ...]) -> str:
    hasher = hashlib.sha256()
    for example_id in example_ids:
        hasher.update(example_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def freeze_subset(
    task: Task,
    examples: list[Example],
    cap: int = DEFAULT_CAP,
    seed: int = SEED,
) -> SubsetManifest:
    """Draw a reproducible, stratified subset, preserving the original order."""
    if len(examples) <= cap:
        selected_ids = tuple(example.id for example in examples)
        return SubsetManifest(task.spec.name, seed, cap, selected_ids, digest(selected_ids))

    buckets: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        buckets[task.stratum(example)].append(example.id)

    rng = random.Random(seed)
    quota_base, remainder = divmod(cap, len(buckets))
    chosen: set[str] = set()
    # Largest strata absorb the remainder so the quota sum is exactly `cap`.
    ordered = sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    for index, (_, ids) in enumerate(ordered):
        quota = min(len(ids), quota_base + (1 if index < remainder else 0))
        chosen.update(rng.sample(sorted(ids), quota))

    # A stratum smaller than its quota leaves the subset short; top it up.
    if len(chosen) < cap:
        leftovers = sorted({example.id for example in examples} - chosen)
        chosen.update(rng.sample(leftovers, cap - len(chosen)))

    selected_ids = tuple(example.id for example in examples if example.id in chosen)
    return SubsetManifest(task.spec.name, seed, cap, selected_ids, digest(selected_ids))


def apply_subset(examples: list[Example], manifest: SubsetManifest) -> list[Example]:
    if digest(manifest.example_ids) != manifest.sha256:
        raise ValueError(f"Manifest sha256 mismatch for task {manifest.task}")
    by_id = {example.id: example for example in examples}
    missing = [example_id for example_id in manifest.example_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} manifest ids are absent from the split, e.g. {missing[0]}")
    return [by_id[example_id] for example_id in manifest.example_ids]


def subset_path(task_name: str, root: Path) -> Path:
    return root / "data" / "subsets" / f"{task_name}.json"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_subsets.py -v`
Expected: PASS, 7 passed

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/tasks llm-eval/llm_eval/subsets.py llm-eval/tests/conftest.py llm-eval/tests/test_subsets.py
git commit -m "feat(llm-eval): add task protocol and frozen stratified subsets

Refs #5"
```

---

### Task 4: Classification tasks

**Files:**
- Create: `llm-eval/llm_eval/tasks/classification.py`
- Create: `llm-eval/tests/test_classification.py`

**Interfaces:**
- Consumes: `Example`, `ParseResult`, `TaskSpec`, `register` from `llm_eval.tasks`.
- Produces: 8 registered task names — `polemo2`, `polemo2_in`, `polemo2_out`, `abusive_clauses`, `cdsc_e`, `dyk`, `psc`, `dialogue_acts` — each a `ClassificationTask` instance. Public helpers: `ClassificationTask` class and `classification_metrics(golds, preds, labels) -> dict[str, float]`.
- Metric keys produced (must match the historical leaderboard exactly): `accuracy`, `f1_macro`, `f1_micro`, `f1_weighted`, `precision_macro`, `precision_micro`, `precision_weighted`, `recall_macro`, `recall_micro`, `recall_weighted`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_classification.py`:

```python
import pytest

from llm_eval.tasks import get_task
from llm_eval.tasks.base import Example
from llm_eval.tasks.classification import classification_metrics

EXPECTED_METRICS = {
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
}


def test_metric_names_match_the_historical_leaderboard():
    metrics = classification_metrics(["a", "b", "a"], ["a", "b", "b"], labels=("a", "b"))
    assert set(metrics) == EXPECTED_METRICS


def test_perfect_predictions_score_one():
    metrics = classification_metrics(["a", "b"], ["a", "b"], labels=("a", "b"))
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_unparsed_prediction_counts_as_wrong_not_dropped():
    metrics = classification_metrics(["a", "b"], ["a", None], labels=("a", "b"))
    assert metrics["accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "name,dataset,leaderboard_task",
    [
        ("polemo2", "clarin-pl/polemo2-official", "Sentiment Analysis"),
        ("polemo2_in", "allegro/klej-polemo2-in", "Sentiment Analysis"),
        ("polemo2_out", "allegro/klej-polemo2-out", "Sentiment Analysis"),
        ("abusive_clauses", "laugustyniak/abusive-clauses-pl", "Abusive Clauses Detection"),
        ("cdsc_e", "allegro/klej-cdsc-e", "Entailment Classification"),
        ("dyk", "allegro/klej-dyk", "Q&A Classification"),
        ("psc", "allegro/klej-psc", "Paraphrase Classification"),
        ("dialogue_acts", "clarin-pl/dialogue-acts", "Dialogue Acts Classification"),
    ],
)
def test_registered_tasks_match_the_leaderboard_identifiers(name, dataset, leaderboard_task):
    spec = get_task(name).spec
    assert spec.dataset == dataset
    assert spec.leaderboard_task == leaderboard_task
    assert spec.kind == "classification"
    assert spec.labels


def test_parse_accepts_the_guided_json_payload():
    task = get_task("polemo2")
    example = Example(id="x", inputs={"text": "t"}, gold="__label__meta_plus_m")
    result = task.parse('{"label": "__label__meta_plus_m"}', example)
    assert result.ok
    assert result.prediction == "__label__meta_plus_m"


def test_parse_rejects_an_out_of_vocabulary_label():
    task = get_task("polemo2")
    example = Example(id="x", inputs={"text": "t"}, gold="__label__meta_plus_m")
    result = task.parse('{"label": "wspaniale"}', example)
    assert not result.ok
    assert result.prediction is None
    assert "wspaniale" in result.error


def test_parse_rejects_malformed_json():
    task = get_task("polemo2")
    example = Example(id="x", inputs={"text": "t"}, gold="__label__meta_plus_m")
    result = task.parse("nie wiem", example)
    assert not result.ok
    assert "json" in result.error.lower()


def test_output_schema_constrains_generation_to_known_labels():
    task = get_task("cdsc_e")
    schema = task.output_schema()
    assert schema["properties"]["label"]["enum"] == list(task.spec.labels)
    assert schema["required"] == ["label"]
    assert schema["additionalProperties"] is False


def test_stratum_is_the_gold_label():
    task = get_task("dyk")
    example = Example(id="x", inputs={"question": "q", "answer": "a"}, gold="1")
    assert task.stratum(example) == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_classification.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.tasks.classification'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/tasks/classification.py`:

```python
"""Sentence- and pair-classification tasks on the LEPISZCZE leaderboard."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llm_eval.tasks import register
from llm_eval.tasks.base import KIND_CLASSIFICATION, Example, ParseResult, TaskSpec

# Sentinel written into predictions when the model produced unusable output.
# It is scored as an incorrect label, never dropped.
UNPARSED = "__unparsed__"


def classification_metrics(
    golds: list[Any],
    preds: list[Any],
    labels: tuple[str, ...],
) -> dict[str, float]:
    """Metric names mirror the historical `embeddings` result files exactly."""
    resolved = [UNPARSED if pred is None else pred for pred in preds]
    label_list = list(labels) + [UNPARSED]
    metrics = {"accuracy": float(accuracy_score(golds, resolved))}
    for average in ("macro", "micro", "weighted"):
        common = {
            "y_true": golds,
            "y_pred": resolved,
            "labels": label_list,
            "average": average,
            "zero_division": 0,
        }
        metrics[f"f1_{average}"] = float(f1_score(**common))
        metrics[f"precision_{average}"] = float(precision_score(**common))
        metrics[f"recall_{average}"] = float(recall_score(**common))
    return metrics


@dataclass
class ClassificationTask:
    spec: TaskSpec
    input_columns: tuple[str, ...]
    target_column: str
    label_map: dict[Any, str] = field(default_factory=dict)

    def load(self, split: str) -> list[Example]:
        from datasets import load_dataset

        rows = load_dataset(
            self.spec.dataset,
            split=split,
            revision=self.spec.dataset_revision,
        )
        examples: list[Example] = []
        for index, row in enumerate(rows):
            gold = row[self.target_column]
            examples.append(
                Example(
                    id=f"{self.spec.name}-{split}-{index}",
                    inputs={column: str(row[column]) for column in self.input_columns},
                    gold=self.label_map.get(gold, str(gold)),
                )
            )
        return examples

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"label": {"enum": list(self.spec.labels)}},
            "required": ["label"],
            "additionalProperties": False,
        }

    def parse(self, raw: str, example: Example) -> ParseResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            return ParseResult(prediction=None, ok=False, error=f"invalid json: {error}")
        if not isinstance(payload, dict):
            return ParseResult(prediction=None, ok=False, error="payload is not an object")
        label = payload.get("label")
        if label not in self.spec.labels:
            return ParseResult(prediction=None, ok=False, error=f"unknown label: {label!r}")
        return ParseResult(prediction=label, ok=True)

    def score(self, golds: list[Any], preds: list[Any]) -> dict[str, float]:
        return classification_metrics(golds, preds, self.spec.labels)

    def stratum(self, example: Example) -> str:
        return str(example.gold)


def _classification(
    name: str,
    leaderboard_task: str,
    dataset: str,
    labels: tuple[str, ...],
    input_columns: tuple[str, ...],
    target_column: str,
    label_map: dict[Any, str] | None = None,
) -> ClassificationTask:
    return register(
        ClassificationTask(
            spec=TaskSpec(
                name=name,
                leaderboard_task=leaderboard_task,
                dataset=dataset,
                # Filled by scripts/resolve_revisions.py in Task 3 of the data pass.
                dataset_revision="main",
                kind=KIND_CLASSIFICATION,
                labels=labels,
                max_new_tokens=16,
            ),
            input_columns=input_columns,
            target_column=target_column,
            label_map=label_map or {},
        )
    )


POLEMO_LABELS = (
    "__label__meta_plus_m",
    "__label__meta_minus_m",
    "__label__meta_zero",
    "__label__meta_amb",
)

polemo2 = _classification(
    "polemo2", "Sentiment Analysis", "clarin-pl/polemo2-official",
    POLEMO_LABELS, ("text",), "target",
)
polemo2_in = _classification(
    "polemo2_in", "Sentiment Analysis", "allegro/klej-polemo2-in",
    POLEMO_LABELS, ("sentence",), "target",
)
polemo2_out = _classification(
    "polemo2_out", "Sentiment Analysis", "allegro/klej-polemo2-out",
    POLEMO_LABELS, ("sentence",), "target",
)
abusive_clauses = _classification(
    "abusive_clauses", "Abusive Clauses Detection", "laugustyniak/abusive-clauses-pl",
    ("BEZPIECZNE_POSTANOWIENIE_UMOWNE", "KLAUZULA_ABUZYWNA"), ("text",), "class",
)
cdsc_e = _classification(
    "cdsc_e", "Entailment Classification", "allegro/klej-cdsc-e",
    ("NEUTRAL", "ENTAILMENT", "CONTRADICTION"), ("sentence_A", "sentence_B"), "entailment_judgment",
)
dyk = _classification(
    "dyk", "Q&A Classification", "allegro/klej-dyk",
    ("0", "1"), ("question", "answer"), "target",
)
psc = _classification(
    "psc", "Paraphrase Classification", "allegro/klej-psc",
    ("0", "1"), ("extract_text", "summary_text"), "label",
)
dialogue_acts = _classification(
    "dialogue_acts", "Dialogue Acts Classification", "clarin-pl/dialogue-acts",
    ("statement", "question", "backchannel", "other"), ("text",), "target",
)
```

- [ ] **Step 4: Verify the label vocabularies against the real datasets**

The label tuples above are the plan's best reconstruction. Before trusting them, dump the
real ones:

```bash
cd llm-eval && uv run python - <<'PY'
from datasets import load_dataset

for dataset, target in [
    ("clarin-pl/polemo2-official", "target"),
    ("allegro/klej-polemo2-in", "target"),
    ("allegro/klej-polemo2-out", "target"),
    ("laugustyniak/abusive-clauses-pl", "class"),
    ("allegro/klej-cdsc-e", "entailment_judgment"),
    ("allegro/klej-dyk", "target"),
    ("allegro/klej-psc", "label"),
    ("clarin-pl/dialogue-acts", "target"),
]:
    rows = load_dataset(dataset, split="train")
    feature = rows.features[target]
    names = getattr(feature, "names", None)
    print(dataset, "->", names or sorted({str(value) for value in rows[target]})[:12])
PY
```

Update `labels`, `input_columns`, `target_column` and `label_map` to whatever this prints.
`label_map` exists to turn integer `ClassLabel` ids into the readable strings the prompts
use. Re-run this snippet after every edit until the printed vocabulary and the registry
agree exactly.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_classification.py -v`
Expected: PASS, 16 passed

- [ ] **Step 6: Commit**

```bash
git add llm-eval/llm_eval/tasks/classification.py llm-eval/tests/test_classification.py
git commit -m "feat(llm-eval): add classification tasks with historical metric names

Refs #5"
```

---

### Task 5: Prompt templates, variants and few-shot demo selection

**Files:**
- Create: `llm-eval/llm_eval/prompts/__init__.py`
- Create: `llm-eval/llm_eval/prompts/classification.yaml`
- Create: `llm-eval/tests/test_prompts.py`

**Interfaces:**
- Consumes: `Example`, `Task` from `llm_eval.tasks.base`; `SEED`.
- Produces:
  - `PromptTemplate(kind: str, variant: int, system: str, user: str)` frozen dataclass.
  - `TaskPrompt(task: str, instruction: str, label_glossary: dict[str, str], fields: tuple[str, ...])` frozen dataclass.
  - `N_VARIANTS = 5`.
  - `get_template(kind: str, variant: int) -> PromptTemplate`.
  - `get_task_prompt(task_name: str) -> TaskPrompt`.
  - `select_demos(train: list[Example], task: Task, shots: int, variant: int, seed: int = SEED) -> list[Example]` — stratified, disjoint across variants where the pool allows.
  - `render(task: Task, example: Example, variant: int, shots: int, train: list[Example]) -> list[dict[str, str]]` returning OpenAI-style chat messages.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_prompts.py`:

```python
import pytest

from llm_eval.prompts import N_VARIANTS, get_task_prompt, get_template, render, select_demos
from llm_eval.tasks import TASKS, get_task


def test_five_variants_exist_for_every_kind():
    for kind in ("classification", "sequence_labeling", "qa"):
        variants = {get_template(kind, variant).variant for variant in range(N_VARIANTS)}
        assert variants == set(range(N_VARIANTS))


def test_variants_are_textually_distinct():
    systems = {get_template("classification", variant).system for variant in range(N_VARIANTS)}
    assert len(systems) == N_VARIANTS


def test_every_registered_task_has_a_prompt():
    for name in TASKS:
        prompt = get_task_prompt(name)
        assert prompt.instruction.strip()


def test_label_glossary_covers_every_label():
    for name, task in TASKS.items():
        if task.spec.kind != "classification":
            continue
        glossary = get_task_prompt(name).label_glossary
        assert set(glossary) == set(task.spec.labels), name


def test_zero_shot_render_produces_system_and_user_messages(tiny_task):
    messages = render(tiny_task, tiny_task.load("test")[0], variant=0, shots=0, train=[])
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "zdanie 0" in messages[-1]["content"]


def test_few_shot_render_interleaves_demonstrations(tiny_task):
    train = tiny_task.load("train")
    messages = render(tiny_task, tiny_task.load("test")[0], variant=0, shots=3, train=train)
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]


def test_demo_assistant_turns_are_valid_task_output(tiny_task):
    train = tiny_task.load("train")
    messages = render(tiny_task, tiny_task.load("test")[0], variant=1, shots=2, train=train)
    for message, demo in zip(messages[2::2], select_demos(train, tiny_task, 2, variant=1)):
        result = tiny_task.parse(message["content"], demo)
        assert result.ok, message["content"]


def test_demos_are_stratified_across_labels(tiny_task):
    demos = select_demos(tiny_task.load("train"), tiny_task, shots=3, variant=0)
    assert sorted(demo.gold for demo in demos) == ["negative", "neutral", "positive"]


def test_demo_selection_is_deterministic_per_variant(tiny_task):
    train = tiny_task.load("train")
    first = [demo.id for demo in select_demos(train, tiny_task, 3, variant=2)]
    second = [demo.id for demo in select_demos(train, tiny_task, 3, variant=2)]
    assert first == second


def test_different_variants_draw_different_demos(tiny_task):
    train = tiny_task.load("train")
    drawn = {
        tuple(demo.id for demo in select_demos(train, tiny_task, 3, variant=variant))
        for variant in range(N_VARIANTS)
    }
    assert len(drawn) > 1


def test_render_never_leaks_the_gold_label_of_the_query(tiny_task):
    example = tiny_task.load("test")[0]
    messages = render(tiny_task, example, variant=0, shots=0, train=[])
    assert str(example.gold) not in messages[-1]["content"]


def test_get_template_rejects_an_unknown_variant():
    with pytest.raises(KeyError):
        get_template("classification", 99)


def test_polemo_prompt_mentions_the_task_domain():
    assert "recenzj" in get_task_prompt("polemo2").instruction.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.prompts'`

- [ ] **Step 3: Write the prompt data file**

Create `llm-eval/llm_eval/prompts/classification.yaml`. Five wrapper variants shared by all
classification tasks, plus per-task instruction and label glossary:

```yaml
variants:
  - system: "Jesteś precyzyjnym anotatorem języka polskiego. Odpowiadasz wyłącznie obiektem JSON."
    user: "{instruction}\n\nDopuszczalne etykiety:\n{glossary}\n\n{fields}\n\nZwróć JSON: {{\"label\": \"<etykieta>\"}}"
  - system: "You are a careful annotator of Polish text. Reply with a single JSON object and nothing else."
    user: "{instruction}\n\nAllowed labels:\n{glossary}\n\n{fields}\n\nReturn JSON: {{\"label\": \"<label>\"}}"
  - system: "Anotuj tekst po polsku. Format odpowiedzi: JSON."
    user: "{fields}\n\n{instruction}\nEtykiety: {label_list}\nOdpowiedź jako JSON z kluczem \"label\"."
  - system: "Jesteś ekspertem lingwistyki komputerowej. Zwracasz tylko JSON, bez komentarza."
    user: "Zadanie: {instruction}\n\nGlosariusz etykiet:\n{glossary}\n\nDane wejściowe:\n{fields}\n\nWynik (JSON):"
  - system: "Klasyfikator tekstu polskiego. Wyjście: JSON."
    user: "{instruction}\n\n{fields}\n\nWybierz dokładnie jedną etykietę z: {label_list}. Odpowiedz obiektem JSON {{\"label\": ...}}."

tasks:
  polemo2:
    instruction: "Określ wydźwięk recenzji napisanej po polsku."
    fields: [text]
    glossary:
      __label__meta_plus_m: "wydźwięk pozytywny"
      __label__meta_minus_m: "wydźwięk negatywny"
      __label__meta_zero: "wydźwięk neutralny"
      __label__meta_amb: "wydźwięk niejednoznaczny, mieszany"
  polemo2_in:
    instruction: "Określ wydźwięk recenzji napisanej po polsku (domena zgodna z treningiem)."
    fields: [sentence]
    glossary:
      __label__meta_plus_m: "wydźwięk pozytywny"
      __label__meta_minus_m: "wydźwięk negatywny"
      __label__meta_zero: "wydźwięk neutralny"
      __label__meta_amb: "wydźwięk niejednoznaczny, mieszany"
  polemo2_out:
    instruction: "Określ wydźwięk recenzji napisanej po polsku (domena spoza treningu)."
    fields: [sentence]
    glossary:
      __label__meta_plus_m: "wydźwięk pozytywny"
      __label__meta_minus_m: "wydźwięk negatywny"
      __label__meta_zero: "wydźwięk neutralny"
      __label__meta_amb: "wydźwięk niejednoznaczny, mieszany"
  abusive_clauses:
    instruction: "Oceń, czy postanowienie umowne jest klauzulą abuzywną w rozumieniu polskiego prawa konsumenckiego."
    fields: [text]
    glossary:
      BEZPIECZNE_POSTANOWIENIE_UMOWNE: "postanowienie dozwolone"
      KLAUZULA_ABUZYWNA: "klauzula niedozwolona"
  cdsc_e:
    instruction: "Określ relację logiczną między zdaniem A a zdaniem B."
    fields: [sentence_A, sentence_B]
    glossary:
      NEUTRAL: "brak wynikania i brak sprzeczności"
      ENTAILMENT: "zdanie B wynika ze zdania A"
      CONTRADICTION: "zdanie B jest sprzeczne ze zdaniem A"
  dyk:
    instruction: "Oceń, czy odpowiedź jest poprawną odpowiedzią na zadane pytanie."
    fields: [question, answer]
    glossary:
      "0": "odpowiedź niepoprawna"
      "1": "odpowiedź poprawna"
  psc:
    instruction: "Oceń, czy streszczenie jest parafrazą podanego fragmentu artykułu."
    fields: [extract_text, summary_text]
    glossary:
      "0": "nie jest parafrazą"
      "1": "jest parafrazą"
  dialogue_acts:
    instruction: "Określ akt mowy wypowiedzi w polskim dialogu."
    fields: [text]
    glossary:
      statement: "stwierdzenie"
      question: "pytanie"
      backchannel: "sygnał podtrzymania rozmowy"
      other: "inne"
```

`sequence_labeling.yaml` and `qa.yaml` are written in Task 9 and Task 10 respectively, in the
same shape.

- [ ] **Step 4: Write the prompt module**

Create `llm-eval/llm_eval/prompts/__init__.py`:

```python
"""Prompt templates and few-shot demonstration selection.

Encoder entries average over 5 training seeds. LLM entries average over 5
prompt variants, which is the closest analogue: it measures the same kind of
run-to-run instability. Variants differ in language, ordering and framing, not
in the information they carry.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from llm_eval import SEED
from llm_eval.tasks.base import Example, Task

N_VARIANTS = 5
PROMPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromptTemplate:
    kind: str
    variant: int
    system: str
    user: str


@dataclass(frozen=True)
class TaskPrompt:
    task: str
    instruction: str
    label_glossary: dict[str, str]
    fields: tuple[str, ...]


@lru_cache(maxsize=None)
def _load(kind: str) -> dict:
    path = PROMPT_DIR / f"{kind}.yaml"
    if not path.exists():
        raise KeyError(f"No prompt file for kind '{kind}' at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_template(kind: str, variant: int) -> PromptTemplate:
    variants = _load(kind)["variants"]
    if not 0 <= variant < len(variants):
        raise KeyError(f"Variant {variant} out of range for kind '{kind}' (0..{len(variants) - 1})")
    entry = variants[variant]
    return PromptTemplate(kind=kind, variant=variant, system=entry["system"], user=entry["user"])


def get_task_prompt(task_name: str) -> TaskPrompt:
    for kind in ("classification", "sequence_labeling", "qa"):
        try:
            tasks = _load(kind)["tasks"]
        except KeyError:
            continue
        if task_name in tasks:
            entry = tasks[task_name]
            return TaskPrompt(
                task=task_name,
                instruction=entry["instruction"],
                label_glossary={str(key): str(value) for key, value in entry.get("glossary", {}).items()},
                fields=tuple(entry["fields"]),
            )
    raise KeyError(f"No prompt defined for task '{task_name}'")


def select_demos(
    train: list[Example],
    task: Task,
    shots: int,
    variant: int,
    seed: int = SEED,
) -> list[Example]:
    """Stratified demonstration draw, deterministic per (variant, seed)."""
    if shots == 0 or not train:
        return []
    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in train:
        buckets[task.stratum(example)].append(example)
    rng = random.Random((seed, variant).__hash__())
    strata = sorted(buckets)
    demos: list[Example] = []
    index = 0
    while len(demos) < shots and any(buckets.values()):
        pool = buckets[strata[index % len(strata)]]
        if pool:
            demos.append(pool.pop(rng.randrange(len(pool))))
        index += 1
    return demos


def _render_fields(task_prompt: TaskPrompt, example: Example) -> str:
    return "\n".join(f"{field}: {example.inputs[field]}" for field in task_prompt.fields)


def _demo_answer(task: Task, demo: Example) -> str:
    import json

    return json.dumps({"label": demo.gold}, ensure_ascii=False)


def render(
    task: Task,
    example: Example,
    variant: int,
    shots: int,
    train: list[Example],
) -> list[dict[str, str]]:
    template = get_template(task.spec.kind, variant)
    task_prompt = get_task_prompt(task.spec.name)
    glossary = "\n".join(
        f"- {label}: {description}" for label, description in task_prompt.label_glossary.items()
    )
    label_list = ", ".join(task.spec.labels)

    def user_turn(target: Example) -> str:
        return template.user.format(
            instruction=task_prompt.instruction,
            glossary=glossary,
            label_list=label_list,
            fields=_render_fields(task_prompt, target),
        )

    messages = [{"role": "system", "content": template.system}]
    for demo in select_demos(train, task, shots, variant):
        messages.append({"role": "user", "content": user_turn(demo)})
        messages.append({"role": "assistant", "content": _demo_answer(task, demo)})
    messages.append({"role": "user", "content": user_turn(example)})
    return messages
```

Note: `TinyTask` in `tests/conftest.py` has no prompt YAML entry, so add one under a
`tests/fixtures/` override — instead, the simpler route the tests above assume: extend
`conftest.py`'s `TinyTask` handling by appending a `tiny` entry to
`llm_eval/prompts/classification.yaml`:

```yaml
  tiny:
    instruction: "Testowe zadanie klasyfikacji wydźwięku."
    fields: [text]
    glossary:
      positive: "pozytywny"
      negative: "negatywny"
      neutral: "neutralny"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_prompts.py -v`
Expected: PASS, 13 passed

- [ ] **Step 6: Commit**

```bash
git add llm-eval/llm_eval/prompts llm-eval/tests/test_prompts.py
git commit -m "feat(llm-eval): add prompt variants and stratified few-shot selection

Refs #5"
```

---

### Task 6: Generation backends

**Files:**
- Create: `llm-eval/llm_eval/backends/__init__.py`
- Create: `llm-eval/llm_eval/backends/dummy.py`
- Create: `llm-eval/llm_eval/backends/vllm_backend.py`
- Create: `llm-eval/tests/test_backends.py`

**Interfaces:**
- Consumes: `ModelSpec` from `llm_eval.registry`.
- Produces:
  - `GenerationRequest(id: str, messages: list[dict[str, str]], json_schema: dict | None, max_tokens: int)` frozen dataclass.
  - `GenerationResult(id: str, text: str, prompt_tokens: int, completion_tokens: int)` frozen dataclass.
  - `Backend` protocol with `generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]` and `name: str`.
  - `DummyBackend(responses: dict[str, str] | None = None, default: str = "{}")` — deterministic, CPU-only, used by every runner test.
  - `VLLMBackend(spec: ModelSpec, gpu_memory_utilization: float = 0.90)` — lazily imports vLLM, uses guided JSON decoding, `@pytest.mark.gpu` only.
  - `build_backend(spec: ModelSpec) -> Backend`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_backends.py`:

```python
import pytest

from llm_eval.backends import Backend, GenerationRequest, GenerationResult
from llm_eval.backends.dummy import DummyBackend


def request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        id=request_id,
        messages=[{"role": "user", "content": "x"}],
        json_schema=None,
        max_tokens=8,
    )


def test_dummy_backend_satisfies_the_protocol():
    assert isinstance(DummyBackend(), Backend)


def test_dummy_backend_preserves_request_order_and_ids():
    backend = DummyBackend(responses={"b": "second"}, default="fallback")
    results = backend.generate([request("a"), request("b"), request("c")])
    assert [result.id for result in results] == ["a", "b", "c"]
    assert results[1].text == "second"
    assert results[0].text == "fallback"


def test_dummy_backend_reports_token_counts():
    result = DummyBackend(default="abcd").generate([request("a")])[0]
    assert isinstance(result, GenerationResult)
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


def test_dummy_backend_is_deterministic():
    first = DummyBackend(default="x").generate([request("a")])
    second = DummyBackend(default="x").generate([request("a")])
    assert first == second


def test_dummy_backend_rejects_duplicate_request_ids():
    with pytest.raises(ValueError, match="duplicate"):
        DummyBackend().generate([request("a"), request("a")])


@pytest.mark.gpu
def test_vllm_backend_honours_guided_json():
    from llm_eval.backends.vllm_backend import VLLMBackend
    from llm_eval.registry import get_model

    backend = VLLMBackend(get_model("bielik-minitron-7b-v3-instruct"))
    schema = {
        "type": "object",
        "properties": {"label": {"enum": ["tak", "nie"]}},
        "required": ["label"],
        "additionalProperties": False,
    }
    results = backend.generate(
        [
            GenerationRequest(
                id="a",
                messages=[{"role": "user", "content": "Czy Warszawa leży w Polsce? Odpowiedz JSON."}],
                json_schema=schema,
                max_tokens=16,
            )
        ]
    )
    import json

    assert json.loads(results[0].text)["label"] in {"tak", "nie"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.backends'`

- [ ] **Step 3: Write the backend protocol and dummy**

Create `llm-eval/llm_eval/backends/__init__.py`:

```python
"""Generation backends.

The runner only ever sees this protocol, which is what lets the entire test
suite run on a machine with no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class GenerationRequest:
    id: str
    messages: list[dict[str, str]]
    json_schema: dict | None
    max_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    id: str
    text: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class Backend(Protocol):
    name: str

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...


def build_backend(spec) -> Backend:
    from llm_eval.backends.vllm_backend import VLLMBackend

    return VLLMBackend(spec)
```

Create `llm-eval/llm_eval/backends/dummy.py`:

```python
"""Deterministic CPU backend used by the test suite and by `--dry-run`."""

from __future__ import annotations

from typing import Sequence

from llm_eval.backends import GenerationRequest, GenerationResult


class DummyBackend:
    name = "dummy"

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}") -> None:
        self._responses = dict(responses or {})
        self._default = default

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        seen: set[str] = set()
        results: list[GenerationResult] = []
        for request in requests:
            if request.id in seen:
                raise ValueError(f"duplicate request id: {request.id}")
            seen.add(request.id)
            text = self._responses.get(request.id, self._default)
            prompt_chars = sum(len(message["content"]) for message in request.messages)
            results.append(
                GenerationResult(
                    id=request.id,
                    text=text,
                    prompt_tokens=max(1, prompt_chars // 4),
                    completion_tokens=max(1, len(text) // 4),
                )
            )
        return results
```

- [ ] **Step 4: Write the vLLM backend**

Create `llm-eval/llm_eval/backends/vllm_backend.py`:

```python
"""vLLM offline engine with guided JSON decoding.

Ampere cards have no native FP8, so quantization is AWQ/GPTQ int4 with Marlin
kernels or plain bf16. One engine owns one GPU; there is no tensor parallelism
anywhere in this benchmark.
"""

from __future__ import annotations

import json
from typing import Sequence

from llm_eval.backends import GenerationRequest, GenerationResult
from llm_eval.registry import ModelSpec

_QUANTIZATION = {"bf16": None, "awq-int4": "awq_marlin", "gptq-int4": "gptq_marlin"}


class VLLMBackend:
    def __init__(self, spec: ModelSpec, gpu_memory_utilization: float = 0.90) -> None:
        from vllm import LLM

        self.spec = spec
        self.name = f"vllm:{spec.id}"
        self._llm = LLM(
            model=spec.hf_id,
            revision=spec.revision,
            dtype="bfloat16",
            quantization=_QUANTIZATION[spec.quantization],
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            seed=441,
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        seen: set[str] = set()
        for request in requests:
            if request.id in seen:
                raise ValueError(f"duplicate request id: {request.id}")
            seen.add(request.id)

        # Greedy decoding: the benchmark measures the model, not sampling luck.
        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=request.max_tokens,
                guided_decoding=(
                    GuidedDecodingParams(json=request.json_schema)
                    if request.json_schema is not None
                    else None
                ),
            )
            for request in requests
        ]
        outputs = self._llm.chat(
            [list(request.messages) for request in requests],
            sampling_params=params,
            use_tqdm=False,
        )
        results = []
        for request, output in zip(requests, outputs):
            results.append(
                GenerationResult(
                    id=request.id,
                    text=output.outputs[0].text,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(output.outputs[0].token_ids),
                )
            )
        return results

    def __repr__(self) -> str:
        return f"VLLMBackend({self.spec.hf_id!r}, revision={self.spec.revision[:8]!r})"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_backends.py -v`
Expected: PASS, 5 passed, 1 deselected (the `gpu` test)

- [ ] **Step 6: Verify the GPU path once on real hardware**

Run: `cd llm-eval && CUDA_VISIBLE_DEVICES=0 uv run --extra gpu pytest tests/test_backends.py -m gpu -v`
Expected: PASS, 1 passed. If vLLM reports an out-of-memory error, lower
`gpu_memory_utilization` to `0.85` and re-run; do not switch to a smaller model to make it pass.

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/backends llm-eval/tests/test_backends.py
git commit -m "feat(llm-eval): add dummy and vllm generation backends

Refs #5"
```

---

### Task 7: Cost and energy instrumentation

**Files:**
- Create: `llm-eval/llm_eval/cost.py`
- Create: `llm-eval/tests/test_cost.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `CostReport(wall_seconds: float, gpu_seconds: float, energy_wh: float, peak_vram_mb: int, gpu_name: str, available: bool)` frozen dataclass with `.to_dict() -> dict`.
  - `EnergyMeter(device_index: int = 0)` context manager exposing `.report() -> CostReport` after exit.
  - `per_1k(report: CostReport, examples: int) -> dict[str, float]` returning `{"gpu_seconds_per_1k", "energy_wh_per_1k"}`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_cost.py`:

```python
import pytest

from llm_eval.cost import CostReport, EnergyMeter, per_1k


def test_meter_works_without_nvml_and_reports_unavailable():
    with EnergyMeter() as meter:
        sum(range(10_000))
    report = meter.report()
    assert isinstance(report, CostReport)
    assert report.wall_seconds >= 0.0
    if not report.available:
        assert report.energy_wh == 0.0
        assert report.peak_vram_mb == 0
        assert report.gpu_name == "cpu"


def test_report_serialises_every_field():
    report = CostReport(1.5, 1.2, 0.8, 4096, "NVIDIA GeForce RTX 3090", True)
    assert report.to_dict() == {
        "wall_seconds": 1.5,
        "gpu_seconds": 1.2,
        "energy_wh": 0.8,
        "peak_vram_mb": 4096,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "available": True,
    }


def test_per_1k_scales_linearly():
    report = CostReport(20.0, 10.0, 5.0, 4096, "gpu", True)
    scaled = per_1k(report, examples=500)
    assert scaled["gpu_seconds_per_1k"] == pytest.approx(20.0)
    assert scaled["energy_wh_per_1k"] == pytest.approx(10.0)


def test_per_1k_with_zero_examples_is_zero_not_a_crash():
    report = CostReport(20.0, 10.0, 5.0, 4096, "gpu", True)
    assert per_1k(report, examples=0) == {"gpu_seconds_per_1k": 0.0, "energy_wh_per_1k": 0.0}


def test_report_before_exit_raises():
    meter = EnergyMeter()
    with pytest.raises(RuntimeError, match="not finished"):
        meter.report()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_cost.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.cost'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/cost.py`:

```python
"""GPU cost accounting.

LEPISZCZE has reported training cost since the 2022 paper. The LLM track
extends that to inference cost, which is the axis where a 110M encoder can beat
a 27B decoder by three orders of magnitude.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CostReport:
    wall_seconds: float
    gpu_seconds: float
    energy_wh: float
    peak_vram_mb: int
    gpu_name: str
    available: bool

    def to_dict(self) -> dict:
        return {
            "wall_seconds": self.wall_seconds,
            "gpu_seconds": self.gpu_seconds,
            "energy_wh": self.energy_wh,
            "peak_vram_mb": self.peak_vram_mb,
            "gpu_name": self.gpu_name,
            "available": self.available,
        }


class EnergyMeter:
    """Context manager sampling NVML counters, degrading to wall-clock on CPU."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._report: CostReport | None = None
        self._handle = None
        self._pynvml = None
        self._start_wall = 0.0
        self._start_energy_mj = 0.0

    def __enter__(self) -> "EnergyMeter":
        self._start_wall = time.perf_counter()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._start_energy_mj = float(
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            )
        except Exception:  # noqa: BLE001 - absence of a GPU is a normal CI state
            self._pynvml = None
            self._handle = None
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        wall = time.perf_counter() - self._start_wall
        if self._pynvml is None or self._handle is None:
            self._report = CostReport(wall, 0.0, 0.0, 0, "cpu", False)
            return False
        pynvml = self._pynvml
        energy_mj = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        name = pynvml.nvmlDeviceGetName(self._handle)
        self._report = CostReport(
            wall_seconds=wall,
            gpu_seconds=wall,  # one engine owns the GPU for the whole run
            energy_wh=(energy_mj - self._start_energy_mj) / 3_600_000.0,
            peak_vram_mb=int(memory.used // (1024 * 1024)),
            gpu_name=name if isinstance(name, str) else name.decode("utf-8"),
            available=True,
        )
        pynvml.nvmlShutdown()
        return False

    def report(self) -> CostReport:
        if self._report is None:
            raise RuntimeError("EnergyMeter has not finished; use it as a context manager")
        return self._report


def per_1k(report: CostReport, examples: int) -> dict[str, float]:
    if examples <= 0:
        return {"gpu_seconds_per_1k": 0.0, "energy_wh_per_1k": 0.0}
    scale = 1000.0 / examples
    return {
        "gpu_seconds_per_1k": report.gpu_seconds * scale,
        "energy_wh_per_1k": report.energy_wh * scale,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_cost.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add llm-eval/llm_eval/cost.py llm-eval/tests/test_cost.py
git commit -m "feat(llm-eval): add nvml cost and energy instrumentation

Refs #5"
```

---

### Task 8: Resumable runner

**Files:**
- Create: `llm-eval/llm_eval/runner.py`
- Create: `llm-eval/tests/test_runner.py`

**Interfaces:**
- Consumes: `Example`/`Task`, `Backend`/`GenerationRequest`/`GenerationResult`, `render`, `EnergyMeter`/`per_1k`.
- Produces:
  - `RunConfig(model_id: str, task_name: str, variant: int, shots: int, results_root: Path, batch_size: int = 32)` frozen dataclass.
  - `prediction_path(config: RunConfig) -> Path` = `results_root / "preds" / model_id / task_name / f"{variant}.jsonl"`.
  - `meta_path(config: RunConfig) -> Path` = same stem with `.meta.json`.
  - `completed_ids(path: Path) -> set[str]` — tolerates a truncated final line from a killed process.
  - `RunSummary(total: int, generated: int, skipped: int, parse_failures: int)` frozen dataclass.
  - `run_task(task, examples, train, backend, config) -> RunSummary`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_runner.py`:

```python
import json

from llm_eval.backends.dummy import DummyBackend
from llm_eval.runner import RunConfig, completed_ids, meta_path, prediction_path, run_task


def config(tmp_path, variant=0, batch_size=4):
    return RunConfig(
        model_id="dummy-model",
        task_name="tiny",
        variant=variant,
        shots=0,
        results_root=tmp_path,
        batch_size=batch_size,
    )


def good(label="positive"):
    return DummyBackend(default=json.dumps({"label": label}))


def test_paths_follow_the_documented_layout(tmp_path):
    cfg = config(tmp_path, variant=3)
    assert prediction_path(cfg) == tmp_path / "preds" / "dummy-model" / "tiny" / "3.jsonl"
    assert meta_path(cfg) == tmp_path / "preds" / "dummy-model" / "tiny" / "3.meta.json"


def test_run_writes_one_line_per_example(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    summary = run_task(tiny_task, examples, [], good(), config(tmp_path))
    lines = prediction_path(config(tmp_path)).read_text(encoding="utf-8").splitlines()
    assert summary.total == 12
    assert summary.generated == 12
    assert summary.skipped == 0
    assert len(lines) == 12


def test_prediction_records_carry_everything_the_scorer_needs(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples, [], good(), config(tmp_path))
    record = json.loads(prediction_path(config(tmp_path)).read_text(encoding="utf-8").splitlines()[0])
    assert set(record) == {
        "example_id",
        "raw",
        "prediction",
        "ok",
        "error",
        "prompt_tokens",
        "completion_tokens",
    }
    assert record["prediction"] == "positive"
    assert record["ok"] is True


def test_second_run_regenerates_nothing(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples, [], good(), config(tmp_path))
    summary = run_task(tiny_task, examples, [], good(), config(tmp_path))
    assert summary.generated == 0
    assert summary.skipped == 12


def test_run_resumes_after_a_partial_file(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples[:5], [], good(), config(tmp_path))
    summary = run_task(tiny_task, examples, [], good(), config(tmp_path))
    assert summary.generated == 7
    assert summary.skipped == 5
    assert len(completed_ids(prediction_path(config(tmp_path)))) == 12


def test_completed_ids_ignores_a_truncated_trailing_line(tmp_path, tiny_task):
    cfg = config(tmp_path)
    path = prediction_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"example_id": "test-0"}\n{"example_id": "test-1"', encoding="utf-8")
    assert completed_ids(path) == {"test-0"}


def test_unparseable_output_is_recorded_and_counted(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    summary = run_task(tiny_task, examples, [], DummyBackend(default="nie wiem"), config(tmp_path))
    assert summary.parse_failures == 12
    record = json.loads(prediction_path(config(tmp_path)).read_text(encoding="utf-8").splitlines()[0])
    assert record["ok"] is False
    assert record["prediction"] is None
    assert record["raw"] == "nie wiem"


def test_meta_file_records_cost_and_configuration(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples, [], good(), config(tmp_path))
    meta = json.loads(meta_path(config(tmp_path)).read_text(encoding="utf-8"))
    assert meta["model_id"] == "dummy-model"
    assert meta["task"] == "tiny"
    assert meta["variant"] == 0
    assert meta["shots"] == 0
    assert meta["examples"] == 12
    assert meta["backend"] == "dummy"
    assert "cost" in meta and "gpu_seconds_per_1k" in meta


def test_meta_cost_accumulates_across_resumed_runs(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples[:5], [], good(), config(tmp_path))
    first = json.loads(meta_path(config(tmp_path)).read_text(encoding="utf-8"))
    run_task(tiny_task, examples, [], good(), config(tmp_path))
    second = json.loads(meta_path(config(tmp_path)).read_text(encoding="utf-8"))
    assert second["examples"] == 12
    assert second["cost"]["wall_seconds"] >= first["cost"]["wall_seconds"]
    assert second["runs"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.runner'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/runner.py`:

```python
"""Resumable generation loop.

The GPUs backing this benchmark are shared and preemptible. Predictions are
therefore append-only, one JSON object per line, flushed and fsynced per batch:
a job killed mid-run loses at most one batch, and restarting it regenerates
nothing that already exists on disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from llm_eval.backends import Backend, GenerationRequest
from llm_eval.cost import EnergyMeter, per_1k
from llm_eval.prompts import render
from llm_eval.tasks.base import Example, Task


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    task_name: str
    variant: int
    shots: int
    results_root: Path
    batch_size: int = 32


@dataclass(frozen=True)
class RunSummary:
    total: int
    generated: int
    skipped: int
    parse_failures: int


def prediction_path(config: RunConfig) -> Path:
    return config.results_root / "preds" / config.model_id / config.task_name / f"{config.variant}.jsonl"


def meta_path(config: RunConfig) -> Path:
    return prediction_path(config).with_suffix(".meta.json")


def completed_ids(path: Path) -> set[str]:
    """Ids already on disk. A truncated final line from a killed job is dropped."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                done.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                break
    return done


def run_task(
    task: Task,
    examples: list[Example],
    train: list[Example],
    backend: Backend,
    config: RunConfig,
) -> RunSummary:
    path = prediction_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(path)
    pending = [example for example in examples if example.id not in done]

    parse_failures = 0
    with EnergyMeter() as meter, path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start : start + config.batch_size]
            requests = [
                GenerationRequest(
                    id=example.id,
                    messages=render(task, example, config.variant, config.shots, train),
                    json_schema=task.output_schema(),
                    max_tokens=task.spec.max_new_tokens,
                )
                for example in batch
            ]
            results = backend.generate(requests)
            for example, result in zip(batch, results):
                parsed = task.parse(result.text, example)
                parse_failures += 0 if parsed.ok else 1
                handle.write(
                    json.dumps(
                        {
                            "example_id": example.id,
                            "raw": result.text,
                            "prediction": parsed.prediction,
                            "ok": parsed.ok,
                            "error": parsed.error,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

    _write_meta(config, meter.report(), examples_done=len(done) + len(pending), backend=backend)
    return RunSummary(
        total=len(examples),
        generated=len(pending),
        skipped=len(done),
        parse_failures=parse_failures,
    )


def _write_meta(config: RunConfig, report, examples_done: int, backend: Backend) -> None:
    path = meta_path(config)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    accumulated = previous.get("cost", {})
    cost = report.to_dict()
    for key in ("wall_seconds", "gpu_seconds", "energy_wh"):
        cost[key] = float(accumulated.get(key, 0.0)) + cost[key]
    cost["peak_vram_mb"] = max(int(accumulated.get("peak_vram_mb", 0)), cost["peak_vram_mb"])
    payload = {
        "model_id": config.model_id,
        "task": config.task_name,
        "variant": config.variant,
        "shots": config.shots,
        "examples": examples_done,
        "backend": backend.name,
        "runs": int(previous.get("runs", 0)) + 1,
        "cost": cost,
        **per_1k(report, examples_done),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_runner.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Prove resumability against a real kill**

Run:

```bash
cd llm-eval && uv run python - <<'PY'
import json, signal, os
from pathlib import Path
from llm_eval.backends.dummy import DummyBackend
from llm_eval.runner import RunConfig, prediction_path, run_task
import sys
sys.path.insert(0, "tests")
from conftest import TinyTask

task = TinyTask()
root = Path("/tmp/llm-eval-resume-check")
cfg = RunConfig("dummy-model", "tiny", 0, 0, root, batch_size=4)
run_task(task, task.load("test")[:4], [], DummyBackend(default=json.dumps({"label": "positive"})), cfg)
# Simulate a hard kill mid-write: truncate the last line.
path = prediction_path(cfg)
text = path.read_text(encoding="utf-8")
path.write_text(text[: -len(text.splitlines()[-1]) - 1] + text.splitlines()[-1][:10], encoding="utf-8")
summary = run_task(task, task.load("test"), [], DummyBackend(default=json.dumps({"label": "positive"})), cfg)
print(summary)
assert summary.skipped == 3 and summary.generated == 9, summary
print("resume OK")
PY
```

Expected: prints `resume OK`

- [ ] **Step 6: Commit**

```bash
git add llm-eval/llm_eval/runner.py llm-eval/tests/test_runner.py
git commit -m "feat(llm-eval): add resumable append-only generation runner

Refs #5"
```

---

### Task 9: Sequence-labeling tasks and span alignment

**Files:**
- Create: `llm-eval/llm_eval/tasks/sequence_labeling.py`
- Create: `llm-eval/llm_eval/prompts/sequence_labeling.yaml`
- Create: `llm-eval/tests/test_sequence_labeling.py`

**Interfaces:**
- Consumes: `Example`, `ParseResult`, `TaskSpec`, `register`, `classification_metrics`.
- Produces:
  - `tokens_to_text(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]` — single-space join plus per-token `(start, end)` character offsets.
  - `spans_to_bio(spans: list[dict], offsets: list[tuple[int, int]]) -> list[str]` — a span covers a token when it overlaps its offset range.
  - `bio_to_spans(tags: list[str], tokens: list[str], offsets: list[tuple[int, int]]) -> list[dict]`.
  - `span_f1(gold_spans, pred_spans, strict: bool) -> float`.
  - `SequenceLabelingTask` class; `Example.gold` is `tuple[tuple[str, ...], tuple[str, ...]]` = `(tokens, bio_tags)`.
  - 5 registered tasks: `kpwr_ner`, `nkjp_pos`, `aspectemo`, `political_advertising`, `punctuation_restoration`.
  - `SequenceLabelingTask.score` returns the 10 historical metric names computed over flattened token tags, plus `_extra_metrics(golds, preds) -> dict` returning `span_f1_strict`, `span_f1_relaxed` — those go into the leaderboard `evaluation.extra` block, **never** into `metrics`, so no new leaderboard column appears next to the encoder rows.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_sequence_labeling.py`:

```python
import pytest

from llm_eval.tasks import get_task
from llm_eval.tasks.base import Example
from llm_eval.tasks.sequence_labeling import (
    bio_to_spans,
    span_f1,
    spans_to_bio,
    tokens_to_text,
)

TOKENS = ["Jan", "Kowalski", "mieszka", "w", "Krakowie", "."]
TAGS = ["B-nam_liv_person", "I-nam_liv_person", "O", "O", "B-nam_loc_gpe_city", "O"]


def test_tokens_to_text_offsets_point_at_the_tokens():
    text, offsets = tokens_to_text(TOKENS)
    assert text == "Jan Kowalski mieszka w Krakowie ."
    assert [text[start:end] for start, end in offsets] == TOKENS


def test_spans_to_bio_recovers_the_gold_tags():
    text, offsets = tokens_to_text(TOKENS)
    spans = [
        {"span": "Jan Kowalski", "start": 0, "label": "nam_liv_person"},
        {"span": "Krakowie", "start": text.index("Krakowie"), "label": "nam_loc_gpe_city"},
    ]
    assert spans_to_bio(spans, offsets) == TAGS


def test_bio_to_spans_is_the_inverse():
    text, offsets = tokens_to_text(TOKENS)
    spans = bio_to_spans(TAGS, TOKENS, offsets)
    assert spans == [
        {"span": "Jan Kowalski", "start": 0, "label": "nam_liv_person"},
        {"span": "Krakowie", "start": 23, "label": "nam_loc_gpe_city"},
    ]


def test_span_outside_the_text_is_ignored_not_crashing():
    _, offsets = tokens_to_text(TOKENS)
    tags = spans_to_bio([{"span": "Gdańsk", "start": 900, "label": "nam_loc_gpe_city"}], offsets)
    assert tags == ["O"] * len(TOKENS)


def test_overlapping_spans_resolve_to_the_first_one():
    text, offsets = tokens_to_text(TOKENS)
    spans = [
        {"span": "Jan Kowalski", "start": 0, "label": "nam_liv_person"},
        {"span": "Kowalski", "start": 4, "label": "nam_loc_gpe_city"},
    ]
    assert spans_to_bio(spans, offsets)[:2] == ["B-nam_liv_person", "I-nam_liv_person"]


def test_strict_span_f1_requires_exact_boundaries():
    gold = [{"span": "Jan Kowalski", "start": 0, "label": "PER"}]
    pred = [{"span": "Kowalski", "start": 4, "label": "PER"}]
    assert span_f1(gold, pred, strict=True) == pytest.approx(0.0)


def test_relaxed_span_f1_credits_partial_overlap():
    gold = [{"span": "Jan Kowalski", "start": 0, "label": "PER"}]
    pred = [{"span": "Kowalski", "start": 4, "label": "PER"}]
    assert span_f1(gold, pred, strict=False) == pytest.approx(1.0)


def test_relaxed_span_f1_still_requires_the_right_label():
    gold = [{"span": "Jan Kowalski", "start": 0, "label": "PER"}]
    pred = [{"span": "Kowalski", "start": 4, "label": "LOC"}]
    assert span_f1(gold, pred, strict=False) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "name,dataset,leaderboard_task",
    [
        ("kpwr_ner", "clarin-pl/kpwr-ner", "Named Entity Recognition"),
        ("nkjp_pos", "clarin-pl/nkjp-pos", "Part-of-speech Tagging"),
        ("aspectemo", "clarin-pl/aspectemo", "Aspect-based Sentiment Analysis"),
        ("political_advertising", "laugustyniak/political-advertising-pl", "Political Advertising Detection"),
        ("punctuation_restoration", "clarin-pl/2021-punctuation-restoration", "Punctuation Restoration"),
    ],
)
def test_registered_tasks_match_the_leaderboard_identifiers(name, dataset, leaderboard_task):
    spec = get_task(name).spec
    assert spec.dataset == dataset
    assert spec.leaderboard_task == leaderboard_task
    assert spec.kind == "sequence_labeling"


def test_parse_converts_generated_spans_into_bio():
    task = get_task("kpwr_ner")
    text, _ = tokens_to_text(TOKENS)
    example = Example(id="x", inputs={"text": text}, gold=(tuple(TOKENS), tuple(TAGS)))
    raw = '{"spans": [{"span": "Jan Kowalski", "start": 0, "label": "nam_liv_person"}]}'
    result = task.parse(raw, example)
    assert result.ok
    assert result.prediction[0] == "B-nam_liv_person"


def test_parse_failure_yields_all_O_and_not_none():
    task = get_task("kpwr_ner")
    text, _ = tokens_to_text(TOKENS)
    example = Example(id="x", inputs={"text": text}, gold=(tuple(TOKENS), tuple(TAGS)))
    result = task.parse("przepraszam, nie rozumiem", example)
    assert not result.ok
    assert result.prediction is None
    assert "json" in result.error.lower()


def test_score_uses_only_the_historical_metric_names():
    task = get_task("kpwr_ner")
    golds = [tuple(TAGS)]
    preds = [tuple(TAGS)]
    assert set(task.score(golds, preds)) == {
        "accuracy",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
    }


def test_unparsed_prediction_scores_as_all_O():
    task = get_task("kpwr_ner")
    metrics = task.score([tuple(TAGS)], [None])
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_stratum_groups_by_the_label_set_present():
    task = get_task("kpwr_ner")
    example = Example(id="x", inputs={"text": "t"}, gold=(tuple(TOKENS), tuple(TAGS)))
    assert task.stratum(example) == "nam_liv_person|nam_loc_gpe_city"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_sequence_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.tasks.sequence_labeling'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/tasks/sequence_labeling.py`:

```python
"""Token-level tasks.

Decoder LLMs cannot emit per-token BIO tags reliably: their tokenizer does not
agree with the gold tokenization. They emit character spans instead, which are
aligned back onto gold tokens here. Both strict and relaxed span F1 are
reported, and a failed parse is scored as an all-"O" prediction rather than
being dropped from the test set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm_eval.tasks import register
from llm_eval.tasks.base import KIND_SEQUENCE_LABELING, Example, ParseResult, TaskSpec
from llm_eval.tasks.classification import classification_metrics

OUTSIDE = "O"


def tokens_to_text(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join on single spaces and return each token's character offsets."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            cursor += 1
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return " ".join(tokens), offsets


def spans_to_bio(spans: list[dict], offsets: list[tuple[int, int]]) -> list[str]:
    tags = [OUTSIDE] * len(offsets)
    for span in spans:
        try:
            start = int(span["start"])
            label = str(span["label"])
            end = start + len(str(span["span"]))
        except (KeyError, TypeError, ValueError):
            continue
        covered = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and start < token_end
        ]
        # First span wins; later overlapping spans do not overwrite it.
        if not covered or any(tags[index] != OUTSIDE for index in covered):
            continue
        tags[covered[0]] = f"B-{label}"
        for index in covered[1:]:
            tags[index] = f"I-{label}"
    return tags


def bio_to_spans(
    tags: list[str],
    tokens: list[str],
    offsets: list[tuple[int, int]],
) -> list[dict]:
    spans: list[dict] = []
    current: list[int] = []
    label = ""
    for index, tag in list(enumerate(tags)) + [(len(tags), OUTSIDE)]:
        prefix, _, tag_label = tag.partition("-")
        if current and (prefix != "I" or tag_label != label):
            spans.append(
                {
                    "span": " ".join(tokens[position] for position in current),
                    "start": offsets[current[0]][0],
                    "label": label,
                }
            )
            current = []
        if prefix == "B":
            current = [index]
            label = tag_label
        elif prefix == "I" and current:
            current.append(index)
    return spans


def _key(span: dict) -> tuple[int, int, str]:
    start = int(span["start"])
    return start, start + len(str(span["span"])), str(span["label"])


def span_f1(gold_spans: list[dict], pred_spans: list[dict], strict: bool) -> float:
    gold = [_key(span) for span in gold_spans]
    pred = [_key(span) for span in pred_spans]
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    matched_gold: set[int] = set()
    hits = 0
    for pred_span in pred:
        for index, gold_span in enumerate(gold):
            if index in matched_gold or gold_span[2] != pred_span[2]:
                continue
            exact = gold_span[:2] == pred_span[:2]
            overlap = gold_span[0] < pred_span[1] and pred_span[0] < gold_span[1]
            if exact or (not strict and overlap):
                matched_gold.add(index)
                hits += 1
                break
    precision = hits / len(pred)
    recall = hits / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@dataclass
class SequenceLabelingTask:
    spec: TaskSpec
    token_column: str
    tag_column: str

    def load(self, split: str) -> list[Example]:
        from datasets import load_dataset

        rows = load_dataset(self.spec.dataset, split=split, revision=self.spec.dataset_revision)
        feature = rows.features[self.tag_column].feature
        names = getattr(feature, "names", None)
        examples: list[Example] = []
        for index, row in enumerate(rows):
            tokens = [str(token) for token in row[self.token_column]]
            tags = [str(names[tag]) if names else str(tag) for tag in row[self.tag_column]]
            text, _ = tokens_to_text(tokens)
            examples.append(
                Example(
                    id=f"{self.spec.name}-{split}-{index}",
                    inputs={"text": text},
                    gold=(tuple(tokens), tuple(tags)),
                )
            )
        return examples

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span": {"type": "string"},
                            "start": {"type": "integer", "minimum": 0},
                            "label": {"enum": list(self.spec.labels)},
                        },
                        "required": ["span", "start", "label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["spans"],
            "additionalProperties": False,
        }

    def parse(self, raw: str, example: Example) -> ParseResult:
        tokens, _ = example.gold
        _, offsets = tokens_to_text(list(tokens))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            return ParseResult(prediction=None, ok=False, error=f"invalid json: {error}")
        spans = payload.get("spans") if isinstance(payload, dict) else None
        if not isinstance(spans, list):
            return ParseResult(prediction=None, ok=False, error="missing 'spans' array")
        return ParseResult(prediction=tuple(spans_to_bio(spans, offsets)), ok=True)

    def score(self, golds: list, preds: list) -> dict[str, float]:
        flat_gold: list[str] = []
        flat_pred: list[str] = []
        for gold, pred in zip(golds, preds):
            tags = list(gold)
            flat_gold.extend(tags)
            flat_pred.extend(list(pred) if pred is not None else [OUTSIDE] * len(tags))
        return classification_metrics(flat_gold, flat_pred, labels=tuple(sorted(set(flat_gold))))

    def extra_metrics(self, golds: list, preds: list, tokens_list: list[tuple[str, ...]]) -> dict[str, float]:
        strict_scores: list[float] = []
        relaxed_scores: list[float] = []
        for gold, pred, tokens in zip(golds, preds, tokens_list):
            _, offsets = tokens_to_text(list(tokens))
            gold_spans = bio_to_spans(list(gold), list(tokens), offsets)
            pred_tags = list(pred) if pred is not None else [OUTSIDE] * len(tokens)
            pred_spans = bio_to_spans(pred_tags, list(tokens), offsets)
            strict_scores.append(span_f1(gold_spans, pred_spans, strict=True))
            relaxed_scores.append(span_f1(gold_spans, pred_spans, strict=False))
        count = max(1, len(strict_scores))
        return {
            "span_f1_strict": sum(strict_scores) / count,
            "span_f1_relaxed": sum(relaxed_scores) / count,
        }

    def stratum(self, example: Example) -> str:
        _, tags = example.gold
        labels = sorted({tag.partition("-")[2] for tag in tags if tag != OUTSIDE})
        return "|".join(labels) if labels else OUTSIDE
```

Register the five tasks at the bottom of the same module, filling `labels` from the real
dataset (Step 4 verifies them):

```python
def _sequence(name, leaderboard_task, dataset, labels, token_column, tag_column, max_new_tokens):
    return register(
        SequenceLabelingTask(
            spec=TaskSpec(
                name=name,
                leaderboard_task=leaderboard_task,
                dataset=dataset,
                dataset_revision="main",
                kind=KIND_SEQUENCE_LABELING,
                labels=labels,
                max_new_tokens=max_new_tokens,
            ),
            token_column=token_column,
            tag_column=tag_column,
        )
    )


kpwr_ner = _sequence(
    "kpwr_ner", "Named Entity Recognition", "clarin-pl/kpwr-ner",
    ("nam_liv_person", "nam_loc_gpe_city", "nam_org_institution", "nam_adj_country"),
    "tokens", "ner", 512,
)
nkjp_pos = _sequence(
    "nkjp_pos", "Part-of-speech Tagging", "clarin-pl/nkjp-pos",
    ("subst", "verb", "adj", "adv", "prep", "conj", "interp"),
    "tokens", "pos_tags", 1024,
)
aspectemo = _sequence(
    "aspectemo", "Aspect-based Sentiment Analysis", "clarin-pl/aspectemo",
    ("a_plus_m", "a_minus_m", "a_zero", "a_amb"),
    "tokens", "labels", 512,
)
political_advertising = _sequence(
    "political_advertising", "Political Advertising Detection",
    "laugustyniak/political-advertising-pl",
    ("DEFENSE_AND_SECURITY", "HEALHCARE", "SOCIETY", "WELFARE"),
    "tokens", "tags", 512,
)
punctuation_restoration = _sequence(
    "punctuation_restoration", "Punctuation Restoration",
    "clarin-pl/2021-punctuation-restoration",
    (".", ",", "?", "!", "-", ":", ";"),
    "tokens", "tags", 1024,
)
```

- [ ] **Step 4: Verify tag vocabularies and column names against the real datasets**

```bash
cd llm-eval && uv run python - <<'PY'
from datasets import load_dataset

for dataset in [
    "clarin-pl/kpwr-ner",
    "clarin-pl/nkjp-pos",
    "clarin-pl/aspectemo",
    "laugustyniak/political-advertising-pl",
    "clarin-pl/2021-punctuation-restoration",
]:
    rows = load_dataset(dataset, split="train")
    print(dataset, "columns:", rows.column_names)
    for column, feature in rows.features.items():
        names = getattr(getattr(feature, "feature", None), "names", None)
        if names:
            print("   ", column, "->", names)
PY
```

Set `token_column`, `tag_column` and the full `labels` tuple (the bare label, without the
`B-`/`I-` prefix, deduplicated and sorted) from this output. `nkjp_pos` in particular has a
large tagset — copy all of it.

- [ ] **Step 5: Write the sequence-labeling prompt file**

Create `llm-eval/llm_eval/prompts/sequence_labeling.yaml` in the same shape as
`classification.yaml`, with `{{"spans": [...]}}` in place of `{{"label": ...}}`:

```yaml
variants:
  - system: "Jesteś precyzyjnym anotatorem języka polskiego. Odpowiadasz wyłącznie obiektem JSON."
    user: "{instruction}\n\nDopuszczalne etykiety:\n{glossary}\n\nTekst:\n{fields}\n\nZwróć JSON: {{\"spans\": [{{\"span\": \"<fragment>\", \"start\": <offset znakowy>, \"label\": \"<etykieta>\"}}]}}. Offsety liczone od zera w podanym tekście."
  - system: "You are a careful annotator of Polish text. Reply with a single JSON object and nothing else."
    user: "{instruction}\n\nAllowed labels:\n{glossary}\n\nText:\n{fields}\n\nReturn JSON: {{\"spans\": [{{\"span\": \"<surface form>\", \"start\": <char offset>, \"label\": \"<label>\"}}]}}."
  - system: "Anotuj tekst po polsku. Format odpowiedzi: JSON."
    user: "Tekst:\n{fields}\n\n{instruction}\nEtykiety: {label_list}\nZwróć listę fragmentów jako JSON z kluczem \"spans\"."
  - system: "Jesteś ekspertem lingwistyki komputerowej. Zwracasz tylko JSON, bez komentarza."
    user: "Zadanie: {instruction}\n\nGlosariusz etykiet:\n{glossary}\n\nTekst wejściowy:\n{fields}\n\nWynik (JSON, klucz \"spans\"):"
  - system: "Ekstraktor fragmentów z tekstu polskiego. Wyjście: JSON."
    user: "{instruction}\n\n{fields}\n\nZnajdź wszystkie fragmenty o etykietach: {label_list}. Pomiń fragmenty bez etykiety. Odpowiedz obiektem JSON {{\"spans\": [...]}}."

tasks:
  kpwr_ner:
    instruction: "Znajdź w tekście wszystkie nazwy własne i przypisz im kategorie."
    fields: [text]
    glossary: {}
  nkjp_pos:
    instruction: "Przypisz każdemu słowu tekstu jego część mowy zgodnie z tagsetem NKJP."
    fields: [text]
    glossary: {}
  aspectemo:
    instruction: "Znajdź w recenzji fragmenty wyrażające wydźwięk wobec aspektów produktu."
    fields: [text]
    glossary: {}
  political_advertising:
    instruction: "Znajdź w wypowiedzi fragmenty dotyczące obszarów przekazu politycznego."
    fields: [text]
    glossary: {}
  punctuation_restoration:
    instruction: "Wskaż miejsca, w których w tekście pozbawionym interpunkcji powinien pojawić się znak interpunkcyjny."
    fields: [text]
    glossary: {}
```

Fill each `glossary` with the real label descriptions from the dataset cards after Step 4.
`test_label_glossary_covers_every_label` in `tests/test_prompts.py` only applies to
classification tasks, so an empty glossary is legal but a filled one measurably helps the
smaller models — do not leave them empty in the final run.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_sequence_labeling.py tests/test_prompts.py -v`
Expected: PASS, all green

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/tasks/sequence_labeling.py llm-eval/llm_eval/prompts/sequence_labeling.yaml llm-eval/tests/test_sequence_labeling.py
git commit -m "feat(llm-eval): add sequence labeling tasks with span alignment

Refs #5"
```

---

### Task 10: Extractive QA tasks

**Files:**
- Create: `llm-eval/llm_eval/tasks/qa.py`
- Create: `llm-eval/llm_eval/prompts/qa.yaml`
- Create: `llm-eval/tests/test_qa.py`

**Interfaces:**
- Consumes: `Example`, `ParseResult`, `TaskSpec`, `register`.
- Produces:
  - `normalize_answer(text: str) -> str` — lowercase, strip Polish punctuation, collapse whitespace.
  - `qa_metrics(golds: list[list[str]], preds: list[str | None]) -> dict[str, float]` returning `accuracy` (= exact match) and `f1_macro` (token-overlap F1), plus the remaining eight historical keys mirrored from those two so the leaderboard column set stays identical.
  - `ExtractiveQATask` class and 4 registered tasks: `qa_all`, `qa_kpwr`, `qa_nkjp`, `qa_wikipedia`.
  - `Example.gold` is `tuple[str, ...]` — the set of acceptable gold answers.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_qa.py`:

```python
import pytest

from llm_eval.tasks import get_task
from llm_eval.tasks.base import Example
from llm_eval.tasks.qa import normalize_answer, qa_metrics


def test_normalisation_strips_punctuation_and_case():
    assert normalize_answer("  Kraków, ") == "kraków"


def test_normalisation_collapses_internal_whitespace():
    assert normalize_answer("Jan\n Kowalski") == "jan kowalski"


def test_exact_match_accepts_any_gold_alternative():
    metrics = qa_metrics([["Kraków", "Krakowa"]], ["krakowa"])
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_token_f1_credits_partial_answers():
    metrics = qa_metrics([["Jan Kowalski"]], ["Kowalski"])
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["f1_macro"] == pytest.approx(2 / 3)


def test_unanswered_question_scores_zero_not_dropped():
    metrics = qa_metrics([["Kraków"], ["Gdańsk"]], ["Kraków", None])
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_metric_names_match_the_historical_leaderboard():
    assert set(qa_metrics([["a"]], ["a"])) == {
        "accuracy",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
    }


@pytest.mark.parametrize("name", ["qa_all", "qa_kpwr", "qa_nkjp", "qa_wikipedia"])
def test_registered_qa_tasks(name):
    spec = get_task(name).spec
    assert spec.leaderboard_task == "Question Answering"
    assert spec.kind == "qa"


def test_parse_extracts_the_answer_string():
    task = get_task("qa_all")
    example = Example(id="x", inputs={"question": "q", "context": "c"}, gold=("Kraków",))
    result = task.parse('{"answer": "Kraków"}', example)
    assert result.ok
    assert result.prediction == "Kraków"


def test_parse_rejects_a_missing_answer_key():
    task = get_task("qa_all")
    example = Example(id="x", inputs={"question": "q", "context": "c"}, gold=("Kraków",))
    result = task.parse('{"odpowiedz": "Kraków"}', example)
    assert not result.ok
    assert "answer" in result.error


def test_stratum_buckets_by_answer_length():
    task = get_task("qa_all")
    short = Example(id="a", inputs={"question": "q", "context": "c"}, gold=("Kraków",))
    long = Example(id="b", inputs={"question": "q", "context": "c"}, gold=("stolica dawnej Polski i miasto królów",))
    assert task.stratum(short) != task.stratum(long)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_qa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.tasks.qa'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/tasks/qa.py`:

```python
"""Extractive question answering.

The leaderboard's QA rows use the same ten metric keys as every other task, so
exact match is published as `accuracy` and token-overlap F1 as `f1_*`. Keeping
the key set identical is what stops the leaderboard UI from sprouting a column
that only LLM rows can fill.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass

from llm_eval.tasks import register
from llm_eval.tasks.base import KIND_QA, Example, ParseResult, TaskSpec

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}«»„”–—]")
_SPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    common = 0
    remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_metrics(golds: list[list[str]], preds: list[str | None]) -> dict[str, float]:
    exact_scores: list[float] = []
    f1_scores: list[float] = []
    for gold_options, pred in zip(golds, preds):
        if pred is None:
            exact_scores.append(0.0)
            f1_scores.append(0.0)
            continue
        exact_scores.append(
            float(any(normalize_answer(gold) == normalize_answer(pred) for gold in gold_options))
        )
        f1_scores.append(max(_token_f1(gold, pred) for gold in gold_options))
    count = max(1, len(exact_scores))
    exact = sum(exact_scores) / count
    overlap = sum(f1_scores) / count
    return {
        "accuracy": exact,
        "f1_macro": overlap,
        "f1_micro": overlap,
        "f1_weighted": overlap,
        "precision_macro": overlap,
        "precision_micro": overlap,
        "precision_weighted": overlap,
        "recall_macro": overlap,
        "recall_micro": overlap,
        "recall_weighted": overlap,
    }


@dataclass
class ExtractiveQATask:
    spec: TaskSpec
    hf_dataset: str
    hf_config: str | None

    def load(self, split: str) -> list[Example]:
        from datasets import load_dataset

        rows = load_dataset(
            self.hf_dataset,
            self.hf_config,
            split=split,
            revision=self.spec.dataset_revision,
        )
        examples: list[Example] = []
        for index, row in enumerate(rows):
            answers = row["answers"]["text"] if isinstance(row["answers"], dict) else row["answers"]
            examples.append(
                Example(
                    id=f"{self.spec.name}-{split}-{index}",
                    inputs={"question": str(row["question"]), "context": str(row["context"])},
                    gold=tuple(str(answer) for answer in answers),
                )
            )
        return examples

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

    def parse(self, raw: str, example: Example) -> ParseResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            return ParseResult(prediction=None, ok=False, error=f"invalid json: {error}")
        if not isinstance(payload, dict) or "answer" not in payload:
            return ParseResult(prediction=None, ok=False, error="missing 'answer' key")
        return ParseResult(prediction=str(payload["answer"]), ok=True)

    def score(self, golds: list, preds: list) -> dict[str, float]:
        return qa_metrics([list(gold) for gold in golds], preds)

    def stratum(self, example: Example) -> str:
        longest = max((len(answer.split()) for answer in example.gold), default=0)
        return "short" if longest <= 3 else "long"


def _qa(name: str, hf_dataset: str, hf_config: str | None) -> ExtractiveQATask:
    return register(
        ExtractiveQATask(
            spec=TaskSpec(
                name=name,
                leaderboard_task="Question Answering",
                dataset=name,
                dataset_revision="main",
                kind=KIND_QA,
                labels=(),
                max_new_tokens=64,
            ),
            hf_dataset=hf_dataset,
            hf_config=hf_config,
        )
    )


# `dataset` deliberately keeps the leaderboard's synthetic identifiers
# (qa_all, qa_kpwr, ...) so new rows land in the existing dataset selector.
qa_all = _qa("qa_all", "clarin-pl/poquad", None)
qa_kpwr = _qa("qa_kpwr", "clarin-pl/poquad", "kpwr")
qa_nkjp = _qa("qa_nkjp", "clarin-pl/poquad", "nkjp")
qa_wikipedia = _qa("qa_wikipedia", "clarin-pl/poquad", "wikipedia")
```

- [ ] **Step 4: Verify the QA source dataset**

The four `qa_*` leaderboard rows came from `clarin-knext/plt5-large-poquad__inference` in the
`embeddings` repo. Confirm which Hub dataset and which config each row used:

```bash
git clone --depth 1 https://github.com/CLARIN-PL/embeddings.git /tmp/embeddings
grep -rn "qa_kpwr\|qa_nkjp\|qa_wikipedia\|poquad" /tmp/embeddings --include='*.py' --include='*.json' --include='*.yaml' | head -40
```

Set `hf_dataset` and `hf_config` to what that shows. If the four subsets turn out to be
filters over one split rather than Hub configs, replace `hf_config` with a `filter_key`
field and filter in `load`. **Do not guess** — a wrong QA split silently produces
unpublishable numbers.

- [ ] **Step 5: Write `llm-eval/llm_eval/prompts/qa.yaml`**

```yaml
variants:
  - system: "Odpowiadasz na pytania na podstawie podanego kontekstu. Odpowiedź to dosłowny fragment kontekstu. Zwracasz wyłącznie JSON."
    user: "Kontekst:\n{fields}\n\nZwróć JSON: {{\"answer\": \"<dosłowny fragment kontekstu>\"}}"
  - system: "Answer questions strictly from the given Polish context, quoting it verbatim. Reply with JSON only."
    user: "{fields}\n\nReturn JSON: {{\"answer\": \"<verbatim span from the context>\"}}"
  - system: "System ekstrakcji odpowiedzi. Wyjście: JSON."
    user: "{fields}\n\nWytnij z kontekstu najkrótszy fragment odpowiadający na pytanie. Odpowiedz JSON z kluczem \"answer\"."
  - system: "Jesteś systemem QA dla języka polskiego. Nie parafrazujesz, cytujesz."
    user: "{instruction}\n\n{fields}\n\nWynik (JSON):"
  - system: "Ekstraktor odpowiedzi. Tylko JSON, bez wyjaśnień."
    user: "{fields}\n\nOdpowiedz najkrótszym możliwym cytatem z kontekstu, jako {{\"answer\": ...}}."

tasks:
  qa_all:
    instruction: "Odpowiedz na pytanie, cytując dosłowny fragment kontekstu."
    fields: [question, context]
    glossary: {}
  qa_kpwr:
    instruction: "Odpowiedz na pytanie, cytując dosłowny fragment kontekstu."
    fields: [question, context]
    glossary: {}
  qa_nkjp:
    instruction: "Odpowiedz na pytanie, cytując dosłowny fragment kontekstu."
    fields: [question, context]
    glossary: {}
  qa_wikipedia:
    instruction: "Odpowiedz na pytanie, cytując dosłowny fragment kontekstu."
    fields: [question, context]
    glossary: {}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_qa.py -v`
Expected: PASS, 12 passed

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/tasks/qa.py llm-eval/llm_eval/prompts/qa.yaml llm-eval/tests/test_qa.py
git commit -m "feat(llm-eval): add extractive qa tasks

Refs #5"
```

---

### Task 11: Scoring and aggregation into leaderboard-ready results

**Files:**
- Create: `llm-eval/llm_eval/score.py`
- Create: `llm-eval/llm_eval/aggregate.py`
- Create: `llm-eval/tests/test_aggregate.py`

**Interfaces:**
- Consumes: `Task`, `Example`, `ModelSpec`, `prediction_path`/`meta_path`/`RunConfig`.
- Produces:
  - `VariantScore(variant: int, metrics: dict[str, float], parse_failure_rate: float, extra: dict[str, float], cost: dict, examples: int)` frozen dataclass.
  - `score_variant(task: Task, examples: list[Example], config: RunConfig) -> VariantScore`.
  - `aggregate(spec: ModelSpec, task: Task, scores: list[VariantScore], protocol: str, shots: int, subset: str) -> dict` — the result payload.
  - `result_path(root: Path, model_id: str, task_name: str) -> Path` = `root / "llm" / f"{model_id}__{task_name}.json"`.
  - `write_result(payload: dict, root: Path) -> Path`.
- The result payload mirrors the historical `embeddings` result files field for field, so `scripts/build_leaderboard.py` can read both with one code path: `submission_name`, `leaderboard_task_name`, `dataset_name`, `dataset_version`, `embedding_name`, `averaged_over`, `metrics_avg`, `metrics_std`, plus a new `evaluation` object.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_aggregate.py`:

```python
import json

import pytest

from llm_eval.aggregate import aggregate, result_path, write_result
from llm_eval.backends.dummy import DummyBackend
from llm_eval.registry import get_model
from llm_eval.runner import RunConfig, run_task
from llm_eval.score import VariantScore, score_variant


def config(tmp_path, variant):
    return RunConfig("dummy-model", "tiny", variant, 0, tmp_path, batch_size=4)


def test_score_variant_reads_predictions_from_disk(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples, [], DummyBackend(default=json.dumps({"label": "positive"})), config(tmp_path, 0))
    score = score_variant(tiny_task, examples, config(tmp_path, 0))
    assert score.variant == 0
    assert score.examples == 12
    assert score.metrics["accuracy"] == pytest.approx(4 / 12)
    assert score.parse_failure_rate == pytest.approx(0.0)


def test_score_variant_reports_parse_failures(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples, [], DummyBackend(default="nope"), config(tmp_path, 1))
    score = score_variant(tiny_task, examples, config(tmp_path, 1))
    assert score.parse_failure_rate == pytest.approx(1.0)
    assert score.metrics["accuracy"] == pytest.approx(0.0)


def test_score_variant_fails_loudly_on_an_incomplete_run(tmp_path, tiny_task):
    examples = tiny_task.load("test")
    run_task(tiny_task, examples[:5], [], DummyBackend(default=json.dumps({"label": "positive"})), config(tmp_path, 2))
    with pytest.raises(ValueError, match="7 missing"):
        score_variant(tiny_task, examples, config(tmp_path, 2))


def _scores():
    return [
        VariantScore(0, {"accuracy": 0.80, "f1_macro": 0.70}, 0.00, {"span_f1_strict": 0.6}, {"energy_wh": 1.0}, 100),
        VariantScore(1, {"accuracy": 0.90, "f1_macro": 0.80}, 0.02, {"span_f1_strict": 0.7}, {"energy_wh": 3.0}, 100),
    ]


def test_aggregate_reports_mean_and_population_std(tiny_task):
    payload = aggregate(get_model("bielik-11b-v3-instruct"), tiny_task, _scores(), "zero-shot", 0, "full")
    assert payload["metrics_avg"]["accuracy"] == pytest.approx(0.85)
    assert payload["metrics_std"]["accuracy"] == pytest.approx(0.05)
    assert payload["averaged_over"] == 2


def test_aggregate_uses_the_historical_field_names(tiny_task):
    payload = aggregate(get_model("bielik-11b-v3-instruct"), tiny_task, _scores(), "zero-shot", 0, "full")
    assert payload["submission_name"] == "tiny_speakleash__Bielik-11B-v3.0-Instruct_zero-shot"
    assert payload["leaderboard_task_name"] == "Sentiment Analysis"
    assert payload["dataset_name"] == "test/tiny"
    assert payload["embedding_name"] == "speakleash/Bielik-11B-v3.0-Instruct"


def test_aggregate_records_the_evaluation_block(tiny_task):
    payload = aggregate(get_model("bielik-11b-v3-instruct"), tiny_task, _scores(), "5-shot", 5, "capped-2000")
    evaluation = payload["evaluation"]
    assert evaluation["protocol"] == "5-shot"
    assert evaluation["shots"] == 5
    assert evaluation["quantization"] == "awq-int4"
    assert evaluation["revision"] == get_model("bielik-11b-v3-instruct").revision
    assert evaluation["prompt_variants"] == 2
    assert evaluation["test_subset"] == "capped-2000"
    assert evaluation["parse_failure_rate"] == pytest.approx(0.01)
    assert evaluation["extra"]["span_f1_strict"] == pytest.approx(0.65)


def test_aggregate_rejects_an_empty_score_list(tiny_task):
    with pytest.raises(ValueError, match="no variant scores"):
        aggregate(get_model("bielik-11b-v3-instruct"), tiny_task, [], "zero-shot", 0, "full")


def test_write_result_round_trips(tmp_path, tiny_task):
    payload = aggregate(get_model("bielik-11b-v3-instruct"), tiny_task, _scores(), "zero-shot", 0, "full")
    path = write_result(payload, tmp_path)
    assert path == result_path(tmp_path, "bielik-11b-v3-instruct", "tiny")
    assert json.loads(path.read_text(encoding="utf-8")) == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.aggregate'`

- [ ] **Step 3: Write `llm-eval/llm_eval/score.py`**

```python
"""Turn a variant's prediction file into metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm_eval.runner import RunConfig, meta_path, prediction_path
from llm_eval.tasks.base import Example, Task


@dataclass(frozen=True)
class VariantScore:
    variant: int
    metrics: dict[str, float]
    parse_failure_rate: float
    extra: dict[str, float]
    cost: dict
    examples: int


def score_variant(task: Task, examples: list[Example], config: RunConfig) -> VariantScore:
    path = prediction_path(config)
    records: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.endswith("\n"):
                    record = json.loads(line)
                    records[record["example_id"]] = record

    missing = [example.id for example in examples if example.id not in records]
    if missing:
        raise ValueError(
            f"{len(missing)} missing predictions for {config.model_id}/{config.task_name}"
            f" variant {config.variant}, e.g. {missing[0]}"
        )

    golds = [example.gold for example in examples]
    preds = [records[example.id]["prediction"] for example in examples]
    # JSON round-trips tuples into lists; sequence-labeling golds are tuples.
    preds = [tuple(pred) if isinstance(pred, list) else pred for pred in preds]
    failures = sum(1 for example in examples if not records[example.id]["ok"])

    extra: dict[str, float] = {}
    if hasattr(task, "extra_metrics"):
        tokens_list = [example.gold[0] for example in examples]
        extra = task.extra_metrics(
            [example.gold[1] for example in examples], preds, tokens_list
        )
        golds = [example.gold[1] for example in examples]

    meta = json.loads(meta_path(config).read_text(encoding="utf-8")) if meta_path(config).exists() else {}
    return VariantScore(
        variant=config.variant,
        metrics=task.score(golds, preds),
        parse_failure_rate=failures / max(1, len(examples)),
        extra=extra,
        cost=meta.get("cost", {}),
        examples=len(examples),
    )
```

- [ ] **Step 4: Write `llm-eval/llm_eval/aggregate.py`**

```python
"""Aggregate prompt variants into one leaderboard-ready result file.

The payload deliberately reuses the historical `embeddings` field names, so
`scripts/build_leaderboard.py` reads 2022 encoder results and 2026 LLM results
with a single code path.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from llm_eval.registry import ModelSpec
from llm_eval.score import VariantScore
from llm_eval.tasks.base import Task


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def aggregate(
    spec: ModelSpec,
    task: Task,
    scores: list[VariantScore],
    protocol: str,
    shots: int,
    subset: str,
) -> dict:
    if not scores:
        raise ValueError(f"no variant scores for {spec.id}/{task.spec.name}")

    metric_names = sorted(scores[0].metrics)
    metrics_avg: dict[str, float] = {}
    metrics_std: dict[str, float] = {}
    for name in metric_names:
        mean, std = _mean_std([score.metrics[name] for score in scores])
        metrics_avg[name] = mean
        metrics_std[name] = std

    extra: dict[str, float] = {}
    for name in sorted(scores[0].extra):
        extra[name] = statistics.fmean([score.extra[name] for score in scores])

    total_examples = sum(score.examples for score in scores)
    energy = sum(float(score.cost.get("energy_wh", 0.0)) for score in scores)
    gpu_seconds = sum(float(score.cost.get("gpu_seconds", 0.0)) for score in scores)
    peak_vram = max((int(score.cost.get("peak_vram_mb", 0)) for score in scores), default=0)
    scale = 1000.0 / total_examples if total_examples else 0.0

    return {
        "submission_name": f"{task.spec.name}_{spec.hf_id.replace('/', '__')}_{protocol}",
        "leaderboard_task_name": task.spec.leaderboard_task,
        "dataset_name": task.spec.dataset,
        "dataset_version": task.spec.dataset_revision,
        "embedding_name": spec.hf_id,
        "averaged_over": len(scores),
        "metrics_avg": metrics_avg,
        "metrics_std": metrics_std,
        "evaluation": {
            "protocol": protocol,
            "shots": shots,
            "quantization": spec.quantization,
            "revision": spec.revision,
            "prompt_variants": len(scores),
            "test_subset": subset,
            "parse_failure_rate": statistics.fmean([score.parse_failure_rate for score in scores]),
            "gpu_seconds_per_1k": gpu_seconds * scale,
            "energy_wh_per_1k": energy * scale,
            "peak_vram_mb": peak_vram,
            "extra": extra,
        },
    }


def result_path(root: Path, model_id: str, task_name: str) -> Path:
    return root / "llm" / f"{model_id}__{task_name}.json"


def write_result(payload: dict, root: Path) -> Path:
    model_id = payload["evaluation"].get("model_id") or payload["embedding_name"].split("/")[-1]
    raise NotImplementedError  # replaced in Step 5
```

- [ ] **Step 5: Finish `write_result` so it does not guess the model id**

`write_result` must not re-derive `model_id` from `embedding_name` — two registry entries can
share an `hf_id` prefix. Add `model_id` to the payload in `aggregate` and read it back:

In `aggregate`, add to the returned dict, inside `evaluation`:

```python
            "model_id": spec.id,
```

Then replace `write_result` with:

```python
def write_result(payload: dict, root: Path) -> Path:
    path = result_path(root, payload["evaluation"]["model_id"], payload["submission_name"].split("_")[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
```

Deriving the task name by splitting `submission_name` is fragile for task names containing
`_`. Pass it explicitly instead — change the signature to
`write_result(payload: dict, root: Path, task_name: str) -> Path` and update the test's
last case to `write_result(payload, tmp_path, "tiny")`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_aggregate.py -v`
Expected: PASS, 9 passed

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/score.py llm-eval/llm_eval/aggregate.py llm-eval/tests/test_aggregate.py
git commit -m "feat(llm-eval): aggregate prompt variants into leaderboard results

Refs #5"
```

---

### Task 12: Leaderboard schema v3 and merge path

**Files:**
- Modify: `scripts/build_leaderboard.py` (SCHEMA_VERSION, `build_snapshot`, `validate_snapshot`, `parse_args`, `main`)
- Modify: `docs/assets/leaderboard.js:159-256` (ranked rows and table rendering)
- Modify: `docs/data/leaderboard.json` (regenerated)
- Create: `tests/test_build_leaderboard.py` (repo root, run with the system `python3`)

**Interfaces:**
- Consumes: result JSON files produced by `llm_eval.aggregate.write_result`.
- Produces:
  - `SCHEMA_VERSION = 3`.
  - `build_snapshot(source_dir: Path | None, source_commit: str, llm_results_dir: Path | None = None, base: Path | None = None, llm_commit: str | None = None) -> dict`.
  - Snapshot top level gains `sources: list[dict]`, each `{"id", "repository", "commit", "results_path", "license"}`; the singular `source` key is removed.
  - Each entry gains `"source": "embeddings" | "llm-eval"` and `"evaluation": dict | None`.
  - `--llm-results-dir`, `--base` and `--llm-commit` CLI flags.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_leaderboard.py` at the repo root:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_leaderboard as bl

SHA = "a" * 40
LLM_SHA = "b" * 40


def llm_result():
    return {
        "submission_name": "polemo2_speakleash__Bielik-11B-v3.0-Instruct_zero-shot",
        "leaderboard_task_name": "Sentiment Analysis",
        "dataset_name": "clarin-pl/polemo2-official",
        "dataset_version": "1.0.0",
        "embedding_name": "speakleash/Bielik-11B-v3.0-Instruct",
        "averaged_over": 5,
        "metrics_avg": {"accuracy": 0.81, "f1_macro": 0.79},
        "metrics_std": {"accuracy": 0.02, "f1_macro": 0.03},
        "evaluation": {
            "protocol": "zero-shot",
            "shots": 0,
            "quantization": "awq-int4",
            "revision": SHA,
            "model_id": "bielik-11b-v3-instruct",
            "prompt_variants": 5,
            "test_subset": "full",
            "parse_failure_rate": 0.004,
            "gpu_seconds_per_1k": 812.0,
            "energy_wh_per_1k": 61.3,
            "peak_vram_mb": 21000,
            "extra": {},
        },
    }


@pytest.fixture
def base_snapshot(tmp_path):
    path = tmp_path / "leaderboard.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {
                    "repository": "https://github.com/CLARIN-PL/embeddings",
                    "commit": SHA,
                    "results_path": "webpage/data/results",
                    "license": "MIT",
                },
                "summary": {"entries": 1, "datasets": 1, "models": 1, "tasks": 1},
                "entries": [
                    {
                        "id": "polemo2_allegro__herbert-large-cased",
                        "task": "Sentiment Analysis",
                        "dataset": "clarin-pl/polemo2-official",
                        "dataset_label": "PolEmo 2.0",
                        "model": "allegro/herbert-large-cased",
                        "dataset_version": "0.0.0",
                        "averaged_over": 5,
                        "metrics": {"accuracy": {"mean": 0.88, "std": 0.01}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def llm_dir(tmp_path):
    directory = tmp_path / "llm"
    directory.mkdir()
    (directory / "bielik.json").write_text(json.dumps(llm_result()), encoding="utf-8")
    return directory


def test_migration_preserves_historical_entries(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    historical = [entry for entry in snapshot["entries"] if entry["source"] == "embeddings"]
    assert len(historical) == 1
    assert historical[0]["metrics"] == {"accuracy": {"mean": 0.88, "std": 0.01}}
    assert historical[0]["evaluation"] is None


def test_schema_version_is_three(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    assert snapshot["schema_version"] == 3
    assert "source" not in snapshot


def test_sources_list_names_both_provenances(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    ids = {source["id"] for source in snapshot["sources"]}
    assert ids == {"embeddings", "llm-eval"}


def test_llm_entry_carries_the_evaluation_block(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    entry = next(e for e in snapshot["entries"] if e["source"] == "llm-eval")
    assert entry["model"] == "speakleash/Bielik-11B-v3.0-Instruct"
    assert entry["dataset_label"] == "PolEmo 2.0"
    assert entry["evaluation"]["protocol"] == "zero-shot"
    assert entry["evaluation"]["energy_wh_per_1k"] == 61.3


def test_summary_counts_every_entry(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    assert snapshot["summary"]["entries"] == 2
    assert snapshot["summary"]["models"] == 2
    bl.validate_snapshot(snapshot)


def test_validator_rejects_a_missing_evaluation_key_on_llm_entries(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    for entry in snapshot["entries"]:
        if entry["source"] == "llm-eval":
            entry["evaluation"] = None
    with pytest.raises(ValueError, match="evaluation"):
        bl.validate_snapshot(snapshot)


def test_validator_rejects_an_unknown_source(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    snapshot["entries"][0]["source"] = "somewhere-else"
    with pytest.raises(ValueError, match="source"):
        bl.validate_snapshot(snapshot)


def test_duplicate_ids_across_sources_are_rejected(base_snapshot, llm_dir):
    snapshot = bl.build_snapshot(None, SHA, llm_results_dir=llm_dir, base=base_snapshot, llm_commit=LLM_SHA)
    snapshot["entries"][1]["id"] = snapshot["entries"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        bl.validate_snapshot(snapshot)


def test_checked_in_snapshot_still_validates():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_leaderboard.py", "--check", "docs/data/leaderboard.json"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_build_leaderboard.py -v`
Expected: FAIL — `build_snapshot() got an unexpected keyword argument 'llm_results_dir'`

- [ ] **Step 3: Extend `scripts/build_leaderboard.py`**

Change `SCHEMA_VERSION` to `3` and add above `build_snapshot`:

```python
SOURCE_EMBEDDINGS = "embeddings"
SOURCE_LLM = "llm-eval"
LLM_REPOSITORY = "https://github.com/CLARIN-PL/LEPISZCZE"
KNOWN_SOURCES = {SOURCE_EMBEDDINGS, SOURCE_LLM}
REQUIRED_EVALUATION_KEYS = {
    "protocol",
    "shots",
    "quantization",
    "revision",
    "model_id",
    "prompt_variants",
    "test_subset",
    "parse_failure_rate",
    "gpu_seconds_per_1k",
    "energy_wh_per_1k",
    "peak_vram_mb",
}


def _entry_from_result(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Shared conversion for both the historical files and the LLM results."""
    entry_id = str(raw["submission_name"])
    dataset = str(raw["dataset_name"])
    averages = raw.get("metrics_avg", {})
    deviations = raw.get("metrics_std", {})
    metrics: dict[str, dict[str, float | None]] = {}
    for metric_name in sorted(_metric_names(raw)):
        mean = _finite_number(averages[metric_name], f"{entry_id}.{metric_name}.mean")
        std_value = deviations.get(metric_name)
        std = (
            _finite_number(std_value, f"{entry_id}.{metric_name}.std")
            if std_value is not None
            else None
        )
        metrics[metric_name] = {"mean": mean, "std": std}
    if not metrics:
        raise ValueError(f"No summary metrics found for {entry_id}")
    return {
        "id": entry_id,
        "task": str(raw["leaderboard_task_name"]),
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "model": str(raw["embedding_name"]),
        "dataset_version": str(raw.get("dataset_version") or "unknown"),
        "averaged_over": raw.get("averaged_over"),
        "metrics": metrics,
        "source": source,
        "evaluation": raw.get("evaluation"),
    }


def _migrate_entries(base_path: Path) -> list[dict[str, Any]]:
    """Carry schema v2 entries forward untouched apart from the new fields."""
    snapshot = json.loads(base_path.read_text(encoding="utf-8"))
    entries = []
    for entry in snapshot["entries"]:
        migrated = dict(entry)
        migrated.setdefault("source", SOURCE_EMBEDDINGS)
        migrated.setdefault("evaluation", None)
        entries.append(migrated)
    return entries
```

Rewrite `build_snapshot` to accept the new inputs:

```python
def build_snapshot(
    source_dir: Path | None,
    source_commit: str,
    llm_results_dir: Path | None = None,
    base: Path | None = None,
    llm_commit: str | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if base is not None:
        entries.extend(_migrate_entries(base))
    if source_dir is not None:
        for source_file in sorted(source_dir.glob("*.json")):
            raw = json.loads(source_file.read_text(encoding="utf-8"))
            entries.append(_entry_from_result(raw, SOURCE_EMBEDDINGS))
    if llm_results_dir is not None:
        for source_file in sorted(llm_results_dir.glob("*.json")):
            raw = json.loads(source_file.read_text(encoding="utf-8"))
            entries.append(_entry_from_result(raw, SOURCE_LLM))
    if not entries:
        raise ValueError("No results found; pass --source-dir, --llm-results-dir or --base")

    seen_ids: set[str] = set()
    for entry in entries:
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate submission_name: {entry['id']}")
        seen_ids.add(entry["id"])

    entries.sort(
        key=lambda item: (
            item["task"] == "Question Answering",
            item["task"],
            item["dataset"],
            item["model"],
        )
    )
    sources = [
        {
            "id": SOURCE_EMBEDDINGS,
            "repository": SOURCE_REPOSITORY,
            "commit": source_commit,
            "results_path": "webpage/data/results",
            "license": "MIT",
        }
    ]
    if llm_results_dir is not None:
        if llm_commit is None:
            raise ValueError("--llm-commit is required with --llm-results-dir")
        sources.append(
            {
                "id": SOURCE_LLM,
                "repository": LLM_REPOSITORY,
                "commit": llm_commit,
                "results_path": "results/llm",
                "license": "MIT",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "summary": {
            "entries": len(entries),
            "datasets": len({entry["dataset"] for entry in entries}),
            "models": len({entry["model"] for entry in entries}),
            "tasks": len({entry["task"] for entry in entries}),
        },
        "entries": entries,
    }
```

In `validate_snapshot`, replace the `source` block with:

```python
    sources = snapshot.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Snapshot must declare at least one source")
    for source in sources:
        if source.get("id") not in KNOWN_SOURCES:
            raise ValueError(f"Unknown source id: {source.get('id')}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))):
            raise ValueError(f"Source {source.get('id')} commit must be a full lowercase Git SHA")
```

and inside the per-entry loop, after the existing metric checks:

```python
        if entry.get("source") not in KNOWN_SOURCES:
            raise ValueError(f"Unknown source for {entry['id']}: {entry.get('source')}")
        if entry["source"] == SOURCE_LLM:
            evaluation = entry.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError(f"Missing evaluation block for {entry['id']}")
            missing_keys = REQUIRED_EVALUATION_KEYS - evaluation.keys()
            if missing_keys:
                raise ValueError(
                    f"Incomplete evaluation for {entry['id']}: {', '.join(sorted(missing_keys))}"
                )
```

Extend `parse_args`: make the mutually exclusive group non-required, and add

```python
    parser.add_argument("--llm-results-dir", type=Path, help="Directory of llm-eval result JSON files")
    parser.add_argument("--base", type=Path, help="Existing snapshot to carry forward")
    parser.add_argument("--llm-commit", help="Full SHA of this repository for llm-eval provenance")
```

and update `main` to forward all four to `build_snapshot`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_build_leaderboard.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Migrate the checked-in snapshot**

Run:

```bash
python3 scripts/build_leaderboard.py \
  --base docs/data/leaderboard.json \
  --source-commit a434193a871c7d1bccdb5fd9106be5a950e38aad \
  --output docs/data/leaderboard.json
python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json
git diff --stat docs/data/leaderboard.json
```

Expected: `Validated 86 leaderboard results`. The diff must touch only `schema_version`,
`source` → `sources`, and add `"source": "embeddings"` / `"evaluation": null` per entry —
no metric value may change.

- [ ] **Step 6: Add the protocol and cost columns to the UI**

In `docs/assets/leaderboard.js`, inside the row builder (around line 205), after the model
cell is appended:

```javascript
      const setup = document.createElement("td");
      setup.className = "setup-cell";
      const evaluation = entry.evaluation;
      if (evaluation) {
        const badge = document.createElement("span");
        badge.className = "protocol-badge";
        badge.textContent = evaluation.protocol;
        badge.title = `${evaluation.quantization}, ${evaluation.prompt_variants} prompt variants, ` +
          `${(evaluation.parse_failure_rate * 100).toFixed(1)}% unparsed, ` +
          `${evaluation.test_subset} test split`;
        setup.append(badge);
      } else {
        setup.textContent = "fine-tuned";
      }
      row.append(setup);

      const cost = document.createElement("td");
      cost.className = "cost-cell";
      cost.textContent = evaluation ? `${Math.round(evaluation.energy_wh_per_1k)} Wh` : "—";
      cost.title = evaluation ? `${Math.round(evaluation.gpu_seconds_per_1k)} GPU-s per 1000 examples` : "";
      row.append(cost);
```

Add the two matching `<th>` cells in the header builder (around line 188), before the metric
columns: `Setup` and `Energy / 1k`. Add to `docs/assets/leaderboard.css`:

```css
.protocol-badge {
  display: inline-block;
  padding: 0.1rem 0.45rem;
  border-radius: 0.75rem;
  font-size: 0.75rem;
  background: var(--surface-muted, #eef1f5);
}

.cost-cell {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
```

Also extend the CSV export near line 258 with `row.protocol = entry.evaluation?.protocol ?? "fine-tuned"`
and `row.energy_wh_per_1k = entry.evaluation?.energy_wh_per_1k ?? null`.

- [ ] **Step 7: Verify the site still builds**

Run:

```bash
python3 scripts/check_site.py docs
node --check docs/assets/leaderboard.js
python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json
```

Expected: all three exit 0

- [ ] **Step 8: Commit**

```bash
git add scripts/build_leaderboard.py docs/data/leaderboard.json docs/assets/leaderboard.js docs/assets/leaderboard.css tests/test_build_leaderboard.py
git commit -m "feat(leaderboard): add schema v3 with llm provenance and cost columns

Refs #5"
```

---

### Task 13: CLI and shared-GPU job agenda

**Files:**
- Create: `llm-eval/llm_eval/cli.py`
- Create: `llm-eval/tests/test_cli.py`
- Create: `scripts/queue_agenda.sh`

**Interfaces:**
- Consumes: everything from Tasks 2–11.
- Produces:
  - `app = typer.Typer()` with commands `freeze-subsets`, `run`, `score`, `export`, `status`.
  - `llm-eval run --model <id> --task <name> --variant N --shots N [--dry-run] [--results-root PATH]`.
  - `llm-eval status [--results-root PATH]` printing one line per (model, task, variant) with `done/total`.
  - `agenda(models: list[str], tasks: list[str], variants: int, shots: int) -> list[tuple[str, str, int]]` — the job list, ordered cheapest-first so a short GPU window still produces complete cells.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_cli.py`:

```python
import json

from typer.testing import CliRunner

from llm_eval.cli import agenda, app

runner = CliRunner()


def test_agenda_is_ordered_cheapest_first():
    jobs = agenda(models=["a", "b"], tasks=["polemo2", "nkjp_pos"], variants=2, shots=0)
    assert len(jobs) == 8
    first_tasks = [task for _, task, _ in jobs[:4]]
    assert set(first_tasks) == {"polemo2"}


def test_agenda_covers_every_combination():
    jobs = set(agenda(models=["a"], tasks=["polemo2"], variants=5, shots=0))
    assert jobs == {("a", "polemo2", variant) for variant in range(5)}


def test_dry_run_uses_the_dummy_backend_and_touches_no_gpu(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--model", "bielik-11b-v3-instruct", "--task", "tiny", "--variant", "0",
         "--shots", "0", "--dry-run", "--results-root", str(tmp_path), "--limit", "4"],
    )
    assert result.exit_code == 0, result.output
    path = tmp_path / "preds" / "bielik-11b-v3-instruct" / "tiny" / "0.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_status_reports_progress(tmp_path):
    runner.invoke(
        app,
        ["run", "--model", "bielik-11b-v3-instruct", "--task", "tiny", "--variant", "0",
         "--shots", "0", "--dry-run", "--results-root", str(tmp_path), "--limit", "4"],
    )
    result = runner.invoke(app, ["status", "--results-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "bielik-11b-v3-instruct" in result.output
    assert "tiny" in result.output


def test_run_rejects_an_unknown_model(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--model", "nope", "--task", "tiny", "--variant", "0", "--shots", "0",
         "--dry-run", "--results-root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "Known models" in result.output


def test_freeze_subsets_writes_a_manifest(tmp_path, tiny_task):
    result = runner.invoke(
        app, ["freeze-subsets", "--task", "tiny", "--cap", "6", "--root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "data" / "subsets" / "tiny.json").read_text(encoding="utf-8"))
    assert len(manifest["example_ids"]) == 6
    assert manifest["seed"] == 441
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.cli'`

- [ ] **Step 3: Write the CLI**

Create `llm-eval/llm_eval/cli.py`:

```python
"""Command line entry point.

`agenda` orders jobs cheapest-first: on a shared GPU a window may close at any
moment, and a complete cheap cell is worth more than a half-finished expensive
one.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from llm_eval import SEED
from llm_eval.aggregate import aggregate, write_result
from llm_eval.backends import build_backend
from llm_eval.backends.dummy import DummyBackend
from llm_eval.registry import get_model
from llm_eval.runner import RunConfig, completed_ids, prediction_path, run_task
from llm_eval.score import score_variant
from llm_eval.subsets import apply_subset, freeze_subset, subset_path, SubsetManifest
from llm_eval.tasks import TASKS, get_task

app = typer.Typer(add_completion=False, help="LEPISZCZE LLM evaluation track")

# Rough per-example generation cost; drives agenda ordering only.
_COST_RANK = {"classification": 0, "qa": 1, "sequence_labeling": 2}


def agenda(models: list[str], tasks: list[str], variants: int, shots: int) -> list[tuple[str, str, int]]:
    jobs = [
        (model, task, variant)
        for task in tasks
        for model in models
        for variant in range(variants)
    ]
    return sorted(jobs, key=lambda job: (_COST_RANK.get(get_task(job[1]).spec.kind, 9), job[1], job[0], job[2]))


def _load_examples(task, root: Path, split: str = "test"):
    examples = task.load(split)
    manifest_path = subset_path(task.spec.name, root)
    if manifest_path.exists():
        manifest = SubsetManifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
        return apply_subset(examples, manifest)
    return examples


@app.command("freeze-subsets")
def freeze_subsets(
    task: str = typer.Option(..., help="Task name"),
    cap: int = typer.Option(2000),
    root: Path = typer.Option(Path(".")),
) -> None:
    task_obj = get_task(task)
    manifest = freeze_subset(task_obj, task_obj.load("test"), cap=cap, seed=SEED)
    path = subset_path(task, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
    typer.echo(f"{task}: froze {len(manifest.example_ids)} examples -> {path}")


@app.command()
def run(
    model: str = typer.Option(...),
    task: str = typer.Option(...),
    variant: int = typer.Option(0),
    shots: int = typer.Option(0),
    results_root: Path = typer.Option(Path("results")),
    batch_size: int = typer.Option(32),
    limit: int = typer.Option(0, help="Evaluate only the first N examples (smoke runs)"),
    dry_run: bool = typer.Option(False, help="Use the deterministic dummy backend"),
) -> None:
    spec = get_model(model)
    task_obj = get_task(task)
    examples = _load_examples(task_obj, results_root.parent)
    if limit:
        examples = examples[:limit]
    train = task_obj.load("train") if shots else []
    backend = DummyBackend(default=_dummy_output(task_obj)) if dry_run else build_backend(spec)
    config = RunConfig(spec.id, task, variant, shots, results_root, batch_size)
    summary = run_task(task_obj, examples, train, backend, config)
    typer.echo(
        f"{spec.id}/{task}/v{variant}: {summary.generated} generated, "
        f"{summary.skipped} cached, {summary.parse_failures} unparsed"
    )


def _dummy_output(task) -> str:
    if task.spec.kind == "classification":
        return json.dumps({"label": task.spec.labels[0]})
    if task.spec.kind == "sequence_labeling":
        return json.dumps({"spans": []})
    return json.dumps({"answer": ""})


@app.command()
def score(
    model: str = typer.Option(...),
    task: str = typer.Option(...),
    variants: int = typer.Option(5),
    shots: int = typer.Option(0),
    results_root: Path = typer.Option(Path("results")),
    subset: str = typer.Option("full"),
) -> None:
    spec = get_model(model)
    task_obj = get_task(task)
    examples = _load_examples(task_obj, results_root.parent)
    scores = [
        score_variant(task_obj, examples, RunConfig(spec.id, task, variant, shots, results_root))
        for variant in range(variants)
    ]
    protocol = "zero-shot" if shots == 0 else f"{shots}-shot"
    payload = aggregate(spec, task_obj, scores, protocol, shots, subset)
    path = write_result(payload, results_root, task)
    typer.echo(f"wrote {path}")


@app.command()
def export(results_root: Path = typer.Option(Path("results"))) -> None:
    files = sorted((results_root / "llm").glob("*.json"))
    typer.echo(f"{len(files)} result files ready for scripts/build_leaderboard.py --llm-results-dir")


@app.command()
def status(results_root: Path = typer.Option(Path("results"))) -> None:
    for path in sorted((results_root / "preds").glob("*/*/*.jsonl")):
        model_id, task_name = path.parent.parent.name, path.parent.name
        typer.echo(f"{model_id:34s} {task_name:26s} v{path.stem} {len(completed_ids(path)):6d} done")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_cli.py -v`
Expected: PASS, 6 passed. If `test_dry_run_...` fails because task `tiny` is not registered
outside the test fixtures, register it in `conftest.py` via `llm_eval.tasks.register(TinyTask())`
guarded by `if "tiny" not in TASKS`.

- [ ] **Step 5: Write the job agenda script**

Create `scripts/queue_agenda.sh`:

```bash
#!/usr/bin/env bash
# Enqueue the LLM evaluation agenda on a shared, preemptible GPU.
#
# Jobs are idempotent: re-running one that already finished costs a directory
# listing. Killing the queue at any point is safe. Requires `task-spooler`
# (`apt install task-spooler`), which serialises jobs onto one GPU.
set -euo pipefail

MODELS="${MODELS:-bielik-11b-v3-instruct bielik-minitron-7b-v3-instruct pllum-12b-instruct mistral-nemo-12b-instruct llama-pllum-8b-instruct llama-3.1-8b-instruct}"
TASKS="${TASKS:-polemo2 polemo2_in polemo2_out abusive_clauses cdsc_e dyk psc dialogue_acts qa_all qa_kpwr qa_nkjp qa_wikipedia kpwr_ner nkjp_pos aspectemo political_advertising punctuation_restoration}"
VARIANTS="${VARIANTS:-5}"
SHOTS="${SHOTS:-0}"
GPU="${GPU:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

export TS_SOCKET="/tmp/ts-lepiszcze-gpu${GPU}"
tsp -S 1

for task in ${TASKS}; do
  for model in ${MODELS}; do
    for variant in $(seq 0 $((VARIANTS - 1))); do
      tsp -L "${model}/${task}/v${variant}" \
        env CUDA_VISIBLE_DEVICES="${GPU}" \
        uv run --extra gpu --directory llm-eval llm-eval run \
          --model "${model}" --task "${task}" --variant "${variant}" \
          --shots "${SHOTS}" --results-root "../${RESULTS_ROOT}"
    done
  done
done

echo "queued $(tsp -l | tail -n +2 | wc -l) jobs on GPU ${GPU}; watch with: TS_SOCKET=${TS_SOCKET} tsp -l"
```

- [ ] **Step 6: Verify the agenda script end to end with the dummy backend**

Run:

```bash
chmod +x scripts/queue_agenda.sh
cd llm-eval && uv run llm-eval run --model bielik-11b-v3-instruct --task polemo2 \
  --variant 0 --shots 0 --dry-run --limit 8 --results-root ../results
uv run llm-eval status --results-root ../results
```

Expected: `8 generated, 0 cached`, then a status line showing `8 done`. Re-running the same
command must print `0 generated, 8 cached`.

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/cli.py llm-eval/tests/test_cli.py scripts/queue_agenda.sh
git commit -m "feat(llm-eval): add cli and shared-gpu job agenda

Refs #5"
```

---

### Task 14: Track T2 — QLoRA supervised fine-tuning

**Files:**
- Create: `llm-eval/llm_eval/finetune/__init__.py`
- Create: `llm-eval/llm_eval/finetune/lora.py`
- Create: `llm-eval/tests/test_lora.py`

**Interfaces:**
- Consumes: `Task`, `Example`, `ModelSpec`, `render`.
- Produces:
  - `LoraConfig(model_id: str, task_name: str, seed: int, output_dir: Path, epochs: float = 3.0, learning_rate: float = 1e-4, rank: int = 16, alpha: int = 32, max_seq_len: int = 1024, batch_size: int = 1, grad_accum: int = 16)` frozen dataclass.
  - `build_sft_dataset(task: Task, examples: list[Example]) -> list[dict[str, str]]` — chat-format records `{"messages": [...]}` where the assistant turn is the task's canonical JSON answer, reusing `render(..., variant=0, shots=0, train=[])` for the user turn so training and inference see the same prompt.
  - `train_lora(spec: ModelSpec, task: Task, config: LoraConfig) -> Path` — returns the adapter directory. GPU-only.
  - `adapter_path(root: Path, model_id: str, task_name: str, seed: int) -> Path`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_lora.py`:

```python
import json

import pytest

from llm_eval.finetune.lora import LoraConfig, adapter_path, build_sft_dataset


def test_sft_records_are_chat_formatted(tiny_task):
    records = build_sft_dataset(tiny_task, tiny_task.load("train"))
    assert len(records) == 12
    roles = [message["role"] for message in records[0]["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_assistant_turn_is_exactly_the_expected_json(tiny_task):
    examples = tiny_task.load("train")
    records = build_sft_dataset(tiny_task, examples)
    for record, example in zip(records, examples):
        payload = json.loads(record["messages"][-1]["content"])
        assert payload == {"label": example.gold}


def test_training_prompt_matches_the_inference_prompt(tiny_task):
    from llm_eval.prompts import render

    example = tiny_task.load("train")[0]
    record = build_sft_dataset(tiny_task, [example])[0]
    inference = render(tiny_task, example, variant=0, shots=0, train=[])
    assert record["messages"][:2] == inference


def test_adapter_path_layout(tmp_path):
    assert adapter_path(tmp_path, "bielik-11b-v3-instruct", "polemo2", 441) == (
        tmp_path / "adapters" / "bielik-11b-v3-instruct" / "polemo2" / "seed-441"
    )


def test_config_defaults_fit_a_24gb_card():
    config = LoraConfig(model_id="m", task_name="t", seed=441, output_dir=None)
    assert config.max_seq_len <= 1024
    assert config.batch_size * config.grad_accum == 16


@pytest.mark.gpu
def test_train_lora_produces_an_adapter(tmp_path, tiny_task):
    from llm_eval.finetune.lora import train_lora
    from llm_eval.registry import get_model

    config = LoraConfig(
        model_id="bielik-minitron-7b-v3-instruct",
        task_name="tiny",
        seed=441,
        output_dir=tmp_path,
        epochs=0.02,
    )
    path = train_lora(get_model("bielik-minitron-7b-v3-instruct"), tiny_task, config)
    assert (path / "adapter_model.safetensors").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_lora.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.finetune'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/finetune/__init__.py` (empty docstring module) and
`llm-eval/llm_eval/finetune/lora.py`:

```python
"""Track T2: QLoRA supervised fine-tuning on the official train splits.

T1 answers "what do you get for free". T2 is the only apples-to-apples
comparison with a fine-tuned HerBERT, because it uses the same training data.
Training reuses prompt variant 0 verbatim so the model never sees a prompt at
inference that it did not see during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_eval.prompts import render
from llm_eval.registry import ModelSpec
from llm_eval.tasks.base import Example, Task


@dataclass(frozen=True)
class LoraConfig:
    model_id: str
    task_name: str
    seed: int
    output_dir: Path | None
    epochs: float = 3.0
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: int = 32
    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 16


def adapter_path(root: Path, model_id: str, task_name: str, seed: int) -> Path:
    return root / "adapters" / model_id / task_name / f"seed-{seed}"


def _target_answer(task: Task, example: Example) -> str:
    if task.spec.kind == "classification":
        return json.dumps({"label": example.gold}, ensure_ascii=False)
    if task.spec.kind == "sequence_labeling":
        from llm_eval.tasks.sequence_labeling import bio_to_spans, tokens_to_text

        tokens, tags = example.gold
        _, offsets = tokens_to_text(list(tokens))
        return json.dumps(
            {"spans": bio_to_spans(list(tags), list(tokens), offsets)}, ensure_ascii=False
        )
    return json.dumps({"answer": example.gold[0] if example.gold else ""}, ensure_ascii=False)


def build_sft_dataset(task: Task, examples: list[Example]) -> list[dict]:
    records = []
    for example in examples:
        messages = render(task, example, variant=0, shots=0, train=[])
        messages.append({"role": "assistant", "content": _target_answer(task, example)})
        records.append({"messages": messages})
    return records


def train_lora(spec: ModelSpec, task: Task, config: LoraConfig) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    output = adapter_path(config.output_dir or Path("results"), spec.id, task.spec.name, config.seed)
    output.mkdir(parents=True, exist_ok=True)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(build_sft_dataset(task, task.load("train"))),
        peft_config=PeftLoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        args=SFTConfig(
            output_dir=str(output),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            gradient_checkpointing=True,
            learning_rate=config.learning_rate,
            max_seq_length=config.max_seq_len,
            bf16=True,
            seed=config.seed,
            logging_steps=25,
            save_strategy="no",
            report_to=[],
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    return output
```

- [ ] **Step 4: Teach the vLLM backend to serve an adapter**

In `llm-eval/llm_eval/backends/vllm_backend.py`, add an `adapter: Path | None = None`
parameter to `__init__`; when it is set, pass `enable_lora=True, max_lora_rank=16` to `LLM(...)`
and attach `lora_request=LoRARequest("task-adapter", 1, str(adapter))` to every `chat` call.
Add `--adapter` to `llm-eval run` and thread it into `build_backend`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_lora.py -v`
Expected: PASS, 5 passed, 1 deselected

- [ ] **Step 6: Verify one real fine-tune fits in 24 GB**

Run:

```bash
cd llm-eval && CUDA_VISIBLE_DEVICES=0 uv run --extra gpu pytest tests/test_lora.py -m gpu -v
nvidia-smi --query-gpu=memory.used --format=csv
```

Expected: adapter written, peak memory under 24 GB. On OOM, halve `max_seq_len` to 512 before
touching anything else, and record the change in `llm-eval/README.md`.

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/finetune llm-eval/llm_eval/backends/vllm_backend.py llm-eval/llm_eval/cli.py llm-eval/tests/test_lora.py
git commit -m "feat(llm-eval): add qlora fine-tuning track

Refs #5"
```

---

### Task 15: Track T3 — encoder baseline refresh

**Files:**
- Create: `llm-eval/llm_eval/finetune/encoder.py`
- Create: `llm-eval/tests/test_encoder.py`

**Interfaces:**
- Consumes: `Task`, `Example`.
- Produces:
  - `ENCODERS: dict[str, str]` mapping a short id to a pinned `hf_id@revision` for the eight encoders in the spec.
  - `encoder_dataset(task: Task, examples: list[Example]) -> dict` — `{"texts": [...], "labels": [...]}` for classification, `{"tokens": [...], "tags": [...]}` for sequence labeling.
  - `train_and_evaluate(encoder_id: str, task: Task, seed: int, subset_manifest: Path | None) -> dict` — returns the same metric dict shape a `VariantScore` carries, so `aggregate` works unchanged. GPU-only.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_encoder.py`:

```python
import pytest

from llm_eval.finetune.encoder import ENCODERS, encoder_dataset


def test_registry_covers_the_models_from_issue_5():
    assert "xlm-roberta-large" in ENCODERS
    assert "polish-roberta-large-v2" in ENCODERS
    assert "polish-roberta-base-v2" in ENCODERS
    assert "polish-distilroberta" in ENCODERS
    assert "herbert-large-cased" in ENCODERS


def test_every_encoder_pins_a_revision():
    for encoder_id, reference in ENCODERS.items():
        assert "@" in reference, encoder_id
        _, _, revision = reference.partition("@")
        assert len(revision) == 40, encoder_id


def test_classification_dataset_shape(tiny_task):
    data = encoder_dataset(tiny_task, tiny_task.load("test"))
    assert set(data) == {"texts", "labels"}
    assert len(data["texts"]) == 12
    assert data["labels"][0] == "positive"


@pytest.mark.gpu
def test_herbert_large_reproduces_the_historical_polemo_number():
    """The anchor check: if this fails, the protocol is wrong and the track stops."""
    from llm_eval.finetune.encoder import train_and_evaluate
    from llm_eval.tasks import get_task

    metrics = train_and_evaluate("herbert-large-cased", get_task("polemo2"), seed=441, subset_manifest=None)
    # Historical leaderboard: allegro/herbert-large-cased on clarin-pl/polemo2-official.
    # Fill HISTORICAL_MEAN / HISTORICAL_STD from docs/data/leaderboard.json before running.
    historical_mean, historical_std = 0.0, 0.0
    assert abs(metrics["f1_macro"] - historical_mean) <= 2 * historical_std
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.finetune.encoder'`

- [ ] **Step 3: Read the historical anchor value out of the leaderboard**

Run:

```bash
python3 - <<'PY'
import json
data = json.load(open("docs/data/leaderboard.json"))
for entry in data["entries"]:
    if entry["model"] == "allegro/herbert-large-cased" and entry["dataset"] == "clarin-pl/polemo2-official":
        print(json.dumps(entry["metrics"]["f1_macro"], indent=2))
PY
```

Substitute the printed `mean` and `std` into `historical_mean` / `historical_std` in the test.

- [ ] **Step 4: Write the implementation**

Create `llm-eval/llm_eval/finetune/encoder.py` with `ENCODERS` (fill each revision using the
same `HfApi().model_info(...).sha` call as `scripts/resolve_revisions.py`):

```python
"""Track T3: refreshed encoder baselines.

The historical leaderboard rows date to 2021-2022. Comparing a 2026 LLM against
them would flatter the LLM. These encoders are re-trained under the identical
splits, seeds and subsets used by T1 and T2, and this module also closes
issue #5, which asked for exactly this model list.
"""

from __future__ import annotations

from pathlib import Path

from llm_eval.tasks.base import Example, Task

ENCODERS: dict[str, str] = {
    "herbert-large-cased": "allegro/herbert-large-cased@" + "0" * 40,
    "polish-roberta-large-v2": "sdadas/polish-roberta-large-v2@" + "0" * 40,
    "polish-roberta-base-v2": "sdadas/polish-roberta-base-v2@" + "0" * 40,
    "polish-distilroberta": "sdadas/polish-distilroberta@" + "0" * 40,
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large@" + "0" * 40,
    "mdeberta-v3-base": "microsoft/mdeberta-v3-base@" + "0" * 40,
    "eurobert-610m": "EuroBERT/EuroBERT-610m@" + "0" * 40,
    "mmbert-base": "jhu-clsp/mmBERT-base@" + "0" * 40,
}


def encoder_dataset(task: Task, examples: list[Example]) -> dict:
    if task.spec.kind == "sequence_labeling":
        return {
            "tokens": [list(example.gold[0]) for example in examples],
            "tags": [list(example.gold[1]) for example in examples],
        }
    field = next(iter(examples[0].inputs))
    return {
        "texts": [example.inputs[field] for example in examples],
        "labels": [example.gold for example in examples],
    }


def train_and_evaluate(
    encoder_id: str,
    task: Task,
    seed: int,
    subset_manifest: Path | None,
) -> dict[str, float]:
    """Fine-tune one encoder and score it on the same test set the LLMs saw."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    hf_id, _, revision = ENCODERS[encoder_id].partition("@")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    labels = list(task.spec.labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    loader = (
        AutoModelForTokenClassification
        if task.spec.kind == "sequence_labeling"
        else AutoModelForSequenceClassification
    )
    model = loader.from_pretrained(
        hf_id, revision=revision, num_labels=len(labels), torch_dtype=torch.float32
    )

    # Encoding, Trainer wiring and the token-alignment for the sequence-labeling
    # case follow the standard transformers recipes; the only benchmark-specific
    # requirements are: seed=`seed`, the frozen test subset when
    # `subset_manifest` is not None, and metric names produced by `task.score`.
    raise NotImplementedError("implemented in Step 5")
```

- [ ] **Step 5: Finish `train_and_evaluate`**

Complete the function so that it:
1. builds `train`/`validation`/`test` splits with `task.load(...)`, applying
   `apply_subset` to the test split when `subset_manifest` is given;
2. tokenizes with `truncation=True, max_length=512`, and for token classification uses
   `is_split_into_words=True` with `word_ids()` to align labels, writing `-100` on
   continuation sub-tokens;
3. trains with `TrainingArguments(seed=seed, num_train_epochs=5, per_device_train_batch_size=16, learning_rate=2e-5, eval_strategy="epoch", save_strategy="no", bf16=True, report_to=[])`;
4. predicts on the test split, maps ids back to label strings, and returns
   `task.score(golds, preds)` so the metric names match everything else in this plan.

- [ ] **Step 6: Run tests and the anchor check**

Run:

```bash
cd llm-eval && uv run pytest tests/test_encoder.py -v
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu pytest tests/test_encoder.py -m gpu -v
```

Expected: 3 passed on CPU; the GPU anchor test passes. **If the anchor test fails, stop and
fix the protocol before running any other model** — a mismatch means the splits or the metric
differ from the 2022 setup and every downstream comparison would be invalid.

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/finetune/encoder.py llm-eval/tests/test_encoder.py
git commit -m "feat(llm-eval): add refreshed encoder baseline track

Refs #5"
```

---

### Task 16: Contamination audit and reproduction documentation

**Files:**
- Create: `llm-eval/llm_eval/contamination.py`
- Create: `llm-eval/tests/test_contamination.py`
- Create: `llm-eval/README.md`
- Modify: `README.md:26` (add the LLM track to the Resources table)

**Interfaces:**
- Consumes: `Task`, `Example`, `ModelSpec`, `Backend`.
- Produces:
  - `char_ngrams(text: str, n: int = 13) -> set[str]`.
  - `overlap_ratio(test_texts: list[str], corpus_texts: list[str], n: int = 13) -> float`.
  - `no_input_probe(task: Task, examples: list[Example], backend: Backend) -> float` — asks for a label with the input replaced by a placeholder; returns accuracy, which above chance is evidence of memorisation.
  - `chance_level(task: Task, examples: list[Example]) -> float` — majority-class rate.
  - `audit(task, examples, backend, corpus_texts) -> dict` with `overlap_ratio`, `probe_accuracy`, `chance_level`, `suspicious: bool`.

- [ ] **Step 1: Write the failing test**

Create `llm-eval/tests/test_contamination.py`:

```python
import json

import pytest

from llm_eval.backends.dummy import DummyBackend
from llm_eval.contamination import audit, chance_level, char_ngrams, no_input_probe, overlap_ratio


def test_char_ngrams_are_windowed():
    assert char_ngrams("abcdef", n=3) == {"abc", "bcd", "cde", "def"}


def test_short_text_yields_no_ngrams():
    assert char_ngrams("ab", n=13) == set()


def test_overlap_is_one_when_the_corpus_contains_the_test_text():
    assert overlap_ratio(["Kraków to miasto królów"], ["Kraków to miasto królów i smoków"]) == pytest.approx(1.0)


def test_overlap_is_zero_for_unrelated_text():
    assert overlap_ratio(["Kraków to miasto królów"], ["zupełnie inne zdanie o kotach"]) == pytest.approx(0.0)


def test_chance_level_is_the_majority_class(tiny_task):
    assert chance_level(tiny_task, tiny_task.load("test")) == pytest.approx(1 / 3)


def test_probe_accuracy_at_chance_is_not_suspicious(tiny_task):
    examples = tiny_task.load("test")
    backend = DummyBackend(default=json.dumps({"label": "positive"}))
    report = audit(tiny_task, examples, backend, corpus_texts=[])
    assert report["probe_accuracy"] == pytest.approx(1 / 3)
    assert report["suspicious"] is False


def test_probe_above_chance_is_flagged(tiny_task):
    examples = tiny_task.load("test")
    responses = {
        example.id: json.dumps({"label": example.gold}) for example in examples
    }
    report = audit(tiny_task, examples, DummyBackend(responses=responses), corpus_texts=[])
    assert report["probe_accuracy"] == pytest.approx(1.0)
    assert report["suspicious"] is True


def test_no_input_probe_never_shows_the_input(tiny_task, monkeypatch):
    seen = []

    class SpyBackend(DummyBackend):
        def generate(self, requests):
            seen.extend(request.messages[-1]["content"] for request in requests)
            return super().generate(requests)

    no_input_probe(tiny_task, tiny_task.load("test"), SpyBackend(default='{"label": "positive"}'))
    assert seen
    assert all("zdanie 0" not in content for content in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm-eval && uv run pytest tests/test_contamination.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_eval.contamination'`

- [ ] **Step 3: Write the implementation**

Create `llm-eval/llm_eval/contamination.py`:

```python
"""Contamination audit.

PolEmo2, KLEJ, NKJP and KPWr have been public for years and are plausibly in
the pre-training data of every Polish model in this benchmark. RQ1 cannot be
answered honestly without measuring that, so this module ships alongside the
results rather than as an afterthought.
"""

from __future__ import annotations

from collections import Counter

from llm_eval.backends import Backend, GenerationRequest
from llm_eval.tasks.base import Example, Task

PROBE_MARGIN = 0.10


def char_ngrams(text: str, n: int = 13) -> set[str]:
    if len(text) < n:
        return set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def overlap_ratio(test_texts: list[str], corpus_texts: list[str], n: int = 13) -> float:
    corpus: set[str] = set()
    for text in corpus_texts:
        corpus |= char_ngrams(text, n)
    if not corpus:
        return 0.0
    hits = 0
    total = 0
    for text in test_texts:
        grams = char_ngrams(text, n)
        if not grams:
            continue
        total += 1
        hits += 1 if grams & corpus else 0
    return hits / total if total else 0.0


def chance_level(task: Task, examples: list[Example]) -> float:
    counts = Counter(task.stratum(example) for example in examples)
    return max(counts.values()) / len(examples) if examples else 0.0


def no_input_probe(task: Task, examples: list[Example], backend: Backend) -> float:
    """Ask for the label with the input withheld. Above chance means memorisation."""
    requests = [
        GenerationRequest(
            id=example.id,
            messages=[
                {"role": "system", "content": "Odpowiadasz wyłącznie obiektem JSON."},
                {
                    "role": "user",
                    "content": (
                        f"Zbiór: {task.spec.dataset}. Przykład numer {index} ze splitu testowego. "
                        f"Podaj etykietę tego przykładu. Zwróć JSON: {{\"label\": \"<etykieta>\"}}"
                    ),
                },
            ],
            json_schema=task.output_schema(),
            max_tokens=task.spec.max_new_tokens,
        )
        for index, example in enumerate(examples)
    ]
    results = backend.generate(requests)
    hits = 0
    for example, result in zip(examples, results):
        parsed = task.parse(result.text, example)
        hits += 1 if parsed.ok and parsed.prediction == example.gold else 0
    return hits / len(examples) if examples else 0.0


def audit(
    task: Task,
    examples: list[Example],
    backend: Backend,
    corpus_texts: list[str],
) -> dict:
    texts = [" ".join(example.inputs.values()) for example in examples]
    probe = no_input_probe(task, examples, backend)
    chance = chance_level(task, examples)
    return {
        "task": task.spec.name,
        "overlap_ratio": overlap_ratio(texts, corpus_texts),
        "probe_accuracy": probe,
        "chance_level": chance,
        "suspicious": probe > chance + PROBE_MARGIN,
    }
```

Add an `audit` command to `llm_eval/cli.py`:

```python
@app.command("audit")
def audit_command(
    model: str = typer.Option(...),
    task: str = typer.Option(...),
    corpus: Path = typer.Option(None, help="Optional text file, one document per line"),
    results_root: Path = typer.Option(Path("results")),
) -> None:
    from llm_eval.contamination import audit as run_audit

    spec = get_model(model)
    task_obj = get_task(task)
    corpus_texts = corpus.read_text(encoding="utf-8").splitlines() if corpus else []
    report = run_audit(task_obj, _load_examples(task_obj, results_root.parent), build_backend(spec), corpus_texts)
    path = results_root / "contamination" / f"{spec.id}__{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-eval && uv run pytest tests/test_contamination.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Write the reproduction guide**

Create `llm-eval/README.md`:

````markdown
# LEPISZCZE LLM evaluation track

Reproduces every LLM row on the [LEPISZCZE leaderboard](https://clarin-pl.github.io/LEPISZCZE/leaderboard/).

Spec: `planning/specs/2026-09-01-llm-eval-track.md`.
Plan: `planning/plans/2026-09-01-llm-eval-track.md`.

## Requirements

- One NVIDIA GPU with 24 GB (RTX 3090 / A5000). Ampere: no FP8, int4 is AWQ/GPTQ + Marlin.
- `uv`, and `task-spooler` if you want the shared-GPU queue.

## Install

```bash
cd llm-eval
uv sync --extra gpu
```

## Freeze the test subsets (once, already committed)

```bash
for task in nkjp_pos punctuation_restoration; do
  uv run llm-eval freeze-subsets --task "$task" --cap 2000 --root ..
done
git status --short ../data/subsets
```

## Run one cell

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu llm-eval run \
  --model bielik-11b-v3-instruct --task polemo2 --variant 0 --shots 0 \
  --results-root ../results
```

Safe to kill at any time. Re-running resumes from disk and regenerates nothing.

## Run the whole agenda on a shared GPU

```bash
GPU=0 ../scripts/queue_agenda.sh
TS_SOCKET=/tmp/ts-lepiszcze-gpu0 tsp -l
uv run llm-eval status --results-root ../results
```

## Score and publish

```bash
uv run llm-eval score --model bielik-11b-v3-instruct --task polemo2 --variants 5 --results-root ../results
cd .. && python3 scripts/build_leaderboard.py \
  --base docs/data/leaderboard.json \
  --llm-results-dir results/llm \
  --source-commit a434193a871c7d1bccdb5fd9106be5a950e38aad \
  --llm-commit "$(git rev-parse HEAD)" \
  --output docs/data/leaderboard.json
python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json
```

## Contamination audit

```bash
uv run --extra gpu llm-eval audit --model bielik-11b-v3-instruct --task polemo2 --results-root ../results
```

## Tests

```bash
uv run pytest              # CPU only, no GPU required
uv run --extra gpu pytest -m gpu   # GPU path
```
````

Add one row to the Resources table in the repo `README.md`:

```markdown
| LLM track             | Prompting, QLoRA and refreshed encoder baselines over every LEPISZCZE task, with inference-cost columns | [llm-eval/](llm-eval/README.md) |
```

- [ ] **Step 6: Verify the whole suite and the site**

Run:

```bash
cd llm-eval && uv run pytest -v && cd ..
python3 -m pytest tests/test_build_leaderboard.py -v
python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json
python3 scripts/check_datasets.py docs/data/datasets.json
python3 scripts/check_site.py docs
node --check docs/assets/leaderboard.js
```

Expected: every command exits 0

- [ ] **Step 7: Commit**

```bash
git add llm-eval/llm_eval/contamination.py llm-eval/llm_eval/cli.py llm-eval/tests/test_contamination.py llm-eval/README.md README.md
git commit -m "feat(llm-eval): add contamination audit and reproduction guide

Refs #5"
```

---

## Execution order and compute budget

Tasks 1–13 are strictly sequential: each builds on the interfaces of the previous one. Tasks
14, 15 and 16 are independent of each other and can run in parallel worktrees, but all three
depend on Task 13.

Recommended first experiment, immediately after Task 13 lands — it answers RQ1 with two
models and one task, in under an hour of GPU time:

```bash
cd llm-eval
for model in pllum-12b-instruct mistral-nemo-12b-instruct; do
  for variant in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 uv run --extra gpu llm-eval run \
      --model "$model" --task polemo2_in --variant "$variant" --shots 0 --results-root ../results
  done
  uv run llm-eval score --model "$model" --task polemo2_in --variants 5 --results-root ../results
done
```

Rough budget for the full agenda on one 24 GB card, from the generation-length differences
between task kinds:

| Track | Cells | Est. GPU-hours |
| --- | --- | --- |
| T1 zero-shot, classification + QA (12 datasets x 10 models x 5 variants) | 600 | 60–90 |
| T1 zero-shot, sequence labeling (5 datasets x 10 models x 5 variants, 2000-capped) | 250 | 120–180 |
| T1 5-shot, same coverage | 850 | 200–280 |
| T2 QLoRA (7 models x 17 datasets x 3 seeds) | 357 | 250–400 |
| T3 encoders (8 models x 17 datasets x 5 seeds) | 680 | 40–60 |

Two cards at partial availability put the full programme in the range of one to two months of
wall-clock. Cut scope by dropping the 5-shot row first — it is the least informative per
GPU-hour once T2 exists.

## Self-review notes

- **Spec coverage.** RQ1 → Task 2 `paired_models` + the first experiment above. RQ2 → Tasks 9
  and 15. RQ3 → Tasks 7, 11, 12. RQ4 → Tasks 5 and 11. RQ5 → Task 16. Every acceptance
  criterion in spec §8 has a verification step: CPU suite (Task 1 CI), leaderboard validators
  (Task 12 Step 7 and Task 16 Step 6), byte-identical migration (Task 12 Step 5), HerBERT
  anchor (Task 15 Step 6), pinned revisions (Task 2 Step 5), non-null evaluation fields
  (Task 12 validator).
- **Known unknowns, each with an explicit verification step rather than a guess:** dataset
  column names and label vocabularies (Task 4 Step 4, Task 9 Step 4), the QA source dataset
  and its four subsets (Task 10 Step 4), exact Hub ids and revisions for every model
  (Task 2 Step 5, Task 15 Step 4), and the historical HerBERT anchor value (Task 15 Step 3).
  None of these can be settled from this repository alone; all four fail loudly rather than
  silently producing wrong numbers.
- **Deliberate design decisions worth not re-litigating during execution:** span metrics live
  in `evaluation.extra`, never in `metrics`, so encoder rows do not sprout empty columns;
  parse failures are scored as wrong answers, never dropped; the legacy `lepiszcze` package
  and its Python 3.9 pins are untouched; one model = one GPU everywhere.

## Execution Handoff

Plan complete and saved to `planning/plans/2026-09-01-llm-eval-track.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in one session using `superpowers:executing-plans`, batch execution with checkpoints.
