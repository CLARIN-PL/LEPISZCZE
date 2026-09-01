# LEPISZCZE LLM Evaluation Track — Specification

**Status:** approved for implementation
**Date:** 2026-09-01
**Refs:** [#5 Models proposed to test](https://github.com/CLARIN-PL/LEPISZCZE/issues/5)
**Owner:** Łukasz Augustyniak

## 1. Problem

The LEPISZCZE leaderboard (`docs/data/leaderboard.json`, 86 entries, schema v2) contains
only fine-tuned encoder baselines from 2021–2022: HerBERT base/large, `dkleczek` Polish
BERT, `xlm-roberta-base`, `sentence-transformers/paraphrase-xlm-r-multilingual-v1`,
`clarin-knext/plt5-large-poquad` and an `XGBoost+Tfidf` reference. No decoder LLM has ever
been evaluated on LEPISZCZE, and the Polish LLM ecosystem (Bielik, PLLuM, Qra) has never
been measured against these fine-tuned baselines on a common, versioned test protocol.

## 2. Research questions

**RQ1 — Does Polish-specific adaptation still pay off in 2026?**
Measured with paired comparisons: each Polish-adapted model against the exact base model it
was continued from (Bielik-11B vs its base, `Llama-PLLuM-8B` vs `Llama-3.1-8B`,
`PLLuM-12B` vs `Mistral-Nemo-12B`). Same architecture, same parameter count, same protocol.

**RQ2 — Where do prompted LLMs beat fine-tuned encoders, and where do they lose?**
Hypothesis: LLMs win on extractive QA and punctuation restoration, lose on token-level
tagging (NKJP-POS, KPWr-NER, AspectEmo, Political Advertising) where a 110M encoder with a
task head remains stronger and 100–1000× cheaper.

**RQ3 — What is the quality/cost Pareto front?**
LEPISZCZE has reported training cost since the 2022 paper. This track extends that to
inference: GPU-seconds per 1k examples, energy (Wh), peak VRAM, output-parse failure rate.

**RQ4 — How much does prompt choice matter relative to training seed?**
Encoder entries are averaged over 5 training seeds. LLM entries are averaged over 5 prompt
variants. The two standard deviations are directly comparable and reported side by side.

**RQ5 — Are the benchmark test splits contaminated?**
PolEmo2, KLEJ, NKJP and KPWr have been public for years and are plausibly in the
pre-training corpora of Bielik and PLLuM. Contamination must be measured, not assumed away.

## 3. Scope

### 3.1 In scope — tasks

All 17 dataset variants already on the leaderboard, unchanged splits:

| Kind | Datasets |
| --- | --- |
| classification (sentence / pair) | `clarin-pl/polemo2-official`, `allegro/klej-polemo2-in`, `allegro/klej-polemo2-out`, `laugustyniak/abusive-clauses-pl`, `allegro/klej-cdsc-e`, `allegro/klej-dyk`, `allegro/klej-psc`, `clarin-pl/dialogue-acts` |
| sequence labeling | `clarin-pl/kpwr-ner`, `clarin-pl/nkjp-pos`, `clarin-pl/aspectemo`, `laugustyniak/political-advertising-pl`, `clarin-pl/2021-punctuation-restoration` |
| extractive QA | `qa_all`, `qa_kpwr`, `qa_nkjp`, `qa_wikipedia` |

No new datasets. No new tasks.

### 3.2 In scope — evaluation tracks

- **T1 prompting** — zero-shot and 5-shot, no weight updates, 5 prompt variants per task.
- **T2 QLoRA SFT** — LoRA fine-tuning on the same official train splits, 3 seeds.
- **T3 encoder refresh** — modern encoder baselines re-run under the identical protocol so
  the LLM comparison is not against 2021 numbers.

### 3.3 Out of scope

- New benchmark datasets (tracked separately in `docs/data/datasets.json`).
- Closed API models. Anything not reproducible from open weights on the described hardware
  is excluded from the leaderboard and may appear only in the paper as a non-comparable
  reference row.
- Models that do not fit a single 24 GB GPU in 4-bit (`Llama-PLLuM-70B`,
  `PLLuM-8x7B` at bf16). They are listed as known gaps.

## 4. Hardware constraints

Two independent hosts, no NVLink, no shared interconnect. One model = one GPU; no
cross-host tensor parallelism.

| GPU | VRAM | Notes |
| --- | --- | --- |
| RTX 3090 | 24 GB | Ampere — no native FP8. Use AWQ/GPTQ int4 with Marlin kernels. |
| RTX A5000 | 24 GB | Same constraint, second host. |
| RTX A4000 | 16 GB | Local, partly occupied. Encoder track and ≤8B int4 inference. |

GPUs are shared and availability is intermittent. **Every long-running job must be killable
at any moment and resumable with zero recomputation.** This is a hard requirement, not an
optimization: it drives the append-only per-example prediction store in §6.3.

VRAM budget used for planning:

| Size | bf16 | int4 (AWQ/GPTQ) | QLoRA on 24 GB |
| --- | --- | --- | --- |
| 7–8B | 16 GB — fits 3090/A5000 | ~5 GB — fits A4000 | yes, seq 2048 |
| 11–14B | does not fit with KV cache | ~8 GB | yes, seq 1024, grad checkpointing |
| 20–27B | no | ~15 GB — fits 3090/A5000 | seq 512, batch 1, slow |
| 47B MoE / 70B | no | no | no |

## 5. Model roster

Every model is pinned by `(hf_id, revision)`. A registry entry whose repo or revision does
not resolve on the Hugging Face Hub is a hard failure, not a warning.

### 5.1 Polish-adapted (T1 + T2)

| Model | Params | Role |
| --- | --- | --- |
| `speakleash/Bielik-11B-v3.0-Instruct` | 11B | flagship Polish instruct |
| `speakleash/Bielik-Minitron-7B-v3.0-Instruct` | 7.35B | pruned, cheaper Pareto point |
| `speakleash/Bielik-11B-v2.6-Instruct` | 11B | previous generation, progress-over-time axis |
| `CYFRAGOVPL/PLLuM-12B-instruct` | 12B | PLLuM, Mistral-Nemo lineage |
| `CYFRAGOVPL/Llama-PLLuM-8B-instruct` | 8B | PLLuM, Llama-3.1 lineage |
| `OPI-PG/Qra-7B` | 7B | base-only, T2 track exclusively |

### 5.2 Open multilingual controls (T1 + T2)

| Model | Params | Role |
| --- | --- | --- |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | **paired base for PLLuM-12B** |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | **paired base for Llama-PLLuM-8B** |
| Gemma 4 12B instruct | 12B | Apache 2.0, 140+ languages |
| Gemma 4 4B instruct | 4B | small multilingual point |
| Qwen 3.6 8B instruct | 8B | strong multilingual dense |
| Qwen3.8-27B | 27B | int4 only, largest dense that fits |
| `utter-project/EuroLLM-9B-Instruct` | 9B | EU-funded, Polish in pre-training |
| `openai/gpt-oss-20b` | 20B (≈3.6B active) | MoE, cheap inference on a 3090 |

### 5.3 Encoder refresh (T3)

`allegro/herbert-large-cased` (anchor — must reproduce the historical number within its
reported std), `sdadas/polish-roberta-large-v2`, `sdadas/polish-roberta-base-v2`,
`sdadas/polish-distilroberta`, `xlm-roberta-large`, `microsoft/mdeberta-v3-base`,
`EuroBERT-610M`, `mmBERT-base`. This closes issue #5 directly.

## 6. Protocol

### 6.1 Splits and subsampling

Official test splits, unchanged, for every task where `|test| <= 2000`.

For larger test sets (NKJP-POS, punctuation restoration) the LLM tracks evaluate a
**stratified subsample of 2000 examples, seed 441** — the seed already used across the
repo's dataset configs. The subset is frozen in `data/subsets/<task>.json` with the example
ids and a SHA-256 digest. **Encoder entries are re-scored on the same subset** and reported
in a dedicated column, so no comparison ever crosses subsets. Full-test encoder numbers stay
on the leaderboard unchanged.

### 6.2 Output format and parsing

Free-form generation is not scored. Every task defines a JSON schema and generation uses
vLLM guided decoding (`xgrammar`). Sequence labeling is emitted as spans, never as per-token
BIO tags, because LLM tokenization does not align with gold tokenization:

```json
[{"span": "Jan Kowalski", "start": 12, "label": "nam_liv_person"}]
```

Spans are aligned back onto gold token offsets. Two metrics are reported: strict span F1 and
relaxed (partial-overlap) F1. **Parse-failure rate is a first-class reported metric** — it is
a real cost of the generative approach that encoders do not pay. A failed parse counts as an
incorrect prediction; it is never silently dropped.

### 6.3 Resumability

Predictions are stored append-only, one JSON object per example, at
`results/preds/<model_id>/<task>/<variant>.jsonl`. A run computes the set difference between
the task's example ids and the ids already present, generates only the missing ones, and
appends with `flush()` + `os.fsync()`. Killing a job mid-batch loses at most one batch. This
makes preemptible, shared GPUs usable without a scheduler.

### 6.4 Cost accounting

Every run writes a sidecar `<variant>.meta.json` with wall-clock seconds, GPU-seconds,
energy in Wh (NVML energy counters), peak VRAM, prompt/completion token counts, engine
version and GPU model. Reported as per-1000-examples figures on the leaderboard.

### 6.5 Aggregation

Per (model, task): 5 prompt variants for T1, 3 seeds for T2, 5 seeds for T3. Reported as
`metrics_avg` / `metrics_std`, matching the historical schema exactly.

### 6.6 Contamination audit

For every (model, dataset) pair: character 13-gram overlap between the test split and any
publicly documented pre-training corpus manifest, plus a no-input probe (prompt the model
with the example id or a truncated prefix and measure label recovery above chance).
Results are published as a separate table. Any RQ1 conclusion must be stated in light of it.

## 7. Deliverables

1. `llm-eval/` — a standalone, uv-managed Python 3.11 sub-project. The legacy `lepiszcze`
   package (Python 3.9, torch 1.11, transformers 4.18) is **not modified and not imported**;
   those pins are incompatible with vLLM and any modern model.
2. Per-example predictions and run metadata under `results/`, committed for reproducibility.
3. Leaderboard schema v3 with multiple sources, an `evaluation` block per entry, and
   `scripts/build_leaderboard.py` extended to merge LLM results into the existing snapshot
   without re-cloning the `embeddings` repository.
4. A one-command reproduction path documented in `llm-eval/README.md`.
5. CI that runs the whole test suite on CPU with a deterministic dummy backend.

## 8. Acceptance criteria

- `uv run pytest` green on a machine with no GPU.
- `python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json` green.
- `python3 scripts/check_datasets.py docs/data/datasets.json` green.
- `python3 scripts/check_site.py docs` green, `node --check docs/assets/leaderboard.js` green.
- The 86 historical entries are byte-identical after the v3 migration except for the added
  `source` field.
- `herbert-large-cased` re-run under T3 reproduces its historical `f1_macro` on PolEmo2
  within 2 standard deviations. If it does not, the protocol is wrong and the track stops.
- Every registry model resolves to a pinned Hub revision.
- Every leaderboard LLM entry carries a non-null `evaluation.protocol`,
  `evaluation.parse_failure_rate` and `evaluation.energy_wh_per_1k`.
