# PRE-REGISTRATION — Budget-Matched, Oracle-Free Comparison of LLM Agent Scaffolds

**Frozen before main runs.** Date frozen: 2026-09-29.
Source: §2 of `docs/IMPLEMENTATION_PLAN.md`, copied verbatim below.
This file is never edited after freezing; every later change is recorded in `docs/DEVIATIONS.md`
with its date and reason.

---

## 2. Frozen experimental design (this section IS the pre-registration — copy it to `PREREGISTRATION.md` in T1)

### 2.1 Model and sampling
- Model: `claude-haiku-4-5` (Anthropic API). Pricing constants (USD per 1M tokens): input **1.00**, output **5.00**, 5-min cache write **1.25** (1.25×), cache read **0.10** (0.1×). Store in `config.py`, never hard-code elsewhere.
- **Minimum cacheable prefix for Haiku 4.5 = 4,096 tokens** (shorter prefixes silently don't cache). This is why §2.6 defines three pricing regimes.
- No extended thinking (omit the `thinking` parameter entirely).
- `temperature = 0.7` for **all** generative calls (actor, planner, executor, solver, worker-LLM, reflector). `temperature = 0.0` for the **judge**. Rationale: independent samples are needed for replicates and vote@k; identical across scaffolds.
- The API has no seed parameter. "Replicate r" = an independent sample; `r` is recorded, not a seed.
- `max_tokens` per role: actor step 512, planner 1024, executor step 512, solver 256, worker_llm 256, reflector 512, judge 256.

### 2.2 Data
- **HotpotQA:** `hotpotqa/hotpot_qa`, config `distractor`, split `validation` (7,405 items). Parquet: `https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet` (resolves via HF API; download the file it points to).
- **GSM8K:** `openai/gsm8k`, config `main`, split `test` (1,319 items) + split `train` (for exemplars only). Parquet URLs: `https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet` and `.../main/train/0.parquet`.
- Sampling: `N = 300` per benchmark, `numpy.random.default_rng(2026)`. HotpotQA sampled **stratified by `type`** (bridge / comparison) proportional to the split. Remove any item whose question string equals a few-shot exemplar question. Save the sampled IDs to `data/samples/{benchmark}_N300.json` — this file is frozen once created; never resample.
- **Pilot subset** = the first 30 IDs of each frozen sample (in file order).
- Do NOT use HotpotQA's provided context paragraphs. Agents retrieve via the Wikipedia tool (as in ReAct/ReWOO/Reflexion).

### 2.3 Tools (identical for all scaffolds within a benchmark)
- **HotpotQA:** `Search[entity]`, `Lookup[keyword]` — ReAct semantics (§T5). ReWOO additionally has `LLM[instruction]` as a worker (part of the ReWOO paradigm; it is an LLM call, logged and costed like any other).
- **GSM8K:** `Calculate[expression]` (safe arithmetic). ReWOO additionally has `LLM[...]`.
- `Lookup` state = the page from the most recent successful `Search` **within the same scaffold run** (one run = one scaffold on one task for one replicate).
- Every observation is truncated to **1,500 characters** before being shown to any LLM.
- All Wikipedia responses are cached on disk (§T5). The cache is what guarantees every scaffold sees identical content for identical queries.

### 2.4 Conditions (scaffolds) and replicates

| Scaffold ID | Definition (full spec in T7–T10) | Replicates per task |
|---|---|---|
| `react` | ReAct loop, max 7 steps | **K = 10** (the pool; `r = 0..9`) |
| `rewoo` | Planner → worker execution → solver | R = 3 |
| `plan_exec` | Planner → per-step ReAct executor (≤4 actions/step, ≤5 steps) → solver. No replanning. | R = 3 |
| `reflexion_gt` | ReAct trials, stop when **exact-match vs gold** is true (the original paper's HotpotQA setup), ≤5 trials, memory 3 | R = 3 |
| `reflexion_self` | Same, but stop when an **LLM judge (no gold access)** says correct | R = 3 |
| `react_vote@k` | **Derived post hoc** from the pool: majority vote over k pool samples, k = 1..10. No extra API calls. `react_vote@1` ≡ ReAct. | — |

- **ReAct baseline replicates** = pool samples `r = 0,1,2`.
- **Reflexion trial 1 reuse:** for `reflexion_*` replicate `r`, trial 1 is **the pool trajectory `react` r** (identical prompt and distribution; saves ~20% cost and makes comparisons paired). Its tokens and dollars are counted in full in the Reflexion run's cost.
- Final answer if a run halts without `Finish`: empty string `""` (graded incorrect).

### 2.5 Grading
- HotpotQA: official `hotpot_evaluate_v1` normalization (lowercase, strip punctuation, remove articles `a|an|the`, collapse whitespace). **EM is primary**, F1 secondary (including the official yes/no/noanswer special case).
- GSM8K: gold = number after `####`, commas removed. Prediction = last number in the final answer string (regex `-?\d[\d,]*(?:\.\d+)?`, commas removed). Correct iff `abs(pred - gold) < 1e-6`. No number → incorrect.
- Grading is post hoc and **never** visible to agents — except as the stopping signal inside `reflexion_gt`, which is the point of that condition.

### 2.6 Cost metrics and the three pricing regimes
Per LLM call we log `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`. Define `I = input_tokens + cache_creation_input_tokens + cache_read_input_tokens` (total input) and `O = output_tokens`.
1. **`usd_nocache`** = `(I × 1.00 + O × 5.00) / 1e6` — every input token at full price.
2. **`usd_empirical`** = `(input_tokens × 1.00 + cache_creation × 1.25 + cache_read × 0.10 + O × 5.00) / 1e6` — what the API actually charged with our cache breakpoints.
3. **`usd_ideal[L]`** for L ∈ {0, 1024, 4096} — a deterministic, schedule-independent counterfactual of perfect prefix caching with minimum length L (algorithm in T14.2). L=4096 is Haiku's real threshold; L=1024 is the threshold of most larger current models.

Headline cost metric: **cost-of-pass** = `mean usd per task / accuracy` (Erol et al. 2025), reported in all regimes. Also: tokens/task (`I+O`), tokens/correct, LLM calls/task, tool calls/task, wall-clock/task.

### 2.7 Hypotheses (pre-registered; two-sided tests; α = 0.05, Holm-corrected within each benchmark)
- **H1 (budget matching).** For each scaffold X ∈ {rewoo, plan_exec, reflexion_self, reflexion_gt}: let `Δdef(X) = Acc(X) − Acc(react)` and `Δmatch(X) = Acc(X) − Acc(react_vote@k*(X))`, where k*(X) matches X's mean `I+O` tokens/task. H1 predicts `Δmatch(X) ≤ 0.5 × Δdef(X)` for every X with `Δdef(X) > 0`.
- **H2 (oracle premium).** `Acc(reflexion_gt) − Acc(reflexion_self) > 0`. Cost direction is **not** pre-specified: it depends on the judge's error profile (false "correct" → early stop → cheaper; false "incorrect" → wasted retries → costlier). We pre-register reporting the **judge confusion matrix vs. gold** as the explanation.
- **H3 (caching changes rankings).** The cost ratio `cost(rewoo)/cost(react)` is closer to 1 under `usd_ideal[1024]` than under raw tokens (`I+O`). Secondary: under `usd_ideal[4096]` and `usd_empirical` on Haiku 4.5 the effect is expected to be small because most single prompts are < 4,096 tokens — itself a reportable finding (rankings depend on the model's cache threshold).
- **H4 (predictability).** A logistic model on first-attempt features (T14.6) predicts "Reflexion-Self fixes a failed first attempt" with cross-validated AUC greater than a benchmark-identity-only baseline.

### 2.8 Statistics
- Unit of analysis = **task**. Per task, a scaffold's accuracy = mean correctness over its replicates (for vote@k: expected correctness over draws, T14.3).
- CIs: **cluster bootstrap over tasks**, B = 2,000, `rng = default_rng(2026)`, percentile 95% CI. Applied to accuracy, mean cost, cost-of-pass, and all paired differences and ratios (paired = same resampled task set for both arms).
- Secondary test: exact McNemar on replicate-0 binary outcomes for each pre-registered pair.
- Pre-registered comparison family per benchmark (Holm): {X vs react} ×4, {X vs react_vote@k*(X)} ×4, {reflexion_gt vs reflexion_self} ×1 → 9 comparisons.

