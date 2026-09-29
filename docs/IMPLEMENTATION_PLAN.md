# IMPLEMENTATION PLAN — Budget-Matched, Oracle-Free Comparison of LLM Agent Scaffolds

**Project root:** `/home/dev/Documents/DL-Research/agent-cost-study/`
**Companion doc:** `../Lit-review/00_Literature_Review_and_Problem_Statement.md` (why; this doc is how)
**Version:** 1.0 (sprint edition — concrete results in 4 days)
**Audience:** a coding LLM or engineer. Implement tasks **in order T0 → T15**. Every task has an exact spec and acceptance tests. **Do not invent design choices** — every choice is frozen in §2; if something seems unspecified, the answer is in §2 or §8. If it truly is not, choose the option that keeps all scaffolds *identical except for their control flow*, and log it in `docs/DEVIATIONS.md`.

---

## 0. What we are building, in one paragraph

One LLM (`claude-haiku-4-5`), one tool set, one task set. Five agent scaffolds — **ReAct, ReWOO, Plan-and-Execute, Reflexion-GT, Reflexion-Self** — are run on the same HotpotQA and GSM8K tasks. Every LLM call goes through one wrapper that logs exact token usage including cache fields. ReAct is additionally run **10 times per task** (the "pool"), which gives us, post hoc and at no extra cost, the **budget-matched control**: majority-vote-over-k ReAct samples (`vote@k`) at any budget. Analysis then answers: (RQ1) does any scaffold beat budget-matched ReAct? (RQ2) how much of Reflexion's gain is the ground-truth stopping oracle? (RQ3/H3) do scaffold cost rankings change when priced in dollars with prompt caching? (RQ4) can cheap first-attempt features predict when the expensive scaffold pays off?

---

## 1. Sprint scope (frozen)

| In scope | Out of scope for this sprint (and why) |
|---|---|
| HotpotQA (distractor dev, tool-heavy) + GSM8K (test, reasoning-heavy) | **GAIA** — needs web search + file tools, heterogeneous grading, high cost; not doable properly in 4 days. Future work. |
| One fixed model: `claude-haiku-4-5` | **Second model family** — future work; the design supports it via config. |
| 5 scaffolds + ReAct pool (vote@k control) | **Batch API** — agent loops are sequential multi-turn; 24h batch latency is incompatible. |
| Token, dollar (3 pricing regimes), latency, call counts | Latency is logged but not a primary outcome (depends on API load). |
| Paired bootstrap stats, McNemar, Holm correction | Formal power analysis. |
| RQ4 predictor (logistic regression, 5-fold CV) | Any trained/fine-tuned model. |

**Honest caveat to state in any write-up:** this is a controlled study of *mechanism instantiations* on terminal-answer tasks, not of production agents, which are hybrids (Rombaut 2026).

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

---

## 3. Repository layout

```
agent-cost-study/
├── pyproject.toml              # deps (T0)
├── .env.example                # ANTHROPIC_API_KEY=
├── PREREGISTRATION.md          # T1 (committed BEFORE main runs)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md  # this file
│   └── DEVIATIONS.md           # any change from this plan, dated
├── src/acs/
│   ├── __init__.py
│   ├── config.py               # T2 all constants
│   ├── pricing.py              # T2 usd functions
│   ├── llm.py                  # T3 wrapper, logging, cache_control, FakeLLM
│   ├── data.py                 # T4 download, sample, load
│   ├── grading.py              # T4 EM/F1/GSM8K
│   ├── tools/
│   │   ├── wiki.py             # T5 Search/Lookup + sqlite cache
│   │   ├── calc.py             # T5 Calculate
│   │   └── toolbox.py          # T5 per-run tool state + dispatch
│   ├── prompts/
│   │   ├── vendor/             # T6 verbatim upstream files (read-only)
│   │   └── library.py          # T6 all prompt strings, built from vendor + authored
│   ├── scaffolds/
│   │   ├── base.py             # T7 RunContext, RunResult, parse helpers
│   │   ├── react.py            # T7 (also exports react_loop used by others)
│   │   ├── rewoo.py            # T8
│   │   ├── plan_exec.py        # T9
│   │   └── reflexion.py        # T10 (gt + self)
│   ├── runner.py               # T11 CLI
│   └── analysis/
│       ├── load.py             # T14 read JSONL → DataFrames
│       ├── costs.py            # T14.1–14.2 regimes + ideal cache model
│       ├── vote.py             # T14.3 vote@k
│       ├── stats.py            # T14.4 bootstrap, McNemar, Holm
│       ├── rq4.py              # T14.6 predictor
│       └── report.py           # T14.7 tables + figures
├── tests/                      # pytest; network tests marked @pytest.mark.network
├── data/{raw,samples}/
├── cache/wiki.sqlite
└── results/{run_tag}/          # calls.jsonl, runs.jsonl, config.json, prompt_tokens.json, analysis/
```

---

## 4. Data contracts (do not change field names — analysis depends on them)

### 4.1 `results/{run_tag}/calls.jsonl` — one line per LLM API call
```json
{"call_id": "uuid4", "run_id": "hotpotqa:5a8b57f2:react:3", "benchmark": "hotpotqa",
 "task_id": "5a8b57f2", "scaffold": "react", "replicate": 3,
 "role": "actor|planner|executor|solver|worker_llm|reflector|judge",
 "conversation_id": "run_id + ':' + short label, e.g. ':trial2' or ':step3'",
 "turn_index": 0, "system_prompt_id": "react_hotpotqa_v1",
 "model": "claude-haiku-4-5", "temperature": 0.7, "max_tokens": 512,
 "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
 "output_tokens": 0, "stop_reason": "end_turn|stop_sequence|max_tokens",
 "latency_s": 1.23, "ts": "ISO-8601", "attempts": 1, "error": null}
```
`conversation_id` + `turn_index`: calls in the same conversation are **append-only** (call t's message list = call t−1's messages + assistant reply + new user message). Every scaffold must respect this so T14.2 is valid.

### 4.2 `results/{run_tag}/runs.jsonl` — one line per scaffold run
```json
{"run_id": "...", "benchmark": "...", "task_id": "...", "scaffold": "...", "replicate": 0,
 "status": "ok|error", "error": null,
 "final_answer": "...", "gold": "...", "em": 1, "f1": 1.0, "correct": 1,
 "n_llm_calls": 5, "n_tool_calls": {"Search": 3, "Lookup": 1, "Calculate": 0, "LLM": 0},
 "n_steps": 5, "halted": false,
 "tokens_in": 0, "tokens_out": 0, "usd_nocache": 0.0, "usd_empirical": 0.0,
 "wall_s": 12.3,
 "extra": {},
 "trace": [ {"kind": "llm", "role": "actor", "call_id": "...", "text": "..."},
            {"kind": "tool", "tool": "Search", "input": "...", "observation": "..."} ]}
```
`extra` per scaffold: ReWOO `{"plan_parse_ok": bool, "n_evidence": int, "plan_text": str}`; Plan-Exec `{"plan_steps": [str], "step_results": [str], "plan_parse_ok": bool}`; Reflexion `{"n_trials": int, "trial_answers": [str], "trial_correct": [0/1], "judge_verdicts": [str|null], "judge_parse_fail": int, "reflections": [str], "trial1_source_run_id": str}`.
For GSM8K, `em`/`f1` = `correct`.

### 4.3 `results/{run_tag}/prompt_tokens.json`
`{system_prompt_id: token_count}` from `client.messages.count_tokens` (one call per distinct system prompt, done at runner start). Needed by T14.2.

### 4.4 `results/{run_tag}/config.json`
Full snapshot of `config.py` values + git commit hash + CLI args at launch.

---

## 5. TASKS

Sizes: S ≈ <1h, M ≈ 1–3h, L ≈ 3–6h of focused implementation. Dependencies in brackets.

---

### T0 — Prerequisites and environment [none] · S
1. **API key:** create an Anthropic API key; put it in `agent-cost-study/.env` as `ANTHROPIC_API_KEY=...` (add `.env` to `.gitignore`). Load with `python-dotenv`.
2. **Rate limits & credits (blocking check):** in the Anthropic Console → Limits, record the org's RPM / ITPM / OTPM for Haiku 4.5 in `docs/DEVIATIONS.md`. Workload ≈ 60–90k calls total. To finish main runs in ≤ 12 hours of wall clock you need roughly **≥ 150 RPM and ≥ 400k ITPM**. If limits are lower, apply the scope-down ladder (§6.3) *before* launch. Ensure ≥ $300 credit balance.
3. **Python:** system Python is 3.14 (some wheels may be missing). Use `uv`:
   ```bash
   cd agent-cost-study && uv venv --python 3.12 && source .venv/bin/activate
   uv pip install "anthropic>=0.69" httpx pandas pyarrow numpy scipy statsmodels scikit-learn matplotlib tqdm python-dotenv pytest pytest-asyncio
   ```
   Record exact installed versions in `pyproject.toml`.
4. `git init`; first commit = this plan. Commit early and often.

**Accept:** `python -c "import anthropic; print(anthropic.__version__)"` works; one live call `client.messages.create(model="claude-haiku-4-5", max_tokens=16, messages=[{"role":"user","content":"hi"}])` returns usage fields.

---

### T1 — Pre-registration [T0] · S
Copy §2 verbatim into `PREREGISTRATION.md`, add date/time and "frozen before main runs", commit. Commit hash goes in the eventual write-up. Any later change → `docs/DEVIATIONS.md` with date and reason; never edit the pre-registration.

---

### T2 — `config.py` and `pricing.py` [T0] · S
`config.py` holds: `MODEL`, pricing constants (§2.1), `CACHE_MIN_TOKENS = {"claude-haiku-4-5": 4096}`, temperatures, max_tokens per role, `REACT_MAX_STEPS=7`, `PE_MAX_PLAN_STEPS=5`, `PE_MAX_EXEC_STEPS=4`, `REWOO_MAX_EVIDENCE=8`, `REFLEXION_MAX_TRIALS=5`, `REFLEXION_MEMORY=3`, `OBS_MAX_CHARS=1500`, `POOL_K=10`, `R=3`, `N=300`, `PILOT_N=30`, `SEED=2026`, `BOOT_B=2000`, concurrency defaults (`MAX_RUNS_IN_FLIGHT=32`, `LLM_CONCURRENCY=24`, `WIKI_CONCURRENCY=8`), `MAX_USD` (default 250).
`pricing.py`: `usd_nocache(call)`, `usd_empirical(call)`, `usd_ideal(call_rows, L)` (the last one implemented in T14.2; here only a stub signature).

**Accept:** unit test: a call with I = 10,000 (all `input_tokens`) and O = 1,000 → `usd_nocache = 0.015`; with `cache_read=8000, input_tokens=2000` → `usd_empirical = (2000×1 + 8000×0.1 + 1000×5)/1e6 = 0.0078`.

---

### T3 — LLM wrapper `llm.py` [T2] · M  ← most important file
**Interface:**
```python
class LLM:
    async def call(self, *, ctx: "RunContext", role: str, conversation_id: str, turn_index: int,
                   system_prompt_id: str, system: str, messages: list[dict],
                   max_tokens: int, temperature: float, stop: list[str] | None = None) -> "LLMResult"
# LLMResult: text:str, stop_reason:str, usage:dict, call_id:str
```
**Behaviour (all mandatory):**
1. Uses `anthropic.AsyncAnthropic(max_retries=8, timeout=120)`. A global `asyncio.Semaphore(LLM_CONCURRENCY)` wraps every request.
2. **Cache breakpoints (identical for every scaffold):** send `system` as `[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}]`. Deep-copy `messages`; convert the **last** message's content to a list of blocks `[{"type":"text","text":..., "cache_control":{"type":"ephemeral"}}]`. **No other message may carry `cache_control`** (the API allows max 4 breakpoints; never accumulate them). Earlier messages stay plain strings.
3. Messages must alternate user/assistant, start with user, end with user. Never send an assistant prefill. If an assistant text to be appended is empty, the caller substitutes `"(no output)"` (API rejects empty content).
4. `stop_sequences=stop` when given.
5. After the response: `text = "".join(b.text for b in resp.content if b.type=="text")`; `usage` from `resp.usage` with `cache_creation_input_tokens or 0`, `cache_read_input_tokens or 0`.
6. Append one line to `calls.jsonl` (§4.1) under an `asyncio.Lock`, flush immediately. On final failure after retries, log a row with `error` set and re-raise.
7. **Spend guard:** maintain a running `usd_nocache` total across the process (initialised from existing `calls.jsonl` on resume). If it exceeds `MAX_USD`, set a global `STOP` flag; the runner stops launching new runs (in-flight runs finish).
8. `FakeLLM` with the same interface: returns scripted responses from a list or a callable `(role, messages) -> str`, fabricates usage (`input_tokens = len(all text)//4`, `output_tokens = len(reply)//4`), writes the same log rows. **All scaffold tests use FakeLLM.**

**Accept (tests/test_llm.py):** FakeLLM round-trip writes a correct row; request builder puts `cache_control` exactly on the system block and the last message's last block (assert over a 5-turn conversation); spend guard trips; one live smoke call (network mark) logs non-null usage.

---

### T4 — Data and grading [T0] · M
`data.py`:
- `download()` fetches the three parquet files (§2.2) into `data/raw/` (skip if present). The HF API URL returns a JSON list of real parquet URLs — fetch that list, then download the file(s).
- `make_samples()` implements §2.2 exactly and writes `data/samples/hotpotqa_N300.json` and `gsm8k_N300.json` as lists of `{"task_id","question","answer", "type"(hotpot only)}`. HotpotQA `task_id` = dataset `id`; GSM8K `task_id` = `f"gsm8k_test_{row_index}"`. Refuse to overwrite an existing sample file.
- `load_tasks(benchmark, pilot: bool) -> list[Task]`.

`grading.py`: `normalize_answer`, `hotpot_em(pred, gold)`, `hotpot_f1(pred, gold)` (official code incl. yes/no/noanswer rule), `gsm8k_gold(answer_field)`, `gsm8k_pred(text)`, `gsm8k_correct(pred_text, gold)`, `grade(benchmark, pred, gold) -> dict(em, f1, correct)`.

**Accept:** `hotpot_em("The Saimaa Gesture", "Saimaa Gesture")==1`; `hotpot_f1("yes","no")==0`; `gsm8k_pred("The answer is $1,234.")==1234`; `gsm8k_correct("72", "…#### 72")`; sample files have 300 unique IDs, HotpotQA type proportions within ±2% of the split.

---

### T5 — Tools [T2] · M
**`tools/wiki.py`** — Wikipedia via the MediaWiki Action API (`https://en.wikipedia.org/w/api.php`), `httpx.AsyncClient`, header `User-Agent: agent-cost-study/0.1 (academic research; contact: <your email>)`, semaphore `WIKI_CONCURRENCY`, 3 retries with backoff on 429/5xx.

`async search(entity) -> (observation: str, page_text: str|None)` — mirror ReAct's `wikienv.search_step`:
1. Exact page attempt: `action=query&titles={entity}&redirects=1&prop=extracts&explaintext=1&format=json&formatversion=2`. If the page exists (no `missing`) and the extract is non-empty → candidate page.
2. If no candidate, `action=query&list=search&srsearch={entity}&srlimit=5&format=json&formatversion=2`. If the top hit's title equals `entity` case-insensitively, fetch it as in step 1 → candidate page. Otherwise return `f"Could not find {entity}. Similar: {titles}."` where `titles` is the Python list repr of up to 5 titles; `page_text=None`.
3. Disambiguation: if the candidate's text contains `"may refer to:"` in its first 500 chars, return the "Could not find … Similar: …" observation built from step 2's search results.
4. Page text cleaning: drop lines that are section headers (`^=+.*=+$`), drop empty lines, keep paragraphs with > 2 words (as `wikienv` does).
5. Observation = first 5 sentences, computed exactly like `wikienv.get_page_obs`: split paragraphs on `\n`, sentences on `'. '`, re-append `'.'`, join first 5 with spaces.

`lookup(page_text, keyword, state) -> observation` — mirror `wikienv` lookup: sentences (same splitting) containing `keyword` case-insensitively; per-run state holds `(keyword, results, cnt)`; new keyword resets `cnt=0`; returns `f"(Result {cnt+1} / {len(results)}) {sentence}"` then increments; when exhausted returns `"No more results."`; with no page loaded returns `"No page loaded. Use Search first."`.

**Disk cache:** `cache/wiki.sqlite`, table `kv(key TEXT PRIMARY KEY, value TEXT, ts TEXT)`; key = `sha256(json.dumps({"endpoint":…, "params":…}, sort_keys=True))`. Access via `asyncio.to_thread` + a `threading.Lock`. **The cache is keyed on the raw API request**, so identical queries return identical content for every scaffold for the whole study. Never clear it between pilot and main.

**`tools/calc.py`** — `calculate(expr) -> str`: strip `$`, `,`, whitespace; a trailing `%` means "/100"; parse with `ast.parse(mode="eval")` allowing only `Constant(int/float)`, `BinOp(Add, Sub, Mult, Div, FloorDiv, Mod, Pow)`, `UnaryOp(USub, UAdd)`; exponent capped at 100 and |result| capped at 1e15; format: if `abs(x - round(x)) < 1e-9` → `str(int(round(x)))`, else `f"{x:.6f}".rstrip("0").rstrip(".")`. Any error → `"Error: invalid expression"`.

**`tools/toolbox.py`** — `ToolBox(benchmark, llm_for_worker=None)` created **once per scaffold run**; holds Lookup state; `async run(tool_name, arg) -> str` dispatches `Search`, `Lookup`, `Calculate`, `LLM`; unknown tool → `f"Error: unknown tool {tool_name}. Valid tools: [...]"`; truncates every observation to `OBS_MAX_CHARS`; appends a `{"kind":"tool",...}` trace entry; increments `n_tool_calls[tool_name]`. Tool availability per benchmark per §2.3 (e.g. calling `Search` on GSM8K → unknown-tool error).

**Accept:** calc tests reproduce the GSM8K exemplar observations exactly (`48 / 2`→`24`, `48 + 24`→`72`, `12 / 60 * 50`→`10`, `100 - 100 / 2 - 15 - 15 * 2`→`5`); `__import__('os')` → error. Network tests: `Search[Colorado orogeny]` begins with "The Colorado orogeny"; a nonsense query yields "Could not find"; second identical call makes 0 HTTP requests (cache hit); Lookup cycles `(Result 1 / n)`, `(Result 2 / n)`, … then "No more results.".

---

### T6 — Prompts [T5] · M
**Vendor (verbatim, read-only, with a provenance header noting repo, commit SHA and path):**

| File in `prompts/vendor/` | Source (pinned commit) |
|---|---|
| `react_prompts_naive.json` | `ysymyth/ReAct@6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9 : prompts/prompts_naive.json` |
| `reflexion_prompts.py`, `reflexion_fewshots.py` | `noahshinn/reflexion@218cf0ef1df84b05ce379dd4a8e47f17766733a0 : hotpotqa_runs/{prompts,fewshots}.py` |
| `rewoo_fewshots.py`, `rewoo_planner.py`, `rewoo_solver.py` | `billxbf/ReWOO@9cd0283043ff4be0c9d614fda2789d143ca6ffd1 : prompts/{fewshots,planner,solver}.py` |

Raw URL pattern: `https://raw.githubusercontent.com/{repo}/{sha}/{path}`. Do **not** import the vendored `.py` files (they import langchain). Extract strings with regex or `ast` at build time in `library.py`.

**`library.py` must expose these strings (each with a stable `system_prompt_id`):**

**A. `ANSWER_STYLE`** (appended to every prompt that produces a final answer, all scaffolds):
> `Your final answer must be as short as possible: a name, entity, date, number, or yes/no — no explanation.`
For GSM8K replace with: `Your final answer must be a single number only.`

**B. ReAct HotpotQA system** (`react_hotpotqa_v1`) = ReAct instruction + `webthink_simple6` + ANSWER_STYLE, where the instruction is exactly:
> `Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:\n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\n(3) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n`
followed by `webthink_simple6` from the vendored JSON (numbered format `Thought 1: / Action 1: / Observation 1:`), then `\n(END OF EXAMPLES)\n` and ANSWER_STYLE.

**C. ReAct GSM8K system** (`react_gsm8k_v1`) — instruction:
> `Solve a math word problem with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:\n(1) Calculate[expression], which evaluates an arithmetic expression (numbers, + - * / // % ** and parentheses) and returns the result.\n(2) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n`
Exemplars — these three, **verbatim** (GSM8K train items 0, 1, 2; observations are real `calculate()` outputs, enforced by a test):
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Thought 1: In May she sold half of 48 clips, so I need 48 / 2.
Action 1: Calculate[48 / 2]
Observation 1: 24
Thought 2: She sold 24 clips in May. The total is 48 + 24.
Action 2: Calculate[48 + 24]
Observation 2: 72
Thought 3: Natalia sold 72 clips altogether.
Action 3: Finish[72]

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Thought 1: She earns 12 dollars per 60 minutes, so for 50 minutes she earns 12 / 60 * 50.
Action 1: Calculate[12 / 60 * 50]
Observation 1: 10
Thought 2: She earned 10 dollars.
Action 2: Finish[10]

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Thought 1: Betty has 100 / 2. Her parents give 15 and her grandparents give 15 * 2. She still needs 100 - 100 / 2 - 15 - 15 * 2.
Action 1: Calculate[100 - 100 / 2 - 15 - 15 * 2]
Observation 1: 5
Thought 2: Betty needs 5 more dollars.
Action 2: Finish[5]
```
then `\n(END OF EXAMPLES)\n` + GSM8K ANSWER_STYLE.

**D. ReWOO planner** (`rewoo_planner_{bench}_v1`) = vendored `DEFAULT_PREFIX` + tool list + few-shot + vendored `DEFAULT_SUFFIX`.
- Tool list (format `Tools can be one of the following:\n{Name}[input]: {description}\n…`):
  - HotpotQA: `Search[input]: Searches the exact entity on Wikipedia and returns the first paragraph if it exists; otherwise returns similar entities. Input should be an entity name.` / `Lookup[input]: Returns the next sentence containing the keyword in the page most recently found by Search. Input should be a keyword.` / `LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.` (last line = ReWOO's own LLMWorker description).
  - GSM8K: `Calculate[input]: Evaluates an arithmetic expression (numbers, + - * / // % ** and parentheses). Input should be a mathematical expression.` / `LLM[input]: …` (as above).
- Few-shot HotpotQA: vendored `HOTPOTQA_PWS_BASE` with every `Wikipedia[` replaced by `Search[` (the only change; log it in DEVIATIONS).
- Few-shot GSM8K (verbatim):
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Plan: Compute the number of clips sold in May, which is half of the April sales.
#E1 = Calculate[48 / 2]
Plan: Add the April and May sales.
#E2 = Calculate[48 + #E1]

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Plan: Compute the earnings for 50 minutes at 12 dollars per 60 minutes.
#E1 = Calculate[12 / 60 * 50]

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Plan: Compute how much money Betty already has.
#E1 = Calculate[100 / 2]
Plan: Compute the grandparents' gift, which is twice the parents' 15 dollars.
#E2 = Calculate[15 * 2]
Plan: Compute how much more money Betty needs.
#E3 = Calculate[100 - #E1 - 15 - #E2]
```
- User message: `Question: {question}`.

**E. Shared solver** (`solver_{bench}_v1`, used by **both ReWOO and Plan-Exec** so that the final synthesis step is identical and only planning/execution differs): system = vendored solver `DEFAULT_PREFIX` + ANSWER_STYLE. User message = for each step i: `Plan: {plan_i}\nEvidence: {evidence_i}\n` (ReWOO: evidence = `#Ei` value; Plan-Exec: evidence = step result) + vendored `DEFAULT_SUFFIX` + `{question}`. (The vendored suffix already says "Respond with the answer directly with no extra words.")

**F. ReWOO LLM worker** (`worker_llm_v1`): system `You are a helpful assistant. Answer the request concisely, using the given context when provided.`; user = substituted input.

**G. Plan-Exec planner** (`pe_planner_{bench}_v1`), authored after Plan-and-Solve ("Let's first understand the problem and devise a plan…"):
> `Let's first understand the problem and devise a plan to solve it. Write a short step-by-step plan of at most 5 steps. Each step must be a sub-task that can be completed with these tools: {tool names with one-line descriptions}. Do not solve the steps yourself. Output only the numbered plan, one step per line, in the form "1. …".\n\nHere are some examples.\n{examples}\n(END OF EXAMPLES)`
- HotpotQA examples (verbatim):
```
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Plan:
1. Find the area that the eastern sector of the Colorado orogeny extends into.
2. Find the elevation range of that area.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Plan:
1. Find who Matt Groening named the character Milhouse after.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Plan:
1. Find what the documentary Adam Clayton Powell is about.
2. Find what The Saimaa Gesture is about.
3. Decide which of the two is about Finnish rock groups.
```
- GSM8K examples (verbatim):
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Plan:
1. Compute the number of clips sold in May, which is half of 48.
2. Add the April and May sales.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Plan:
1. Compute the earnings for 50 minutes at 12 dollars per hour.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Plan:
1. Compute how much money Betty already has.
2. Compute the total gifts from her parents and grandparents.
3. Compute how much more money Betty needs.
```
- User message: `Question: {question}\nPlan:`

**H. Plan-Exec executor** (`pe_executor_{bench}_v1`) = the ReAct system prompt for that benchmark (B or C) **minus ANSWER_STYLE**, plus:
> `\nYou are executing ONE step of a larger plan. Use Finish[result] to return the result of the CURRENT step only (a short fact or number), not necessarily the final answer to the question.`

**I. Reflector** (`reflector_{bench}_v1`): system = vendored `REFLECT_INSTRUCTION` split at `Previous trial:` — the part before it (with `{examples}` filled by vendored `REFLECTIONS` for HotpotQA; for GSM8K replace "an Docstore API environment" with "a calculator tool" and fill `{examples}` with the empty string). User = `Previous trial:\nQuestion: {question}\n{scratchpad}\n\nReflection:`. Scratchpad = the trial's actor transcript (`Thought/Action/Observation` lines); if > 6,000 chars, truncate the longest `Observation` lines to 200 chars each until ≤ 6,000 (simplified version of Reflexion's `truncate_scratchpad`).

**J. Reflection header** for trials ≥ 2: vendored `REFLECTION_HEADER` + `'Reflections:\n- ' + '\n- '.join(last 3 reflections)` (exact `format_reflections` logic).

**K. Judge** (`judge_{bench}_v1`, used only by `reflexion_self`):
> System: `You are a strict verifier. You will see a question, an agent's reasoning trajectory including tool observations, and the agent's final answer. You do NOT know the correct answer. Decide whether the final answer is correct, judging only from the question, the evidence in the observations, and your own knowledge. Write one sentence of justification, then on the last line write exactly "VERDICT: CORRECT" or "VERDICT: INCORRECT".`
> User: `Question: {question}\n\nTrajectory:\n{scratchpad}\n\nFinal answer: {answer}`
Parse: the last match of `VERDICT:\s*(CORRECT|INCORRECT)`; missing → INCORRECT and `judge_parse_fail += 1`.

**Accept:** a test builds every prompt for both benchmarks; asserts the vendored substrings are present, that no `{placeholder}` braces remain, and that the GSM8K ReAct/ReWOO exemplars' Calculate observations equal `calculate()` outputs. `count_tokens` for each system prompt is written to `prompt_tokens.json` at runner start.

---

### T7 — Shared base + ReAct [T3, T5, T6] · L
**`scaffolds/base.py`:**
- `RunContext(run_id, benchmark, task, scaffold, replicate, llm, toolbox, trace, counters)`.
- `RunResult` → serialises to §4.2.
- `parse_action(text) -> (tool, arg) | None`: search the **first** match of `r"Action\s*\d*\s*:\s*(\w+)\[(.*)\]"` with `re.S`. For the arg, take everything between the first `[` after the tool name and the **last** `]` on that action line (answers may contain brackets). Tool name normalised to Title case.
- `truncate_after_first_action(text)`: keep the text up to and including the first `Action…]` line; drop anything after (the model sometimes writes several steps). **The truncated text is what gets appended to the conversation** (append-only is preserved because we append exactly what we store).

**`scaffolds/react.py` — `react_loop(ctx, system, system_prompt_id, first_user_msg, conversation_label, max_steps, role, finish_is_final=True) -> (answer|None, transcript_str, halted)`:**
```
messages = [{"role":"user","content": first_user_msg}]         # e.g. "Question: …"
for i in 1..max_steps:
    res = llm.call(role=role, conversation_id=f"{ctx.run_id}:{label}", turn_index=i-1,
                   system=system, messages=messages, max_tokens=512, temperature=0.7,
                   stop=["\nObservation", "\nQuestion:"])
    text = truncate_after_first_action(res.text).strip() or "(no output)"
    messages.append({"role":"assistant","content": text})
    act = parse_action(text)
    if act is None:           obs = "Invalid action. Valid actions are " + valid_actions_str
    elif act.tool == "Finish": return (act.arg.strip(), transcript, halted=False)
    else:                      obs = await ctx.toolbox.run(act.tool, act.arg)
    messages.append({"role":"user","content": f"Observation {i}: {obs}"})
return (None, transcript, halted=True)
```
- `transcript` = the concatenation of assistant texts and `Observation i:` lines (used by the reflector and judge).
- If the model emits a Thought without an Action, the "Invalid action" observation counts as a step (as in ReAct's loop).
- **`react` scaffold** = `react_loop` with `system = react_{bench}_v1`, `first_user_msg = f"Question: {q}"`, `max_steps = 7`; final answer = `answer or ""`.

**Accept (FakeLLM):** a scripted 3-step episode yields the right answer, 2 tool calls, 3 LLM log rows with turn_index 0,1,2 and one conversation_id; an invalid action produces the error observation; 7 steps without Finish → `halted=True`, answer `""`; `Finish[Arthur's Magazine [1844]]` → `Arthur's Magazine [1844]`; multi-action output is truncated to the first action.

---

### T8 — ReWOO [T7] · M
1. **Planner:** one call, role `planner`, conversation `…:plan`, system `rewoo_planner_{bench}_v1`, user `Question: {q}`, max_tokens 1024, no stop sequence.
2. **Parse:** iterate lines; a line starting with `Plan:` sets the current plan text; a line matching `^\s*#E(\d+)\s*=\s*(\w+)\[(.*)\]\s*$` defines evidence k with (tool, input) paired to the most recent Plan text. Keep at most `REWOO_MAX_EVIDENCE = 8`, in order. `plan_parse_ok = (n_evidence ≥ 1)`.
3. **Workers** (sequential, in order): substitute every `#Ej` (j < k, **longest-first** so `#E10` isn't clobbered by `#E1`) in the input with evidence j's string (raw substitution, as in the original). Tool `LLM` → one call, role `worker_llm`, fresh conversation `…:w{k}`, system `worker_llm_v1`, max_tokens 256. Other tools → `toolbox.run`. Unknown tool → the toolbox error string becomes the evidence.
4. **Solver:** one call, role `solver`, conversation `…:solve`, system `solver_{bench}_v1`, user built per T6-E from (plan_i, evidence_i). If `plan_parse_ok` is False, the solver is called with zero plan/evidence pairs (it answers from its own knowledge). Final answer = solver text stripped; if it contains `Finish[…]`, take the bracket content.

**Accept (FakeLLM):** the vendored HotpotQA few-shot's first example, fed back as a planner output, parses to 4 evidence steps with the right tools; `#E2` substitution works; `#E10` vs `#E1` substitution is correct; a planner output without `#E` lines → `plan_parse_ok=False` and the solver is still called.

---

### T9 — Plan-and-Execute [T7] · M
1. **Planner:** one call, role `planner`, system `pe_planner_{bench}_v1`, user `Question: {q}\nPlan:`, max_tokens 1024. Parse lines matching `^\s*(\d+)[.)]\s+(.+)$` in order; keep the first `PE_MAX_PLAN_STEPS = 5`. If none → a single step `"Answer the question."` and `plan_parse_ok=False`.
2. **Executor:** for step j (1-based): `react_loop` with system `pe_executor_{bench}_v1`, `conversation_label=f"step{j}"`, `role="executor"`, `max_steps=PE_MAX_EXEC_STEPS (4)`, and first user message:
   ```
   Question: {q}
   Plan:
   {numbered plan}
   Results of completed steps:
   {"Step i result: r_i" lines, or "(none)"}
   Current step {j}: {step_j}
   ```
   Step result = Finish arg, or `"No result."` if halted. **The toolbox (and so Lookup state) is shared across all steps of the run.** No replanning.
3. **Solver:** the shared solver (T6-E) with pairs (step_j, result_j).

**Accept (FakeLLM):** a 2-step plan triggers 2 executor conversations with distinct conversation_ids and turn_index restarting at 0; the "completed steps" block contains step 1's result when running step 2; a plan with 8 lines is capped at 5.

---

### T10 — Reflexion (GT and Self) [T7, T11 partly, see note] · L
`reflexion_run(ctx, mode ∈ {"gt","self"})`:
```
trial1 = load pool run  f"{bench}:{task_id}:react:{replicate}"  from runs.jsonl   (must exist; else raise → run marked error, retried later)
copy its trace, calls (by reference: record trial1_source_run_id), answer, halted, transcript
reflections = []
for t in 1..REFLEXION_MAX_TRIALS:
    if t > 1:
        header = format_reflections(reflections[-3:])                     # T6-J
        answer, transcript, halted = react_loop(system=react_{bench}_v1,
             first_user_msg = header + "\n\nQuestion: " + q,
             conversation_label=f"trial{t}", role="actor", max_steps=7)
    # --- stopping signal (the ONLY difference between modes) ---
    if answer is None or answer == "":  success = False; verdict = None     # no judge call for empty answers
    elif mode == "gt":                  success = grade(bench, answer, gold)["correct"] == 1
    else:                               verdict = judge(q, transcript, answer); success = (verdict == "CORRECT")
    record trial answer, graded correctness (always, for analysis), verdict
    if success or t == REFLEXION_MAX_TRIALS: break
    reflection = reflector(q, transcript)       # role "reflector", conversation f"…:refl{t}", 512 max tokens
    reflections.append(reflection.strip())
final_answer = answer of the LAST trial run (original Reflexion semantics), "" if None
```
- The reflector receives **only** the binary failure (implicit in its prompt: "You were unsuccessful…") — **never** the judge's justification and never the gold answer. This keeps the two modes identical except for the source of the stopping bit.
- **Cost accounting:** the run's tokens and USD = trial 1's calls (summed from the source pool run's call rows) + all new calls. In `calls.jsonl` only new calls carry this run_id; T14 adds the trial-1 calls via `trial1_source_run_id`. Store `trial1_usd_nocache`, `trial1_tokens_in`, `trial1_tokens_out` in `extra` for convenience.
- Judge calls: role `judge`, conversation `…:judge{t}`, temperature 0.0, max_tokens 256.

**Accept (FakeLLM):** gt mode stops at the first gold-correct trial; self mode stops when the fake judge says CORRECT even if the answer is wrong (and continues when it says INCORRECT even if right); the reflector's prompt never contains the gold string or judge text (assert); `trial_correct` is always filled; an empty answer triggers no judge call; memory holds at most 3 reflections.

---

### T11 — Runner `runner.py` [T3–T10] · M
CLI:
```
python -m acs.runner --run-tag pilot|main --benchmarks hotpotqa gsm8k \
       --stages pool others --pilot/--no-pilot [--max-usd 250] [--k 10] [--r 3]
```
- **Stage `pool`:** for every task × r in 0..K−1 → `react`. **Stage `others`:** for every task × r in 0..R−1 → `rewoo`, `plan_exec`, `reflexion_gt`, `reflexion_self`. `others` requires the pool runs r < R to exist (Reflexion trial 1).
- `run_id = f"{benchmark}:{task_id}:{scaffold}:{replicate}"`. **Resume:** at start, read `runs.jsonl`; skip run_ids with `status=="ok"`; re-run `error` ones.
- Concurrency: `asyncio` with a `Semaphore(MAX_RUNS_IN_FLIGHT)` over runs; interleave scaffolds (shuffle the work list with `rng(2026)`) so no scaffold systematically gets a warmer cache or a different time of day.
- Each run: new `ToolBox`, new trace; try/except → `status="error"` with the traceback string (never crash the whole process).
- At start: write `config.json`; compute `prompt_tokens.json` via `count_tokens` for every system prompt ID.
- Progress: tqdm per stage + a line every 50 runs with cumulative `usd_nocache`, runs/min, and error count. Respect the spend-guard `STOP` flag.
- Grading happens inside the run (after the scaffold returns) and is written to `runs.jsonl`.

**Accept:** a FakeLLM end-to-end run on 3 tasks × all scaffolds writes consistent `runs.jsonl` / `calls.jsonl` (every call's run_id exists; per-run token sums match the sum of that run's calls; Reflexion runs reference existing pool runs); killing mid-run and restarting completes only the missing runs.

---

### T12 — Pilot and go/no-go [T11] · M
Run: `--run-tag pilot --pilot --k 3 --r 1` on both benchmarks (30 tasks each). Then produce `results/pilot/PILOT_REPORT.md` containing, per scaffold × benchmark:
- accuracy, mean tokens in/out, mean `usd_nocache`, mean LLM calls, error rate;
- parse health: ReAct invalid-action rate, halt rate; ReWOO `plan_parse_ok` rate; Plan-Exec parse rate; judge parse-fail rate; mean Reflexion trials;
- empirical cache-read share (Σcache_read / ΣI) — expected ≈ 0 on most Haiku calls (4,096 threshold);
- **projected main-run cost** = pilot mean usd per run × planned main run counts (N·K for the pool, N·R·4 for the others);
- **projected wall clock** from observed runs/min.

**Go/no-go rules (write the decision into DEVIATIONS.md):**
- Any error rate > 2% or ReAct invalid-action rate > 10% → fix prompts/parsers first (a prompt change after the pilot is allowed but must be logged; after main runs start, prompts are frozen).
- Projected cost > MAX_USD or projected time > available time → apply §6.3.
- Sanity: ReAct HotpotQA EM should be roughly 25–50% and ReWOO should use far fewer calls than ReAct; if not, inspect traces before continuing.

---

### T13 — Main runs [T12, T1 committed] · (wall-clock)
`--run-tag main --no-pilot --k 10 --r 3`, stage `pool` then `others`, both benchmarks. Monitor the progress log and the error count; re-invoke the same command to resume after any crash. Do **not** change code or prompts during T13 except for crash fixes that do not alter model inputs (log any fix).

---

### T14 — Analysis `analysis/` [T13; can be developed on pilot data in parallel] · L

**T14.1 Load.** `load.py` → `calls_df`, `runs_df`. Attach Reflexion trial-1 calls (via `trial1_source_run_id`) to their Reflexion runs for all cost computations. Drop runs with `status != "ok"` and report how many were dropped per scaffold (must be < 2%; otherwise list them).

**T14.2 Ideal-cache counterfactual `usd_ideal[L]`** (deterministic; ignores TTL, i.e. assumes a warm steady state):
For each call c (ordered by `conversation_id`, `turn_index`), let `I_c` be its total input, `S_c = prompt_tokens[system_prompt_id]`.
```
read_c = 0
prev = [calls in same conversation with turn_index < c.turn_index and I_prev >= L]
if prev: read_c = max(I_prev)                                   # entry written at previous call's last block
elif S_c >= L and the system prompt was seen before in this study (any earlier call, any run):
    read_c = S_c                                                # system-block breakpoint shared across tasks
write_c  = (I_c - read_c) if I_c >= L else 0                    # this call's last-block breakpoint writes
uncached = I_c - read_c - write_c
usd_ideal = (uncached*1.00 + write_c*1.25 + read_c*0.10 + O_c*5.00) / 1e6
```
"Seen before" = any call with the same `system_prompt_id` earlier in the `ts` ordering (the very first call per system prompt pays the write). Unit tests: a 3-turn conversation with I = 5000, 5600, 6200 at L = 4096 → reads 0, 5000, 5600; at L = 8000 → no caching; a system prompt of 5,000 tokens with L = 4096 → second task's first call reads 5,000.

**T14.3 vote@k (the budget-matched control).** For each task, the pool has 10 samples with (answer, cost). For k in 1..10 and B = 200 draws (`rng(2026)`): draw k distinct samples without replacement; majority vote over `normalize_answer(answer)` (HotpotQA) or the parsed number (GSM8K), ignoring empty answers; **tie-break = the tied answer from the earliest-drawn sample**; all empty → `""`. Correctness of the voted answer by the standard grader (for HotpotQA, EM of the original answer string of the winning group's first member). Cost = sum of the k samples' costs (every regime, and tokens). Per task: expected accuracy and expected cost over draws. Aggregate → the vote@k curve (accuracy vs mean cost), one curve per cost regime.
**k*(X)**: the (fractional) k at which the vote@k mean `I+O` tokens equals X's mean tokens, by linear interpolation between integer k; accuracy at k* interpolated likewise. If X's cost exceeds vote@10's, report `k* > 10`, use vote@10 as a conservative lower bound on the control, and flag it.
Also compute k* per cost regime for the Pareto figures (the matching in H1 is on tokens; the dollar-matched version is secondary).

**T14.4 Stats.** `stats.py`: `cluster_bootstrap(task_ids, fn, B, rng)`, paired versions, `mcnemar_exact(b, c)` (binomial test on discordant pairs), `holm(pvals)`. For bootstrap-based p-values use the two-sided percentile method: `p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))`.

**T14.5 Metrics table** per benchmark × scaffold (incl. react_vote@k for k ∈ {1,3,5,10} and the vote@k*(X) points): Acc (EM) [95% CI], F1, tokens in/out/total, LLM calls, tool calls, usd in 3 regimes (ideal at L = 0, 1024, 4096), cost-of-pass per regime [95% CI], wall_s, halt/parse-failure rates.

**T14.6 RQ4 predictor.** Units: tasks. Features (all from the **first attempt only** = pool sample r = 0, plus agreement from samples 0–2):
`n_steps, n_search, n_lookup, n_calc, n_invalid_actions, halted, n_could_not_find, answer_len_chars, trial1_tokens, distinct_answers_in_first3 (1–3), benchmark (one-hot)`.
Target `y = 1` iff pool sample 0 is incorrect **and** `reflexion_self` (replicate 0) is correct. Model: `LogisticRegression(C=1.0, max_iter=1000)` with standardised features, 5-fold stratified CV (`random_state=2026`), both benchmarks pooled. Report CV AUC with a bootstrap CI vs. a baseline using only the benchmark one-hot (H4). **Routing policy evaluation:** on held-out folds, route a task to Reflexion-Self if `p > τ` (τ ∈ {0.2, 0.3, 0.5}) else keep the ReAct answer; the cost of routing = ReAct cost + (Reflexion cost − trial-1 cost) for routed tasks, plus the 2 extra pool samples if the agreement feature is used (**include that cost**, or re-fit without the agreement feature and report both). Plot the policies as points on the Pareto figure.

**T14.7 Figures and tables** (`report.py` → `results/main/analysis/`):
- **Fig 1 (headline):** per benchmark, x = mean `usd_nocache` per task (log scale), y = EM; the vote@k curve (k = 1..10) as a line with a CI band; each scaffold as a point with 2-D error bars; Pareto frontier highlighted.
- **Fig 2 (RQ2):** Reflexion-GT vs Self: accuracy, cost, distribution of trials; plus the **judge confusion matrix** (verdict vs gold correctness) across all judged trials: false-accept rate and false-reject rate.
- **Fig 3 (H3):** for each scaffold, cost relative to ReAct under: tokens (I+O), usd_nocache, usd_ideal[0], usd_ideal[1024], usd_ideal[4096], usd_empirical — grouped bars with bootstrap CIs.
- **Fig 4:** cumulative accuracy vs. token budget (cheapest-first ordering, Simhadri-style) — secondary.
- **Fig 5:** RQ4 ROC curves plus the routing policies on the Pareto plane.
- **Table 1:** the T14.5 metrics. **Table 2:** the 9 pre-registered comparisons with Δ, 95% CI, bootstrap p, Holm-adjusted p, McNemar p. **Table 3:** H3 cost ratios. **Table 4:** failure modes (ReWOO plan-parse failures, halts, judge errors).

**Accept:** every number in the tables is reproducible by `python -m acs.analysis.report --run-tag main` from the JSONL files alone; running it on pilot data works too.

---

### T15 — `RESULTS.md` [T14] · M
One page per RQ: the claim, the number with its CI, the figure reference, and whether the pre-registered hypothesis was supported. Then a "Threats to validity" section (copy §8 and add anything observed). Then "Deviations" (from `docs/DEVIATIONS.md`). Include the pre-registration commit hash.

---

## 6. Run protocol

### 6.1 Order (critical path)
`T0 → T1 → T2 → T3 → (T4 ∥ T5) → T6 → T7 → (T8 ∥ T9 ∥ T10) → T11 → T12 → T13 → T14 → T15`
T14 code can (and should) be written against pilot outputs while T13 runs.

### 6.2 Expected volume and cost (verify in T12)
| | runs | ~LLM calls |
|---|---|---|
| pool (2 benchmarks × 300 × 10) | 6,000 | ~30k |
| others (2 × 300 × 3 × 4) | 7,200 | ~40k |
Rough spend at Haiku 4.5 prices, uncached: **~$150–250** (HotpotQA dominates). The T12 projection replaces this estimate.

### 6.3 Scope-down ladder (apply in order, only as far as needed; log each step)
1. GSM8K pool K: 10 → 6.
2. N: 300 → 200 on both benchmarks (keep the first 200 IDs of the frozen sample).
3. R: 3 → 2 for `rewoo` and `plan_exec` only.
4. HotpotQA pool K: 10 → 6.
**Never** drop: the Reflexion-GT/Self pair, R ≥ 2 for the Reflexion modes, pool K ≥ 6, or either benchmark.

---

## 7. Testing summary (all under `tests/`, run `pytest -m "not network"` in CI-style and `pytest -m network` once)
graders · calculator (incl. exemplar observations) · wiki search/lookup/cache (network) · LLM request builder (cache_control placement, alternation) · spend guard · react_loop (finish, invalid action, halt, truncation, bracket answers) · ReWOO parse + substitution (`#E10`) · Plan-Exec parse + executor context · Reflexion modes (stop logic, judge isolation, reflector isolation, memory cap, trial-1 reuse) · runner resume + log consistency · ideal-cache model · vote@k tie-breaks · bootstrap/Holm sanity.

---

## 8. Threats to validity and documented design decisions (state these in the write-up)
1. **Harmonised tools.** ReWOO's original Wikipedia worker returned longer page content; here every scaffold gets ReAct's 5-sentence `Search` + `Lookup`. This isolates control flow, at the cost of fidelity to ReWOO's original tool. ReWOO keeps its `LLM[]` worker (part of its paradigm).
2. **Plan-and-Execute has no canonical implementation**; ours = Plan-and-Solve-style planner + ReAct executor + the same solver as ReWOO, no replanning (matches Zhou et al. 2026's "no replanning" variant).
3. **Reflexion trial 1 = a ReAct pool sample** — identical distribution by construction; it makes comparisons paired.
4. **Uniform ANSWER_STYLE line** added to all answer-producing prompts (modern chat models are verbose; the original prompts were written for completion models). Identical across scaffolds.
5. **Temperature 0.7** everywhere (originals mostly used greedy decoding) — needed for independent replicates and vote@k.
6. **String-majority voting** for HotpotQA vote@k (no LLM aggregator) — conservative for the control, since semantically equivalent answers split votes. If the control still matches or beats a scaffold, the conclusion is strengthened.
7. **EM grading under-credits paraphrases** equally for all scaffolds; F1 is reported as a check.
8. **Live Wikipedia** content differs from 2018 HotpotQA-era content; the disk cache makes it identical *across scaffolds*, which is what the comparison requires.
9. **Empirical caching depends on the run schedule** (TTL, concurrency); hence `usd_ideal[L]` as the schedule-independent primary caching measure.
10. **Single model** (Haiku 4.5); generalisation is future work (the config supports a second model).

---

## 9. Definition of done for the sprint
- [ ] `PREREGISTRATION.md` committed before T13 started (commit hash recorded).
- [ ] `results/main/{calls,runs}.jsonl` complete, errors < 2% per scaffold.
- [ ] Figures 1–3 and Tables 1–3 generated by a single command.
- [ ] `RESULTS.md` answers RQ1–RQ3 with CIs; RQ4 reported (positive or null).
- [ ] `docs/DEVIATIONS.md` lists every change from this plan.
