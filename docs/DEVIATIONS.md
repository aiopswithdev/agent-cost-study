# DEVIATIONS from `docs/IMPLEMENTATION_PLAN.md`

Every change from the plan (and from `PREREGISTRATION.md`) is logged here with its date and reason.
The pre-registration itself is never edited.

---

## 2026-09-29 — T6: ReWOO HotpotQA few-shot tool rename

Plan §T6-D mandates it, logged here as required: the vendored ReWOO few-shot
`HOTPOTQA_PWS_BASE` uses the tool name `Wikipedia[...]`. Every occurrence of `Wikipedia[`
is replaced by `Search[` so the few-shot names the tool this study actually exposes
(§2.3 harmonised tools). This is the **only** edit made to that vendored string.

**Reason:** all scaffolds must share one tool set; ReWOO's original `Wikipedia` worker
returned longer page content, which would confound control flow with tool quality
(threat to validity §8.1).

## 2026-09-29 — T6: Reflexion reflector prompt adapted to GSM8K

The vendored `REFLECT_INSTRUCTION` says the agent had "access to an Docstore API
environment". For GSM8K that phrase is replaced by "a calculator tool" and `{examples}` is
filled with the empty string (the vendored `REFLECTIONS` few-shots are HotpotQA
trajectories and would be off-distribution for arithmetic). Mandated by plan §T6-I.

## 2026-09-29 — §2.1: model switched from `claude-haiku-4-5` to `deepseek-flash` (before any run)

**When:** after the pre-registration commit, **before any pilot or main run** (`results/` was
empty). No data informed this change.

**What changed**
- Model: `deepseek-flash` via DeepSeek's OpenAI-compatible API (`https://api.deepseek.com`),
  with **thinking mode disabled on every request** (`extra_body={"thinking": {"type": "disabled"}}`).
  A call that nevertheless returns reasoning output raises `ThinkingModeError` instead of being
  logged, because hidden reasoning tokens would be billed as output.
- Everything else in §2.1 is unchanged: temperature 0.7 (judge 0.0), per-role `max_tokens`,
  stop sequences, prompts, tools, tasks, replicates, statistics.
- Prices (USD / 1M tokens, list = peak): input (cache miss) 0.30, cache hit 0.006, output 1.20,
  no cache-write premium (so `cache_write` = input price in `usd_ideal[L]`). Every logged call is
  priced by its own `model` field, so logs can never be re-priced with the wrong sheet.
- Off-peak (outside 01–04 and 06–10 UTC, Mon–Fri) DeepSeek charges exactly half on every line.
  All analysis regimes use list prices, so **cost ratios between scaffolds are unaffected by when
  a run happened**. The actual bill is tracked separately as `usd_billed`.
- Spend guard: now on `usd_billed` (what the account is charged), default `MAX_USD = 4.50`
  (was 250 on `usd_nocache`), to stay under a $5 balance.
- `prompt_tokens.json`: DeepSeek has no token-count endpoint, so each system prompt is measured
  with one 1-token call minus the chat-template overhead (measured once). These calls are not in
  `calls.jsonl` and cost well under a cent.

**Why:** budget. The study is self-funded with a $5 limit; Haiku 4.5 projected to $150–250.

**Consequences for the claims**
- Still a single-model study; the model is now `deepseek-flash`. Absolute numbers are not
  comparable to Haiku; the Anthropic client is kept (`ACS_PROVIDER=anthropic`) for a later
  second-model check.
- H3 is **strengthened**, not weakened: DeepSeek caches every identical prefix automatically
  and reports hit tokens per call, so the empirical caching regime is now informative (on Haiku
  it would have been ≈ 0 because of the 4,096-token minimum). The `usd_ideal[L]` grid
  L ∈ {0, 1024, 4096} is kept as the schedule-independent counterfactual, as pre-registered.
- Prompts are sent to DeepSeek's servers. All inputs are public benchmark data.

**Tooling added for the scope-down ladder (§6.3):** `--limit N` (first N tasks of the frozen,
shuffled sample) and `--scaffolds …` (restrict the `others` stage, e.g. R=2 for ReWOO/Plan-Exec
only). The pilot report now projects the *new* spend per run (Reflexion's reused trial 1 is not
counted twice) and prints the whole ladder against the remaining budget.

## 2026-09-29 — T0: Provider limits and credit balance

<!-- FILL IN BEFORE LAUNCHING T13. -->
- Provider: DeepSeek, model `deepseek-flash`. Documented concurrency limit: 2,500 (far above the
  runner's `LLM_CONCURRENCY = 24`).
- Model version behind the `deepseek-flash` alias: DeepSeek-V4.1-Flash (pricing page, 2026-09-29).
- Credit balance at launch: $5.00 (topped up 2026-09-29); balance before the main run: $4.71
  (after pilot 1 $0.156 and pilot 2 $0.138).
- Scope-down ladder (§6.3) steps applied: **step 1** (GSM8K pool K=6); see the T12 entry.

## 2026-09-29 — T12: Pilot go/no-go decision

### Pilot 1 (`results/pilot/`, kept as evidence): **NO-GO**
- 420 runs, 0 errors, billed $0.156; empirical cache-read share 0.84.
- ReAct invalid-action rate 76.7% (HotpotQA) and 40.0% (GSM8K); ReAct HotpotQA halt rate 62%
  and EM 23.3% (below the 25–50% sanity band). Every other scaffold's parse health was fine.
- Cause (from the traces, not from accuracy): `deepseek-flash` often writes the action **without
  its label** (`Search[X]` instead of `Action 1: Search[X]`; 67 of 90 first ReAct moves on
  HotpotQA) and wraps labels in **markdown** on GSM8K (`**Action 1:** Calculate[…]`). The strict
  parser rejected these, the model repeated itself, and runs hit the 7-step cap.
- This also **biased the comparison**: Reflexion's later trials open with a reflection header
  that happened to elicit the labelled format more often, inflating its gap over ReAct.

### Fix: tolerant action parsing (`scaffolds/base.py`), applied identically to every scaffold
- Markdown emphasis around step labels is stripped (`**Action 1:**` → `Action 1:`); only the
  label is touched, so `**` inside an argument (`Calculate[2**3]`) is preserved.
- When no labelled `Action N:` exists, a line that **starts** with a **known** tool name and
  `[` (`Search`, `Lookup`, `Calculate`, `Finish`, `LLM`) is accepted as the action.
- The normalised text is what is stored and appended to the conversation (append-only kept).
- Replayed on all 2,511 pilot actor/executor outputs: parse rate HotpotQA 58.0% → 99.8%,
  GSM8K 88.2% → 99.4%; the remainder are genuine non-actions (thought-only text) and are still
  answered with the invalid-action observation.
- **No prompt was changed.** Model inputs change only in that the stored assistant turn is the
  normalised text. Regression tests: `tests/test_parse.py` (built from real pilot outputs).
- Allowed by plan §T12 ("fix prompts/parsers first … must be logged").

### Pilot 2 (`results/pilot2/`, fixed parser): **GO with §6.3 step 1**
- 420 runs, 0 errors, billed $0.138; cache-read share 0.84.
- Parse health fixed: ReAct invalid-action rate 3.3% on both benchmarks (was 76.7% / 40.0%);
  ReAct HotpotQA EM 48.9% (inside the 25–50% band); plan parse 100%; judge parse failures 0.
- The only failing check was cost. Actual balance before the main run:
  $5.00 − $0.156 − $0.138 = **$4.71**. Projected off-peak bill: as pre-registered **$4.29**
  (9% margin — too thin for a projection from 30 tasks, and a spend-guard trip mid-stage would
  leave arms unevenly incomplete); **§6.3 step 1 (N=300, pool K=10 HotpotQA / 6 GSM8K, R=3)
  $3.97** (16% margin).
- **Decision:** launch §6.3 step 1, off-peak, spend guard `--max-usd 4.40` on the billed amount
  (cumulative across the invocations of run tag `main`). Rationale: it is the first ladder step,
  keeps N=300 and every Reflexion/ReWOO/Plan-Exec replicate, and only trims the GSM8K vote@k
  curve from k≤10 to k≤6 — on a benchmark where the pilot shows every scaffold at 70–97%.
- Launch order protects the key comparisons if the guard ever trips: HotpotQA pool → GSM8K pool
  → Reflexion GT/Self → ReWOO/Plan-Exec.

### Main run, attempt 1 (`results/main_aborted_429/`, kept as evidence): **ABORTED**
- Command: HotpotQA pool stage, K=10. 3,000 runs, **2,451 errors (82%)**, billed $0.255.
- Every error was `HTTPStatusError: status 429` from the **Wikipedia API** (not DeepSeek): 300 new
  tasks at 8 parallel requests triggered Wikipedia's throttling, and the client gave up after 3
  quick retries. Errored runs had made 0.2 Search calls on average before failing.
- **Not resumed, set aside:** the 549 runs that succeeded are survivors of throttling (runs
  whose Wikipedia requests happened to get through), a selection biased toward short, cache-hit
  trajectories. Resuming would have mixed them with fresh draws. The whole attempt is discarded
  and the pool is rerun from scratch under a new `results/main/`.
- Fixes (no prompt, model or scaffold change):
  1. `tools/wiki.py`: honours `Retry-After`; any 429 sets a **shared cooldown** that every
     request waits out; 10 retries with jittered exponential backoff capped at 60 s;
     `WIKI_CONCURRENCY` 8 → 3. The disk cache from the aborted run is kept: it is keyed on the
     raw request and is the study-wide cache by design (§2.3).
  2. **Attempt ids.** Every run execution gets a fresh `attempt_id`, written on its run row and on
     every call it makes. The analysis bills a run **only for its final attempt's calls**
     (`analysis.load.live_calls`); before this, a retried run would have been billed for its
     failed attempts too, and the ideal-cache model would have seen two histories for one
     conversation. Billing reports still count every call (`Dataset.calls_all`), since that money
     was spent. Legacy rows without ids match run rows without ids.
- Tests: `tests/test_retry_and_attempts.py`, plus a runner test that retries get a new id.
- Budget after this attempt: $5.00 − $0.156 − $0.138 − $0.255 ≈ **$4.45** (confirm on the
  DeepSeek dashboard). Configuration unchanged (§6.3 step 1, projected $3.97 off-peak); the
  spend guard for the fresh `main` tag is set to **$4.20**.

### Observation to carry into the write-up: GSM8K ceiling
Pilot-1 GSM8K accuracy was 80–97% for every scaffold. With this model, GSM8K will mostly
discriminate scaffolds on **cost**, not accuracy; accuracy differences there will be small and
need the full N to detect. Kept as pre-registered (it is the reasoning-dominated contrast).

## 2026-09-29 — T5/T8: tool names are canonicalised, not taken verbatim

`parse_action` (T7) is specified to Title-case the tool name. A bare `.title()` turns
`LLM` into `Llm`, which would break ReWOO's `#Ek = LLM[...]` worker, so both parsers now
snap the Title-cased name to the exact spelling of a known tool (`Search`, `Lookup`,
`Calculate`, `Finish`, `LLM`) case-insensitively, and fall through unchanged otherwise.

**Reason:** identical behaviour to the spec for every tool the spec covers, and it stops a
purely cosmetic casing difference in the model's output from being scored as an unknown
tool. Unknown tools still produce the unknown-tool error string.

## 2026-09-29 — T11: `--fake-llm` self-test flag on the runner

The runner accepts `--fake-llm`, which drives the whole pipeline with scripted replies and
makes no API calls. It exists only to exercise runner → JSONL → analysis offline (the T11
acceptance criterion) and is never used for a real run; output written under such a run tag
carries a warning banner in the log.

## 2026-09-29 — T14: analysis-code fixes after the main run (before interpretation)

- **Holm correction bug.** `metrics.py` passed `p_value or 1.0` to `holm`; in Python `0.0 or 1.0`
  is `1.0`, so every comparison with a bootstrap p of exactly 0 was Holm-adjusted to 1.0. Now a
  p is replaced by 1.0 only when it is missing or NaN.
- **Bootstrap p floor.** `two_sided_p` now uses `(count + 1) / (B + 1)` per tail, so a bootstrap
  p is never reported as exactly 0 (smallest value 2/(B+1) ≈ 0.001 at B = 2,000).
- Neither fix changes an estimate or a CI; they change only p-values. Tests added.
- **Exploratory analyses added, labelled as such in RESULTS.md:** dollar-matched budget
  comparison (`exploratory_dollar_matched.csv`) and a cache-threshold sweep L ∈ {0, 256, 512,
  768, 1024, 1536, 2048, 4096} (`exploratory_h3_threshold_sweep.csv`). They were not
  pre-registered and are not used for any hypothesis verdict.
