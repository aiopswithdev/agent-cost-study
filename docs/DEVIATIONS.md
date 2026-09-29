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
- Credit balance at launch: $5.00 (topped up 2026-09-29); balance before the main run: _TBD_
- Scope-down ladder (§6.3) steps applied, if any: _TBD after the pilot_

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

### Pilot 2 (`results/pilot2/`): _TBD — rerun with the fixed parser, then decide here_
- Decision: _TBD_
- Configuration chosen from the scope-down table: _TBD_
- Evidence: `results/pilot2/PILOT_REPORT.md`

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
