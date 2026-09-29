# RESULTS — Budget-Matched, Oracle-Free Comparison of LLM Agent Scaffolds

> **STATUS: template.** The main runs (T13) have not been executed in this repository yet,
> so every number below is a placeholder. Fill each `<!-- FILL -->` marker from the file and
> column named beside it, all of which are produced by one command:
>
> ```bash
> python -m acs.analysis.report --run-tag main   # -> results/main/analysis/
> ```
>
> Do not hand-edit numbers: if a number is not reproducible from `results/main/{calls,runs}.jsonl`
> by that command, it does not belong in this document.

**Pre-registration commit hash:** `6656be08c2c44de55976db2767833c69a93cde08` (frozen 2026-09-29, before any run)
**Model:** `deepseek-flash`, thinking disabled (switched from `claude-haiku-4-5` before any run — see `docs/DEVIATIONS.md`) · **N / Pool K / R:** <!-- FILL: the configuration actually launched, from DEVIATIONS.md -->
**Benchmarks:** HotpotQA (distractor, validation) and GSM8K (main, test)

---

## RQ1 — Does any scaffold beat *budget-matched* ReAct?

**Pre-registered hypothesis (H1).** For each scaffold X ∈ {ReWOO, Plan-and-Execute,
Reflexion-Self, Reflexion-GT} with a positive naive gain, budget matching absorbs at least
half of it: `Δmatch(X) ≤ 0.5 × Δdef(X)`.

**Claim.** <!-- FILL: one sentence -->

| Benchmark | Scaffold | k*(X) | Δ vs ReAct [95% CI] | Δ vs vote@k*(X) [95% CI] | H1 supported? |
|---|---|---|---|---|---|
<!-- FILL from `analysis/h1_budget_matching.csv`: columns k_star, delta_default(+ci), delta_matched(+ci), h1_supported.
     Flag any row where `k_star_beyond_pool` is true: there vote@10 is a *conservative lower bound*
     on the control, so a surviving gain is weaker evidence than it looks. -->

Holm-adjusted significance for the eight RQ1 comparisons: `analysis/table2_comparisons.csv`
(families `X_vs_react` and `X_vs_matched_vote`).

**Figure.** `analysis/fig1_pareto.png` — accuracy against mean `usd_nocache` per task, with
the vote@k curve (k = 1..10) and its bootstrap band. A scaffold beats budget-matched ReAct
only if its point sits **above** that band.

**Verdict.** <!-- FILL: supported / not supported, per benchmark -->

---

## RQ2 — How much of Reflexion's gain is the ground-truth stopping oracle?

**Pre-registered hypothesis (H2).** `Acc(reflexion_gt) − Acc(reflexion_self) > 0`. The cost
direction was deliberately **not** pre-specified: a judge that falsely says "correct" stops
early and is cheaper; one that falsely says "incorrect" burns retries and is costlier.

**Claim.** <!-- FILL -->

| Benchmark | Acc GT | Acc Self | Δ [95% CI] | bootstrap p | Holm p | McNemar p |
|---|---|---|---|---|---|---|
<!-- FILL: `analysis/table2_comparisons.csv`, family == "oracle_premium"; accuracies from table1_metrics.csv -->

**The explanation we pre-registered reporting: the judge's confusion matrix vs gold.**

| Benchmark | judged trials | false-accept rate | false-reject rate | judge accuracy |
|---|---|---|---|---|
<!-- FILL: `analysis/judge_confusion.csv` -->

Cost direction observed: <!-- FILL: which of the two error modes dominated, and whether
reflexion_self therefore came out cheaper or dearer than reflexion_gt -->

**Figure.** `analysis/fig2_reflexion.png`.

**Verdict.** <!-- FILL -->

---

## RQ3 — Do cost rankings change when priced in dollars with prompt caching?

**Pre-registered hypothesis (H3), as written.** `cost(rewoo)/cost(react)` is closer to 1
under `usd_ideal[1024]` than under raw tokens (`I+O`). Secondary prediction: *"on Haiku 4.5
the effect under `usd_ideal[4096]` and `usd_empirical` is small, because most single prompts
in this study are shorter than the 4,096-token cacheable prefix — itself a reportable
finding, namely that rankings depend on the model's cache threshold."*

> **The secondary prediction was conditioned on a model we did not run.** It assumed
> Anthropic's 4,096-token minimum cacheable prefix. `deepseek-flash` caches every identical
> prefix automatically with **no documented minimum** and reports hit tokens per call, so
> `usd_empirical` is informative here rather than ≈ 0. The **primary** H3 test —
> `usd_ideal[1024]` versus raw tokens — is unchanged and is what the verdict below rests on.
> The `usd_ideal[L]` grid L ∈ {0, 1024, 4096} is kept exactly as pre-registered, which is
> what makes the cache-threshold sensitivity itself readable off Table 3.
> See `docs/DEVIATIONS.md`, entry *§2.1: model switched*.

**Claim.** <!-- FILL -->

| Benchmark | ratio (tokens) | ratio `usd_ideal[1024]` | ratio `usd_ideal[4096]` | ratio `usd_empirical` | H3 supported? |
|---|---|---|---|---|---|
<!-- FILL: `analysis/h3_verdict.csv`; the full per-scaffold grid is `analysis/table3_cost_ratios.csv` -->

Observed empirical cache-read share (Σ`cache_read` / Σ`I`): <!-- FILL: `analysis/SUMMARY.md` header -->
— compare against the ≈ 0 we expected from the 4,096-token threshold.

**Figure.** `analysis/fig3_cost_measures.png`.

**Verdict.** <!-- FILL -->

---

## RQ4 — Can cheap first-attempt features predict when the expensive scaffold pays off?

**Pre-registered hypothesis (H4).** A logistic model on first-attempt features predicts
"Reflexion-Self fixes a failed first attempt" with cross-validated AUC above a
benchmark-identity-only baseline.

**Claim.** <!-- FILL -->

| Model | CV AUC [95% CI] | n | positives |
|---|---|---|---|
<!-- FILL: `analysis/rq4_auc.csv` (rows: benchmark_only, first_attempt, first_attempt_plus_agreement) -->

**Routing policies** (route to Reflexion-Self when `p > τ`, else keep the ReAct answer;
the two extra pool samples the agreement feature needs are charged in the variant that
uses it):

| Policy | accuracy | mean tokens/task | routed fraction |
|---|---|---|---|
<!-- FILL: `analysis/rq4_routing_policies.csv` -->

**Figure.** `analysis/fig5_rq4.png` — ROC curves plus the policies on the cost/accuracy plane.

**Verdict.** <!-- FILL: positive or null; a null result is a reportable result -->

---

## Data quality

- Runs dropped for `status != "ok"`, per scaffold (must be < 2%): <!-- FILL: `analysis/dropped_runs.csv` -->
- Parse and failure modes: `analysis/table4_failure_modes.csv` — ReWOO plan-parse failures,
  halt rates, invalid-action rates, judge parse failures.
- Full metrics table (Table 1): `analysis/table1_metrics.csv`.

---

## Threats to validity and documented design decisions

These are copied from §8 of `docs/IMPLEMENTATION_PLAN.md` and must be stated in any write-up.

1. **Harmonised tools.** ReWOO's original Wikipedia worker returned longer page content;
   here every scaffold gets ReAct's 5-sentence `Search` + `Lookup`. This isolates control
   flow, at the cost of fidelity to ReWOO's original tool. ReWOO keeps its `LLM[]` worker,
   which is part of its paradigm.
2. **Plan-and-Execute has no canonical implementation.** Ours is a Plan-and-Solve-style
   planner + a ReAct executor + the same solver as ReWOO, with no replanning (matching
   Zhou et al. 2026's "no replanning" variant).
3. **Reflexion trial 1 is a ReAct pool sample** — identical in distribution by
   construction, which also makes the comparisons paired.
4. **A uniform `ANSWER_STYLE` line** is added to all answer-producing prompts (modern chat
   models are verbose; the original prompts were written for completion models). It is
   identical across scaffolds.
5. **Temperature 0.7 everywhere** (the originals mostly used greedy decoding) — needed for
   independent replicates and for vote@k.
6. **String-majority voting** for HotpotQA vote@k, with no LLM aggregator. This is
   *conservative for the control*, since semantically equivalent answers split votes. If the
   control still matches or beats a scaffold, the conclusion is strengthened.
7. **EM grading under-credits paraphrases**, equally for all scaffolds; F1 is reported as a
   check.
8. **Live Wikipedia** content differs from the 2018 HotpotQA-era content. The disk cache
   makes it identical *across scaffolds*, which is what the comparison requires.
9. **Empirical caching depends on the run schedule** (TTL, concurrency); hence
   `usd_ideal[L]` as the schedule-independent primary caching measure.
10. **Single model** — pre-registered as Haiku 4.5, **run on `deepseek-flash`** with
    thinking disabled (a pre-run, budget-driven change; see `docs/DEVIATIONS.md`).
    Absolute numbers are not comparable to Haiku's; generalisation to a second model is
    future work, and the config still supports it via `ACS_PROVIDER`.

**Additional threats observed during the runs:** <!-- FILL, or write "none observed" -->

**Scope caveat to state in any write-up.** This is a controlled study of *mechanism
instantiations* on terminal-answer tasks, not of production agents, which are hybrids
(Rombaut 2026).

---

## Deviations

Every change from the plan and the pre-registration is listed in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md). At the time of writing they are:

<!-- FILL: summarise the entries in docs/DEVIATIONS.md, with dates -->
