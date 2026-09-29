# RESULTS — Budget-Matched, Oracle-Free Comparison of LLM Agent Scaffolds

> Every number below is reproducible from `results/main/{calls,runs}.jsonl` with
> `python -m acs.analysis.report --run-tag main` (→ `results/main/analysis/`). Two tables are
> marked **exploratory** (not pre-registered); they come from
> `analysis/exploratory_*.csv`. Intervals are 95% cluster-bootstrap CIs over tasks
> (B = 2,000); p-values are bootstrap p (smallest reportable ≈ 0.001), Holm-corrected within
> each benchmark's family of nine pre-registered comparisons.

**Pre-registration commit hash:** `6656be08c2c44de55976db2767833c69a93cde08` (frozen 2026-09-29, before any run)
**Model:** `deepseek-flash` (DeepSeek-V4.1-Flash), thinking disabled — switched from `claude-haiku-4-5` before any run (`docs/DEVIATIONS.md`)
**Configuration launched:** N = 300 tasks per benchmark · ReAct pool K = 10 (HotpotQA) / 6 (GSM8K) · R = 3 for every other scaffold (§6.3 step 1)
**Benchmarks:** HotpotQA (distractor, validation) and GSM8K (main, test)
**Volume:** 12,000 runs, 54,597 LLM calls, 0 runs dropped · total billed for the main run **$3.64**

---

## Headline findings

1. **Reflexion is the only scaffold whose accuracy gain survives budget matching** (HotpotQA
   +5.3 pp for GT stopping, +2.5 pp for realistic self-stopping, both Holm p = 0.009) — but
   budget matching still absorbs 37–42% of its apparent gain, and the realistic version keeps
   only about half of it.
2. **About half of Reflexion's reported gain is the ground-truth stopping oracle.** On HotpotQA,
   Reflexion-GT beats ReAct by +8.3 pp; with a self-judge it is +4.3 pp. The judge is the reason:
   it **accepts 92% of wrong answers**, so Reflexion-Self stops early — it is also half the price.
3. **ReWOO's famous token advantage disappears in real dollars.** It uses **4.7× fewer tokens**
   than ReAct on HotpotQA, yet on DeepSeek's actual bill it cost **exactly the same (ratio 1.00)**,
   and under a 1,024-token cache minimum it costs **2.2× more**. Prompt caching rewards ReAct's
   one growing conversation and does nothing for ReWOO's many short, independent calls.
4. **No scaffold dominates** — consistent with Zhou et al. (2026). ReWOO is cheapest but loses
   25 pp on GSM8K; Plan-and-Execute is never better than budget-matched ReAct.

---

## RQ1 — Does any scaffold beat *budget-matched* ReAct?

**Pre-registered hypothesis (H1).** For each scaffold X with a positive naive gain, budget
matching absorbs at least half of it: `Δmatch(X) ≤ 0.5 × Δdef(X)`.

**Claim.** Budget matching shrinks every positive gain, but by 18–42%, not the ≥ 50% H1
predicted; Reflexion (both stopping rules) still beats token-matched ReAct on HotpotQA.

| Benchmark | Scaffold | k*(X) | Δ vs ReAct [95% CI] | Δ vs vote@k*(X) [95% CI] | H1 supported? |
|---|---|---|---|---|---|
| HotpotQA | ReWOO | 1.00 | −4.0 [−7.7, −0.3] | −3.8 [−7.4, −0.2] | n/a (no gain) |
| HotpotQA | Plan-Exec | 1.58 | +2.4 [−0.4, +5.5] | +1.6 [−1.2, +4.5] | No |
| HotpotQA | Reflexion-GT | 4.31 | **+8.3 [+6.0, +10.9]** | **+5.3 [+3.1, +7.5]** | No |
| HotpotQA | Reflexion-Self | 2.29 | **+4.3 [+2.7, +6.3]** | **+2.5 [+0.9, +4.2]** | No |
| GSM8K | ReWOO | 1.00 | −25.0 [−28.9, −21.3] | −25.4 [−29.3, −21.9] | n/a (no gain) |
| GSM8K | Plan-Exec | 2.35 | −3.0 [−4.8, −1.4] | −4.0 [−5.7, −2.4] | n/a (no gain) |
| GSM8K | Reflexion-GT | 1.21 | **+2.7 [+1.3, +4.3]** | **+2.2 [+1.0, +3.7]** | No |
| GSM8K | Reflexion-Self | 1.21 | +0.8 [−0.0, +1.8] | +0.3 [−0.5, +1.1] | Yes (gain not significant) |

(pp; k*(X) = number of ReAct samples whose majority vote costs the same tokens as X; no row
needed k* beyond the pool.) Holm-adjusted p for the bold rows: 0.009; ReWOO on HotpotQA:
0.12 (not significant after correction); Plan-Exec on HotpotQA: 0.21 / 0.28.

**Exploratory — matching on actual dollars instead of tokens** (`exploratory_dollar_matched.csv`,
point estimates). With caching, Reflexion is relatively more expensive, so the dollar-matched
control uses more samples and the gains shrink further: HotpotQA Reflexion-GT +5.0 pp
(k* = 5.4), Reflexion-Self +1.9 pp (k* = 3.0), Plan-Exec +0.7 pp (k* = 2.2).

**Figure.** `analysis/fig1_pareto.png` — ReAct's vote@k curve (k = 1..10) with its bootstrap band;
Reflexion-GT is the only point clearly above the curve on HotpotQA.

**Verdict.** H1 **not supported** on either benchmark. Budget matching matters (it removes
18–42% of the naive gains, more when priced in dollars), but it does not erase Reflexion's
advantage on the tool-heavy benchmark.

---

## RQ2 — How much of Reflexion's gain is the ground-truth stopping oracle?

**Pre-registered hypothesis (H2).** `Acc(reflexion_gt) − Acc(reflexion_self) > 0`; cost
direction not pre-specified.

**Claim.** The oracle is worth +4.0 pp on HotpotQA and +1.9 pp on GSM8K — about half (HotpotQA)
and about 70% (GSM8K) of Reflexion-GT's gain over ReAct — and the realistic self-judge fails by
**accepting wrong answers**, which also makes it roughly half the price on HotpotQA.

| Benchmark | Acc GT | Acc Self | Δ [95% CI] | bootstrap p | Holm p | McNemar p |
|---|---|---|---|---|---|---|
| HotpotQA | 56.9% | 52.9% | **+4.0 [+2.1, +6.0]** | 0.001 | 0.009 | 0.004 |
| GSM8K | 98.1% | 96.2% | **+1.9 [+0.8, +3.3]** | 0.001 | 0.009 | 0.063 |

**Judge confusion matrix vs gold** (every judged trial):

| Benchmark | judged trials | false-accept rate | false-reject rate | judge accuracy |
|---|---|---|---|---|
| HotpotQA | 859 | **91.6%** (347 of 379 wrong answers accepted) | 1.0% | 59.0% |
| GSM8K | 909 | **91.9%** (34 of 37 wrong answers accepted) | 0.7% | 95.6% |

**Cost direction observed.** False accepts dominate overwhelmingly, so Reflexion-Self stops
early: mean trials 1.51 vs 2.86 on HotpotQA, and cost per task $0.0059 vs $0.0112 (no-cache
list price) — **about half**. On GSM8K both need ~1 trial and cost the same. In practice
Reflexion-Self only retries when trial 1 produced *no* answer (hit the step cap), which is the
mechanism behind RQ4 below.

**Figure.** `analysis/fig2_reflexion.png`.

**Verdict.** H2 **supported** on both benchmarks (Holm p = 0.009). Consistent with Huang et
al. (2024): without external feedback, the model's self-verification barely rejects anything.
Studies that report Reflexion with ground-truth stopping (e.g. Simhadri 2026) overstate its
deployable accuracy by roughly this margin, and understate the gap to cheaper scaffolds.

---

## RQ3 — Do cost rankings change when priced in dollars with prompt caching?

**Pre-registered hypothesis (H3), as written.** `cost(rewoo)/cost(react)` is closer to 1
under `usd_ideal[1024]` than under raw tokens (`I+O`).

> The secondary prediction in the pre-registration assumed Anthropic's 4,096-token minimum
> cacheable prefix. `deepseek-flash` caches every identical prefix automatically and reports
> hit tokens per call, so `usd_empirical` is informative here. The primary test is unchanged.

**Claim.** Caching erases — and at realistic thresholds reverses — ReWOO's token advantage on
HotpotQA; on GSM8K it shrinks it.

| Benchmark | ratio (tokens) | ratio `usd_ideal[1024]` | ratio `usd_ideal[4096]` | ratio `usd_empirical` (DeepSeek bill) | H3 as written |
|---|---|---|---|---|---|
| HotpotQA | 0.21 | **2.16** | 0.26 | **1.00** | Not supported (ratio overshoots past 1) |
| GSM8K | 0.36 | 0.44 | 0.43 | 0.63 | Supported |

Empirical cache-read share: **83.9%** of all input tokens were served from DeepSeek's cache.

**Exploratory — the cache-threshold sweep** (`exploratory_h3_threshold_sweep.csv`; cost relative
to ReAct under perfect caching with minimum cacheable length L):

| HotpotQA | tokens | L=0 | L=512 | L=1024 | L=2048 | L=4096 | DeepSeek bill |
|---|---|---|---|---|---|---|---|
| ReWOO | 0.21 | 1.23 | 1.32 | 2.16 | 0.30 | 0.26 | 1.00 |
| Plan-Exec | 1.57 | 2.47 | 2.82 | 2.82 | 1.78 | 1.63 | 2.20 |
| Reflexion-GT | 4.30 | 6.90 | 6.90 | 6.90 | 2.74 | 4.41 | 5.35 |
| Reflexion-Self | 2.28 | 3.64 | 3.72 | 3.72 | 1.64 | 2.33 | 2.96 |

Why: ReAct is one append-only conversation, so from turn 2 onward almost all of its input is a
cached prefix. ReWOO's calls are short and independent — its planner prompt is 888 tokens,
just below a 1,024 threshold, and its solver and worker prompts are tiny — so there is almost
nothing for a cache to reuse. The same effect makes every multi-conversation scaffold
*relatively* more expensive once caching is on.

**Figure.** `analysis/fig3_cost_measures.png`.

**Verdict.** As literally operationalised, H3 is **supported on GSM8K and not on HotpotQA** —
on HotpotQA the ratio did not move *toward* 1, it moved *past* it (0.21 → 2.16). Substantively
the effect is larger than predicted: **the cost ranking of scaffolds depends on whether, and at
what threshold, the provider caches prompts.** Token counts alone — the measure used by every
prior scaffold comparison — get the ReWOO-vs-ReAct ranking wrong on real billing.

---

## RQ4 — Can cheap first-attempt features predict when the expensive scaffold pays off?

**Pre-registered hypothesis (H4).** A logistic model on first-attempt features predicts
"Reflexion-Self fixes a failed first attempt" with CV AUC above a benchmark-only baseline.

**Claim.** Technically yes, but the evidence is weak and largely structural.

| Model | CV AUC [95% CI] | n | positives |
|---|---|---|---|
| benchmark only | 0.68 [0.55, 0.79] | 600 | 14 |
| first-attempt features | 0.96 [0.94, 0.97] | 600 | 14 |
| + agreement of first 3 samples | 0.96 [0.94, 0.97] | 600 | 14 |

**Routing policies** (route to Reflexion-Self when `p > τ`, else keep ReAct's answer; both
benchmarks pooled; tokens/task):

| Policy | accuracy | mean tokens/task | routed fraction |
|---|---|---|---|
| always ReAct | 72.2% | 5,220 | 0% |
| route p > 0.2 | 73.8% | 8,756 | 6.7% |
| route p > 0.3 | 72.3% | 6,612 | 2.3% |
| always Reflexion-Self | 74.3% | 10,220 | 100% |
| route p > 0.2, charging the agreement samples | 73.8% | 19,513 | 6.8% |

**Figure.** `analysis/fig5_rq4.png`.

**Verdict.** H4 **supported as written, with two caveats that must be stated:** (1) only
**14 of 600** tasks are positives, so the AUC rests on very few events; (2) the strongest
feature is whether the first attempt **halted** (coefficient 1.5) — and because the judge
accepts almost every answer (RQ2), a halted first attempt is nearly the only way
Reflexion-Self retries at all. The predictor is mostly re-discovering Reflexion-Self's own
control flow. The routing gain is small (+1.6 pp at 1.7× ReAct's tokens) and disappears once
the agreement samples are paid for.

---

## Data quality

- **Runs dropped:** 0 of 12,000 (every scaffold 0%, threshold < 2%).
- **Parse health** (`table4_failure_modes.csv`): ReAct invalid-action rate 8.8% (HotpotQA) /
  2.7% (GSM8K) of runs; ReWOO plan-parse 99.7% / 95.2%; Plan-Exec plan-parse 100%; judge
  parse failures 0.0% / 0.1%.
- **Halt (step-cap) rates on HotpotQA:** ReAct 17.3%, Plan-Exec 35.4% (of runs with at least one
  halted step), Reflexion-GT 14.3%, Reflexion-Self 8.2%.
- Full metrics: `table1_metrics.csv`; all comparisons: `table2_comparisons.csv`.

---

## Threats to validity and documented design decisions

From §8 of `docs/IMPLEMENTATION_PLAN.md`:

1. **Harmonised tools.** Every scaffold gets ReAct's 5-sentence `Search` + `Lookup`; ReWOO's
   original worker returned longer pages. ReWOO keeps its `LLM[]` worker.
2. **Plan-and-Execute has no canonical implementation** (Plan-and-Solve planner + ReAct executor
   + ReWOO's solver, no replanning).
3. **Reflexion trial 1 is a ReAct pool sample** — same distribution, and it makes comparisons paired.
4. **A uniform `ANSWER_STYLE` line** on all answer-producing prompts.
5. **Temperature 0.7 everywhere** (the originals mostly used greedy decoding).
6. **String-majority voting** for HotpotQA vote@k — conservative for the control.
7. **EM grading under-credits paraphrases**, equally for all scaffolds; F1 reported.
8. **Live Wikipedia**, identical across scaffolds via the disk cache.
9. **Empirical caching depends on the run schedule**; `usd_ideal[L]` is the schedule-independent measure.
10. **Single model** (`deepseek-flash`, not the pre-registered Haiku 4.5).

**Additional threats observed during the runs:**
- **GSM8K is near ceiling** for this model (ReAct 95.4%); accuracy differences there are small
  and mostly about cost.
- **Action-parser tolerance was added after pilot 1** (the model often omits the `Action N:`
  label). It is applied identically to all scaffolds; without it the comparison was badly biased
  in Reflexion's favour. Prompts were not changed.
- **The first main-run attempt was aborted** (Wikipedia throttling, 82% of runs failed) and
  discarded entirely to avoid survivor bias; the reported data come from a clean rerun.
- **RQ1's pre-registered matching is on tokens.** Matching on actual dollars (exploratory) shrinks
  the Reflexion gains further.
- **Two analysis-code bugs were fixed before interpretation:** the Holm correction mapped p = 0 to
  p = 1, and bootstrap p-values could be exactly 0 (now floored at the resolution 2/(B+1)).

**Scope caveat.** This is a controlled study of *mechanism instantiations* on terminal-answer
tasks with one model, not of production agents, which are hybrids (Rombaut 2026).

---

## Deviations

All entries are in [docs/DEVIATIONS.md](docs/DEVIATIONS.md), dated 2026-09-29:

1. ReWOO few-shot tool renamed `Wikipedia[` → `Search[` (T6).
2. Reflexion reflector prompt adapted to GSM8K (T6).
3. **Model switched from `claude-haiku-4-5` to `deepseek-flash`**, thinking disabled, before any
   run (budget); spend guard on billed dollars; per-model pricing; `--limit` / `--scaffolds` added.
4. Provider limits and balance recorded; model version DeepSeek-V4.1-Flash.
5. **Pilot 1 NO-GO** → tolerant action parsing; **pilot 2 GO** with §6.3 step 1 (GSM8K pool K = 6).
6. **Main attempt 1 aborted** (Wikipedia 429s) → retry/cooldown in the Wikipedia client and
   per-attempt billing; clean rerun.
7. Tool names canonicalised in the parsers (T5/T8).
8. `--fake-llm` self-test flag on the runner (T11).
9. Analysis fixes after the main run: Holm p = 0 bug, bootstrap p floor; exploratory dollar-matched
   and cache-threshold analyses added and labelled as such.
