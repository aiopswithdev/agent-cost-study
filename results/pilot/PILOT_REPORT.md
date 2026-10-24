# PILOT REPORT — run tag `pilot`

- pilot runs: **420** (0 errored)
- pilot LLM calls: **2365**
- pilot bill: **$0.156** actually charged (list price: $0.311 with cache, $1.182 without)
- budget: MAX_USD $4.50 − pilot $0.156 = **$4.34 remaining** for the main run
- empirical cache-read share (Σcache_read / ΣI): **0.8436** (share of input tokens the provider served from its prefix cache)

## Per scaffold × benchmark

| benchmark   | scaffold       |   n_runs |   error_rate |   accuracy |     f1 |   mean_tokens_in |   mean_tokens_out |   mean_usd_nocache |   mean_usd_empirical |   mean_llm_calls |   mean_wall_s |   invalid_action_rate |   halt_rate |   plan_parse_ok_rate |   judge_parse_fail_rate |   mean_trials |
|:------------|:---------------|---------:|-------------:|-----------:|-------:|-----------------:|------------------:|-------------------:|---------------------:|-----------------:|--------------:|----------------------:|------------:|---------------------:|------------------------:|--------------:|
| gsm8k       | react          |       90 |       0.0000 |     0.9000 | 0.9000 |        2976.0778 |          283.2556 |             0.0012 |               0.0006 |           3.8778 |        4.1194 |                0.4000 |      0.0667 |             nan      |                nan      |      nan      |
| gsm8k       | rewoo          |       30 |       0.0000 |     0.8000 | 0.8000 |         719.3667 |          151.0333 |             0.0004 |               0.0003 |           2.0333 |        2.1902 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| gsm8k       | plan_exec      |       30 |       0.0000 |     0.9333 | 0.9333 |        5988.5667 |          298.1000 |             0.0022 |               0.0009 |           9.3667 |        8.7411 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| gsm8k       | reflexion_gt   |       30 |       0.0000 |     0.9667 | 0.9667 |        4190.5667 |          389.5667 |             0.0017 |               0.0008 |           4.8667 |        1.0824 |                0.0000 |      0.0000 |             nan      |                  0.0000 |        1.1333 |
| gsm8k       | reflexion_self |       30 |       0.0000 |     0.9667 | 0.9667 |        3289.0333 |          309.1000 |             0.0014 |               0.0007 |           4.9000 |        1.0491 |                0.0000 |      0.0000 |             nan      |                  0.0000 |        1.0000 |
| hotpotqa    | react          |       90 |       0.0000 |     0.2333 | 0.2685 |       11364.6889 |          124.7778 |             0.0036 |               0.0006 |           6.3111 |        6.2425 |                0.7667 |      0.6222 |             nan      |                nan      |      nan      |
| hotpotqa    | rewoo          |       30 |       0.0000 |     0.5333 | 0.6128 |        1632.2333 |          188.5333 |             0.0007 |               0.0005 |           3.7333 |        4.4306 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| hotpotqa    | plan_exec      |       30 |       0.0000 |     0.6000 | 0.6739 |       14069.4333 |          362.6000 |             0.0047 |               0.0011 |           9.2667 |        9.5051 |                0.0000 |      0.5667 |               1.0000 |                nan      |      nan      |
| hotpotqa    | reflexion_gt   |       30 |       0.0000 |     0.6333 | 0.7182 |       39943.7333 |          806.5000 |             0.0130 |               0.0026 |          18.3667 |       14.3022 |                0.0000 |      0.1333 |             nan      |                  0.0000 |        3.0333 |
| hotpotqa    | reflexion_self |       30 |       0.0000 |     0.6000 | 0.7141 |       31853.2000 |          570.5333 |             0.0102 |               0.0021 |          15.8000 |       11.0022 |                0.0000 |      0.1000 |             nan      |                  0.0000 |        2.3000 |

## Projected main run

Planned counts: N=300, pool K=10, others R=3 → 13,200 runs.

| scaffold       | benchmark   |   pilot_runs |   planned_runs |   mean_usd_per_run |   projected_usd |   mean_wall_s |
|:---------------|:------------|-------------:|---------------:|-------------------:|----------------:|--------------:|
| plan_exec      | gsm8k       |           30 |            900 |             0.0009 |          0.8539 |        8.7411 |
| react          | gsm8k       |           90 |           3000 |             0.0006 |          1.7288 |        4.1194 |
| reflexion_gt   | gsm8k       |           30 |            900 |             0.0003 |          0.2393 |        1.0824 |
| reflexion_self | gsm8k       |           30 |            900 |             0.0002 |          0.1766 |        1.0491 |
| rewoo          | gsm8k       |           30 |            900 |             0.0003 |          0.2636 |        2.1902 |
| plan_exec      | hotpotqa    |           30 |            900 |             0.0011 |          0.9627 |        9.5051 |
| react          | hotpotqa    |           90 |           3000 |             0.0006 |          1.7903 |        6.2425 |
| reflexion_gt   | hotpotqa    |           30 |            900 |             0.0021 |          1.8483 |       14.3022 |
| reflexion_self | hotpotqa    |           30 |            900 |             0.0015 |          1.3858 |       11.0022 |
| rewoo          | hotpotqa    |           30 |            900 |             0.0005 |          0.4411 |        4.4306 |

- **projected bill, as pre-registered: $9.69 at peak rates, $4.85 if run entirely off-peak** (new spend per run at list price with caching; Reflexion trial 1 not double-counted)

## Scope-down ladder vs. the remaining budget

| configuration                    |   N | K (hotpot/gsm8k)   | R (rewoo,pe / reflexion)   |   runs |   usd_peak |   usd_offpeak | fits_offpeak   | fits_peak   |
|:---------------------------------|----:|:-------------------|:---------------------------|-------:|-----------:|--------------:|:---------------|:------------|
| as pre-registered                | 300 | 10/10              | 3 / 3                      |  13200 |       9.69 |          4.85 | False          | False       |
| §6.3 step 1: GSM8K K=6           | 300 | 10/6               | 3 / 3                      |  12000 |       9.00 |          4.50 | False          | False       |
| §6.3 step 2: N=200               | 200 | 10/6               | 3 / 3                      |   8000 |       6.00 |          3.00 | True           | False       |
| §6.3 step 3: R=2 ReWOO/Plan-Exec | 200 | 10/6               | 2 / 3                      |   7200 |       5.44 |          2.72 | True           | False       |
| §6.3 step 4: HotpotQA K=6        | 200 | 6/6                | 2 / 3                      |   6400 |       4.96 |          2.48 | True           | False       |
| extra: N=150                     | 150 | 6/6                | 2 / 3                      |   4800 |       3.72 |          1.86 | True           | True        |
| extra: N=100                     | 100 | 6/6                | 2 / 3                      |   3200 |       2.48 |          1.24 | True           | True        |

**Recommended:** `§6.3 step 2: N=200` — the largest configuration that fits $4.34 off-peak ($3.00). DeepSeek peak = 01:00–04:00 and 06:00–10:00 UTC, Mon–Fri; launch outside it.

- observed throughput: **322.3 runs/min** at MAX_RUNS_IN_FLIGHT=32
- **projected wall clock: 0.7 hours**

## Go / no-go

**NO-GO — resolve these first:**

- **gsm8k/react**: invalid-action rate 40.0% > 10% — fix the ReAct prompt/parser first.
- **hotpotqa/react**: invalid-action rate 76.7% > 10% — fix the ReAct prompt/parser first.
- **hotpotqa/react**: ReAct HotpotQA EM 23.3% is outside the sanity band 25%–50% — inspect traces before continuing.
- **cost**: the pre-registered main run projects to $4.85 off-peak, more than the remaining budget $4.34 — launch the largest configuration marked as fitting in the scope-down table below, and log it in DEVIATIONS.md.

A prompt change after the pilot is allowed but must be logged in `docs/DEVIATIONS.md`. After the main run starts, prompts are frozen.

Record the decision and its reason in `docs/DEVIATIONS.md`.
