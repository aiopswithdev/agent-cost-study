# PILOT REPORT — run tag `pilot2`

- pilot runs: **420** (0 errored)
- pilot LLM calls: **2029**
- pilot bill: **$0.138** actually charged (list price: $0.276 with cache, $0.994 without)
- budget: MAX_USD $4.34 − pilot $0.138 = **$4.20 remaining** for the main run
- empirical cache-read share (Σcache_read / ΣI): **0.8361** (share of input tokens the provider served from its prefix cache)

## Per scaffold × benchmark

| benchmark   | scaffold       |   n_runs |   error_rate |   accuracy |     f1 |   mean_tokens_in |   mean_tokens_out |   mean_usd_nocache |   mean_usd_empirical |   mean_llm_calls |   mean_wall_s |   invalid_action_rate |   halt_rate |   plan_parse_ok_rate |   judge_parse_fail_rate |   mean_trials |
|:------------|:---------------|---------:|-------------:|-----------:|-------:|-----------------:|------------------:|-------------------:|---------------------:|-----------------:|--------------:|----------------------:|------------:|---------------------:|------------------------:|--------------:|
| gsm8k       | react          |       90 |       0.0000 |     0.9556 | 0.9556 |        2412.8333 |          258.7778 |             0.0010 |               0.0005 |           3.5111 |        3.5351 |                0.0333 |      0.0111 |             nan      |                nan      |      nan      |
| gsm8k       | rewoo          |       30 |       0.0000 |     0.7000 | 0.7000 |         711.2333 |          153.1667 |             0.0004 |               0.0003 |           2.1000 |        2.2117 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| gsm8k       | plan_exec      |       30 |       0.0000 |     0.9667 | 0.9667 |        6015.3000 |          309.6667 |             0.0022 |               0.0009 |           9.3667 |        8.5446 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| gsm8k       | reflexion_gt   |       30 |       0.0000 |     0.9667 | 0.9667 |        3138.5000 |          300.3667 |             0.0013 |               0.0006 |           4.3333 |        0.6495 |                0.0000 |      0.0000 |             nan      |                  0.0000 |        1.1333 |
| gsm8k       | reflexion_self |       30 |       0.0000 |     0.9333 | 0.9333 |        3061.3667 |          330.4333 |             0.0013 |               0.0008 |           4.8333 |        1.4103 |                0.0000 |      0.0000 |             nan      |                  0.0000 |        1.0333 |
| hotpotqa    | react          |       90 |       0.0000 |     0.4889 | 0.5670 |        8662.9444 |          125.4444 |             0.0027 |               0.0005 |           4.5333 |        4.5155 |                0.0333 |      0.2778 |             nan      |                nan      |      nan      |
| hotpotqa    | rewoo          |       30 |       0.0000 |     0.4333 | 0.5239 |        1612.5333 |          172.0667 |             0.0007 |               0.0004 |           3.6000 |        3.9388 |                0.0000 |      0.0000 |               1.0000 |                nan      |      nan      |
| hotpotqa    | plan_exec      |       30 |       0.0000 |     0.6000 | 0.6730 |       12626.0000 |          377.6333 |             0.0042 |               0.0010 |           8.4667 |        8.4896 |                0.0000 |      0.3667 |               1.0000 |                nan      |      nan      |
| hotpotqa    | reflexion_gt   |       30 |       0.0000 |     0.5333 | 0.6480 |       37502.2000 |          788.3333 |             0.0122 |               0.0025 |          16.3333 |       13.3981 |                0.0000 |      0.2333 |             nan      |                  0.0000 |        2.8667 |
| hotpotqa    | reflexion_self |       30 |       0.0000 |     0.6667 | 0.7808 |       21810.4667 |          428.1000 |             0.0071 |               0.0015 |          10.7333 |        6.6335 |                0.0000 |      0.0667 |             nan      |                  0.0000 |        1.7333 |

## Projected main run

Planned counts: N=300, pool K=10, others R=3 → 13,200 runs.

| scaffold       | benchmark   |   pilot_runs |   planned_runs |   mean_usd_per_run |   projected_usd |   mean_wall_s |
|:---------------|:------------|-------------:|---------------:|-------------------:|----------------:|--------------:|
| plan_exec      | gsm8k       |           30 |            900 |             0.0009 |          0.8546 |        8.5446 |
| react          | gsm8k       |           90 |           3000 |             0.0005 |          1.5753 |        3.5351 |
| reflexion_gt   | gsm8k       |           30 |            900 |             0.0001 |          0.1168 |        0.6495 |
| reflexion_self | gsm8k       |           30 |            900 |             0.0002 |          0.2131 |        1.4103 |
| rewoo          | gsm8k       |           30 |            900 |             0.0003 |          0.2615 |        2.2117 |
| plan_exec      | hotpotqa    |           30 |            900 |             0.0010 |          0.9301 |        8.4896 |
| react          | hotpotqa    |           90 |           3000 |             0.0005 |          1.4757 |        4.5155 |
| reflexion_gt   | hotpotqa    |           30 |            900 |             0.0020 |          1.8065 |       13.3981 |
| reflexion_self | hotpotqa    |           30 |            900 |             0.0010 |          0.9424 |        6.6335 |
| rewoo          | hotpotqa    |           30 |            900 |             0.0004 |          0.4033 |        3.9388 |

- **projected bill, as pre-registered: $8.58 at peak rates, $4.29 if run entirely off-peak** (new spend per run at list price with caching; Reflexion trial 1 not double-counted)

## Scope-down ladder vs. the remaining budget

| configuration                    |   N | K (hotpot/gsm8k)   | R (rewoo,pe / reflexion)   |   runs |   usd_peak |   usd_offpeak | fits_offpeak   | fits_peak   |
|:---------------------------------|----:|:-------------------|:---------------------------|-------:|-----------:|--------------:|:---------------|:------------|
| as pre-registered                | 300 | 10/10              | 3 / 3                      |  13200 |       8.58 |          4.29 | False          | False       |
| §6.3 step 1: GSM8K K=6           | 300 | 10/6               | 3 / 3                      |  12000 |       7.95 |          3.97 | True           | False       |
| §6.3 step 2: N=200               | 200 | 10/6               | 3 / 3                      |   8000 |       5.30 |          2.65 | True           | False       |
| §6.3 step 3: R=2 ReWOO/Plan-Exec | 200 | 10/6               | 2 / 3                      |   7200 |       4.76 |          2.38 | True           | False       |
| §6.3 step 4: HotpotQA K=6        | 200 | 6/6                | 2 / 3                      |   6400 |       4.36 |          2.18 | True           | False       |
| extra: N=150                     | 150 | 6/6                | 2 / 3                      |   4800 |       3.27 |          1.64 | True           | True        |
| extra: N=100                     | 100 | 6/6                | 2 / 3                      |   3200 |       2.18 |          1.09 | True           | True        |

**Recommended:** `§6.3 step 1: GSM8K K=6` — the largest configuration that fits $4.20 off-peak ($3.97). DeepSeek peak = 01:00–04:00 and 06:00–10:00 UTC, Mon–Fri; launch outside it.

- observed throughput: **387.2 runs/min** at MAX_RUNS_IN_FLIGHT=32
- **projected wall clock: 0.6 hours**

## Go / no-go

**NO-GO — resolve these first:**

- **cost**: the pre-registered main run projects to $4.29 off-peak, more than the remaining budget $4.20 — launch the largest configuration marked as fitting in the scope-down table below, and log it in DEVIATIONS.md.

A prompt change after the pilot is allowed but must be logged in `docs/DEVIATIONS.md`. After the main run starts, prompts are frozen.

Record the decision and its reason in `docs/DEVIATIONS.md`.
