# Analysis summary — run tag `main`

- runs analysed: **12000** (dropped for `status != ok`: 0)
- LLM calls: **54597**
- empirical cache-read share (Σcache_read / ΣI): **0.8392**

## Dropped runs per scaffold (must stay < 2%)

| benchmark   | scaffold       |   n_total |   n_dropped |   frac_dropped |
|:------------|:---------------|----------:|------------:|---------------:|
| gsm8k       | plan_exec      |       900 |           0 |         0.0000 |
| gsm8k       | react          |      1800 |           0 |         0.0000 |
| gsm8k       | reflexion_gt   |       900 |           0 |         0.0000 |
| gsm8k       | reflexion_self |       900 |           0 |         0.0000 |
| gsm8k       | rewoo          |       900 |           0 |         0.0000 |
| hotpotqa    | plan_exec      |       900 |           0 |         0.0000 |
| hotpotqa    | react          |      3000 |           0 |         0.0000 |
| hotpotqa    | reflexion_gt   |       900 |           0 |         0.0000 |
| hotpotqa    | reflexion_self |       900 |           0 |         0.0000 |
| hotpotqa    | rewoo          |       900 |           0 |         0.0000 |

## Table 1 — metrics (Fig 1)

| benchmark   | arm                           |   n_tasks |    acc |   acc_ci_lo |   acc_ci_hi |       f1 |   tokens_total |   n_llm_calls |   usd_nocache |   usd_empirical |   usd_ideal_1024 |   usd_ideal_4096 |   cop_usd_nocache |
|:------------|:------------------------------|----------:|-------:|------------:|------------:|---------:|---------------:|--------------:|--------------:|----------------:|-----------------:|-----------------:|------------------:|
| gsm8k       | react                         |       300 | 0.9544 |      0.9311 |      0.9744 |   0.9544 |      2431.9067 |        3.3056 |        0.0009 |          0.0005 |           0.0009 |           0.0009 |            0.0010 |
| gsm8k       | rewoo                         |       300 | 0.7044 |      0.6622 |      0.7433 |   0.7044 |       871.0356 |        2.1222 |        0.0004 |          0.0003 |           0.0004 |           0.0004 |            0.0006 |
| gsm8k       | plan_exec                     |       300 | 0.9244 |      0.9000 |      0.9478 |   0.9244 |      5696.1556 |        8.6167 |        0.0020 |          0.0009 |           0.0020 |           0.0020 |            0.0021 |
| gsm8k       | reflexion_gt                  |       300 | 0.9811 |      0.9677 |      0.9933 |   0.9811 |      2934.2911 |        3.8156 |        0.0011 |          0.0006 |           0.0011 |           0.0011 |            0.0012 |
| gsm8k       | reflexion_self                |       300 | 0.9622 |      0.9422 |      0.9800 |   0.9622 |      2926.4167 |        4.3844 |        0.0011 |          0.0007 |           0.0011 |           0.0011 |            0.0012 |
| gsm8k       | react_vote@1                  |       300 | 0.9587 |      0.9387 |      0.9767 | nan      |      2427.4671 |      nan      |        0.0009 |          0.0005 |           0.0009 |           0.0009 |            0.0010 |
| gsm8k       | react_vote@3                  |       300 | 0.9686 |      0.9493 |      0.9859 | nan      |      7280.9944 |      nan      |        0.0028 |          0.0014 |           0.0028 |           0.0028 |            0.0029 |
| gsm8k       | react_vote@5                  |       300 | 0.9684 |      0.9485 |      0.9867 | nan      |     12125.9723 |      nan      |        0.0047 |          0.0024 |           0.0046 |           0.0047 |            0.0048 |
| gsm8k       | react_vote@k*(rewoo)          |       300 | 0.9587 |      0.9387 |      0.9767 | nan      |      2427.4671 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| gsm8k       | react_vote@k*(plan_exec)      |       300 | 0.9640 |      0.9446 |      0.9810 | nan      |      5696.1556 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| gsm8k       | react_vote@k*(reflexion_gt)   |       300 | 0.9593 |      0.9392 |      0.9771 | nan      |      2934.2911 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| gsm8k       | react_vote@k*(reflexion_self) |       300 | 0.9593 |      0.9392 |      0.9771 | nan      |      2926.4167 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| hotpotqa    | react                         |       300 | 0.4856 |      0.4322 |      0.5367 |   0.6166 |      8127.6144 |        4.2678 |        0.0025 |          0.0004 |           0.0003 |           0.0025 |            0.0052 |
| hotpotqa    | rewoo                         |       300 | 0.4456 |      0.3922 |      0.4944 |   0.5704 |      1731.5467 |        3.5200 |        0.0007 |          0.0004 |           0.0007 |           0.0007 |            0.0015 |
| hotpotqa    | plan_exec                     |       300 | 0.5100 |      0.4567 |      0.5622 |   0.6437 |     12766.6389 |        8.3767 |        0.0041 |          0.0010 |           0.0009 |           0.0041 |            0.0081 |
| hotpotqa    | reflexion_gt                  |       300 | 0.5689 |      0.5189 |      0.6189 |   0.6765 |     34936.8433 |       15.0978 |        0.0112 |          0.0024 |           0.0021 |           0.0112 |            0.0197 |
| hotpotqa    | reflexion_self                |       300 | 0.5289 |      0.4767 |      0.5811 |   0.6665 |     18521.7667 |        9.1167 |        0.0059 |          0.0013 |           0.0012 |           0.0059 |            0.0112 |
| hotpotqa    | react_vote@1                  |       300 | 0.4835 |      0.4336 |      0.5348 | nan      |      8085.9635 |      nan      |        0.0025 |          0.0004 |           0.0003 |           0.0025 |            0.0052 |
| hotpotqa    | react_vote@3                  |       300 | 0.5106 |      0.4581 |      0.5631 | nan      |     24293.6863 |      nan      |        0.0076 |          0.0013 |           0.0009 |           0.0076 |            0.0148 |
| hotpotqa    | react_vote@5                  |       300 | 0.5187 |      0.4660 |      0.5720 | nan      |     40497.3458 |      nan      |        0.0126 |          0.0022 |           0.0015 |           0.0126 |            0.0244 |
| hotpotqa    | react_vote@10                 |       300 | 0.5333 |      0.4767 |      0.5867 | nan      |     80960.9567 |      nan      |        0.0253 |          0.0044 |           0.0031 |           0.0253 |            0.0474 |
| hotpotqa    | react_vote@k*(rewoo)          |       300 | 0.4835 |      0.4336 |      0.5348 | nan      |      8085.9635 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| hotpotqa    | react_vote@k*(plan_exec)      |       300 | 0.4936 |      0.4431 |      0.5444 | nan      |     12766.6389 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| hotpotqa    | react_vote@k*(reflexion_gt)   |       300 | 0.5160 |      0.4632 |      0.5689 | nan      |     34936.8433 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |
| hotpotqa    | react_vote@k*(reflexion_self) |       300 | 0.5037 |      0.4532 |      0.5553 | nan      |     18521.7667 |      nan      |      nan      |        nan      |         nan      |         nan      |          nan      |

## Table 2 — the nine pre-registered comparisons (Holm-corrected per benchmark)

| benchmark   | family            | arm_a          | arm_b                         |   delta |   ci_lo |   ci_hi |   bootstrap_p |   holm_p |   mcnemar_p |
|:------------|:------------------|:---------------|:------------------------------|--------:|--------:|--------:|--------------:|---------:|------------:|
| gsm8k       | X_vs_react        | rewoo          | react                         | -0.2500 | -0.2889 | -0.2133 |        0.0010 |   0.0090 |      0.0000 |
| gsm8k       | X_vs_react        | plan_exec      | react                         | -0.0300 | -0.0478 | -0.0144 |        0.0010 |   0.0090 |      0.0923 |
| gsm8k       | X_vs_react        | reflexion_gt   | react                         |  0.0267 |  0.0133 |  0.0433 |        0.0010 |   0.0090 |      0.0312 |
| gsm8k       | X_vs_react        | reflexion_self | react                         |  0.0078 | -0.0000 |  0.0178 |        0.0920 |   0.1839 |      1.0000 |
| gsm8k       | X_vs_matched_vote | rewoo          | react_vote@k*(rewoo)          | -0.2543 | -0.2928 | -0.2190 |        0.0010 |   0.0090 |    nan      |
| gsm8k       | X_vs_matched_vote | plan_exec      | react_vote@k*(plan_exec)      | -0.0395 | -0.0571 | -0.0241 |        0.0010 |   0.0090 |    nan      |
| gsm8k       | X_vs_matched_vote | reflexion_gt   | react_vote@k*(reflexion_gt)   |  0.0218 |  0.0103 |  0.0365 |        0.0010 |   0.0090 |    nan      |
| gsm8k       | X_vs_matched_vote | reflexion_self | react_vote@k*(reflexion_self) |  0.0029 | -0.0046 |  0.0111 |        0.4578 |   0.4578 |    nan      |
| gsm8k       | oracle_premium    | reflexion_gt   | reflexion_self                |  0.0189 |  0.0078 |  0.0333 |        0.0010 |   0.0090 |      0.0625 |
| hotpotqa    | X_vs_react        | rewoo          | react                         | -0.0400 | -0.0767 | -0.0033 |        0.0310 |   0.1239 |      0.1934 |
| hotpotqa    | X_vs_react        | plan_exec      | react                         |  0.0244 | -0.0044 |  0.0545 |        0.1029 |   0.2059 |      0.5224 |
| hotpotqa    | X_vs_react        | reflexion_gt   | react                         |  0.0833 |  0.0600 |  0.1089 |        0.0010 |   0.0090 |      0.0000 |
| hotpotqa    | X_vs_react        | reflexion_self | react                         |  0.0433 |  0.0266 |  0.0633 |        0.0010 |   0.0090 |      0.0018 |
| hotpotqa    | X_vs_matched_vote | rewoo          | react_vote@k*(rewoo)          | -0.0380 | -0.0737 | -0.0022 |        0.0410 |   0.1239 |    nan      |
| hotpotqa    | X_vs_matched_vote | plan_exec      | react_vote@k*(plan_exec)      |  0.0164 | -0.0119 |  0.0448 |        0.2759 |   0.2759 |    nan      |
| hotpotqa    | X_vs_matched_vote | reflexion_gt   | react_vote@k*(reflexion_gt)   |  0.0529 |  0.0307 |  0.0753 |        0.0010 |   0.0090 |    nan      |
| hotpotqa    | X_vs_matched_vote | reflexion_self | react_vote@k*(reflexion_self) |  0.0252 |  0.0093 |  0.0420 |        0.0010 |   0.0090 |    nan      |
| hotpotqa    | oracle_premium    | reflexion_gt   | reflexion_self                |  0.0400 |  0.0211 |  0.0600 |        0.0010 |   0.0090 |      0.0044 |

## H1 — budget matching (RQ1)

| benchmark   | scaffold       |   k_star | k_star_beyond_pool   |   matched_tokens |   delta_default |   delta_default_ci_lo |   delta_default_ci_hi |   delta_matched |   delta_matched_ci_lo |   delta_matched_ci_hi | h1_supported   |
|:------------|:---------------|---------:|:---------------------|-----------------:|----------------:|----------------------:|----------------------:|----------------:|----------------------:|----------------------:|:---------------|
| gsm8k       | plan_exec      |   2.3482 | False                |        5696.1556 |         -0.0300 |               -0.0478 |               -0.0144 |         -0.0395 |               -0.0571 |               -0.0241 |                |
| gsm8k       | reflexion_gt   |   1.2092 | False                |        2934.2911 |          0.0267 |                0.0133 |                0.0433 |          0.0218 |                0.0103 |                0.0365 | False          |
| gsm8k       | reflexion_self |   1.2060 | False                |        2926.4167 |          0.0078 |               -0.0000 |                0.0178 |          0.0029 |               -0.0046 |                0.0111 | True           |
| gsm8k       | rewoo          |   1.0000 | False                |        2427.4671 |         -0.2500 |               -0.2889 |               -0.2133 |         -0.2543 |               -0.2928 |               -0.2190 |                |
| hotpotqa    | plan_exec      |   1.5782 | False                |       12766.6389 |          0.0244 |               -0.0044 |                0.0545 |          0.0164 |               -0.0119 |                0.0448 | False          |
| hotpotqa    | reflexion_gt   |   4.3149 | False                |       34936.8433 |          0.0833 |                0.0600 |                0.1089 |          0.0529 |                0.0307 |                0.0753 | False          |
| hotpotqa    | reflexion_self |   2.2885 | False                |       18521.7667 |          0.0433 |                0.0266 |                0.0633 |          0.0252 |                0.0093 |                0.0420 | False          |
| hotpotqa    | rewoo          |   1.0000 | False                |        8085.9635 |         -0.0400 |               -0.0767 |               -0.0033 |         -0.0380 |               -0.0737 |               -0.0022 |                |

## H2 — the oracle premium (RQ2)

| benchmark   |   n_judged_trials |   true_accept |   false_accept |   true_reject |   false_reject |   false_accept_rate |   false_reject_rate |   judge_accuracy |
|:------------|------------------:|--------------:|---------------:|--------------:|---------------:|--------------------:|--------------------:|-----------------:|
| gsm8k       |               909 |           866 |             34 |             3 |              6 |              0.9189 |              0.0069 |           0.9560 |
| hotpotqa    |               859 |           475 |            347 |            32 |              5 |              0.9156 |              0.0104 |           0.5902 |

## Table 3 / H3 — cost ratios by cost measure (RQ3)

| benchmark   | scaffold   |   ratio_tokens |   ratio_ideal_1024 |   ratio_ideal_4096 |   ratio_empirical | h3_supported   |
|:------------|:-----------|---------------:|-------------------:|-------------------:|------------------:|:---------------|
| gsm8k       | rewoo      |         0.3582 |             0.4367 |             0.4304 |            0.6297 | True           |
| hotpotqa    | rewoo      |         0.2130 |             2.1629 |             0.2639 |            1.0013 | False          |

## Table 4 — failure modes

| benchmark   | scaffold       |   n_runs |   halt_rate |   invalid_action_rate |   mean_invalid_actions |   could_not_find_rate |   plan_parse_ok_rate |   mean_trials |   judge_parse_fail_rate |   empty_answer_rate |
|:------------|:---------------|---------:|------------:|----------------------:|-----------------------:|----------------------:|---------------------:|--------------:|------------------------:|--------------------:|
| gsm8k       | react          |     1800 |      0.0050 |                0.0267 |                 0.0267 |                0.0000 |             nan      |      nan      |                nan      |              0.0050 |
| gsm8k       | rewoo          |      900 |      0.0000 |                0.0000 |                 0.0000 |                0.0000 |               0.9522 |      nan      |                nan      |              0.0000 |
| gsm8k       | plan_exec      |      900 |      0.0067 |                0.0000 |                 0.0000 |                0.0000 |               1.0000 |      nan      |                nan      |              0.0000 |
| gsm8k       | reflexion_gt   |      900 |      0.0000 |                0.0000 |                 0.0000 |                0.0000 |             nan      |        1.1200 |                  0.0000 |              0.0000 |
| gsm8k       | reflexion_self |      900 |      0.0000 |                0.0000 |                 0.0000 |                0.0000 |             nan      |        1.0167 |                  0.0011 |              0.0000 |
| hotpotqa    | react          |     3000 |      0.1733 |                0.0877 |                 0.0907 |                0.3957 |             nan      |      nan      |                nan      |              0.1733 |
| hotpotqa    | rewoo          |      900 |      0.0000 |                0.0000 |                 0.0000 |                0.5900 |               0.9967 |      nan      |                nan      |              0.0000 |
| hotpotqa    | plan_exec      |      900 |      0.3544 |                0.0000 |                 0.0000 |                0.4600 |               1.0000 |      nan      |                nan      |              0.0000 |
| hotpotqa    | reflexion_gt   |      900 |      0.1433 |                0.0000 |                 0.0000 |                0.4878 |             nan      |        2.8578 |                  0.0000 |              0.1433 |
| hotpotqa    | reflexion_self |      900 |      0.0822 |                0.0000 |                 0.0000 |                0.4111 |             nan      |        1.5144 |                  0.0000 |              0.0822 |

## H4 — predictability (RQ4)

```json
{
  "baseline_auc": 0.682106289614822,
  "best_model": "first_attempt_plus_agreement",
  "best_auc": 0.9594100438810337,
  "best_auc_ci_lo": 0.9417791571117362,
  "best_auc_ci_hi": 0.9743231782535907,
  "h4_supported": true
}
```

Figures: `fig1_pareto`, `fig2_reflexion`, `fig3_cost_measures`, `fig4_budget_curve`, `fig5_rq4`, `fig6_rq1_gains`, and the exploratory `fig7_rq3_threshold_sweep` (PNG + PDF in this directory).
