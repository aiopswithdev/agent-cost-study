# agent-cost-study

A budget-matched, oracle-free comparison of LLM agent scaffolds.

One model (`claude-haiku-4-5`), one tool set, one task set. Five scaffolds — **ReAct,
ReWOO, Plan-and-Execute, Reflexion-GT, Reflexion-Self** — on HotpotQA and GSM8K. ReAct is
additionally run 10× per task (the "pool"), which yields, post hoc and at no extra API
cost, the budget-matched control: **majority-vote-over-k ReAct** (`vote@k`) at any budget.

- **Why:** `../Lit-review/00_Literature_Review_and_Problem_Statement.md`
- **How:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — the frozen spec
- **Pre-registration:** [`PREREGISTRATION.md`](PREREGISTRATION.md) — frozen before the main runs
- **Every change from the plan:** [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)
- **Findings:** [`RESULTS.md`](RESULTS.md)

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # then put your ANTHROPIC_API_KEY in it
```

Before launching the main runs, record the organisation's RPM / ITPM / OTPM for
`claude-haiku-4-5` and the credit balance in `docs/DEVIATIONS.md`. The workload needs
roughly **≥ 150 RPM and ≥ 400k ITPM** to finish in ≤ 12 hours, and ≥ $300 of credit.

## Run protocol

```bash
# 1. Data — the samples are frozen once written and must never be resampled.
python -m acs.data download
python -m acs.data sample

# 2. Offline self-test: the whole pipeline, no API calls, no spend.
python -m acs.runner --run-tag selftest --benchmarks gsm8k --pilot --k 2 --r 1 --fake-llm

# 3. Pilot (30 tasks per benchmark), then the go/no-go decision.
python -m acs.runner --run-tag pilot --pilot --k 3 --r 1
python -m acs.analysis.pilot --run-tag pilot        # -> results/pilot/PILOT_REPORT.md

# 4. Main runs. Re-invoke the same command to resume after any interruption.
python -m acs.runner --run-tag main --no-pilot --k 10 --r 3

# 5. Every table and figure, from the JSONL logs alone.
python -m acs.analysis.report --run-tag main        # -> results/main/analysis/
```

The runner is **resumable**: it skips run IDs already logged with `status == "ok"` and
retries the errored ones. It also enforces a spend guard — once cumulative `usd_nocache`
passes `MAX_USD`, no new runs are launched (in-flight runs finish).

Stage `others` requires the pool runs with `r < R` to exist already, because Reflexion's
trial 1 *is* a ReAct pool trajectory.

## Layout

| Path | What |
|---|---|
| `src/acs/config.py` | every constant; nothing is hard-coded elsewhere |
| `src/acs/pricing.py` | the three pricing regimes of §2.6 |
| `src/acs/llm.py` | the single gateway to the API: cache breakpoints, logging, spend guard, `FakeLLM` |
| `src/acs/data.py`, `grading.py` | the frozen samples; official EM/F1 and GSM8K grading |
| `src/acs/tools/` | `Search`/`Lookup` (Wikipedia + disk cache), `Calculate`, per-run dispatch |
| `src/acs/prompts/vendor/` | upstream prompts, verbatim and read-only (see `PROVENANCE.md`) |
| `src/acs/prompts/library.py` | every prompt, built from the vendored files plus authored text |
| `src/acs/scaffolds/` | the five scaffolds; they differ **only** in control flow |
| `src/acs/runner.py` | the CLI |
| `src/acs/analysis/` | load, cost regimes, vote@k, statistics, RQ4, figures and tables |
| `results/{run_tag}/` | `calls.jsonl`, `runs.jsonl`, `config.json`, `prompt_tokens.json`, `analysis/` |

Field names in `calls.jsonl` and `runs.jsonl` are a contract (§4 of the plan); the analysis
depends on them.

## Tests

```bash
pytest -m "not network"   # offline: graders, calculator, prompts, all five scaffolds, analysis
pytest -m network         # once: live Wikipedia and one live Anthropic smoke call
```
