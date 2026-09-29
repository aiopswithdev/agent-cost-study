# agent-cost-study

A budget-matched, oracle-free comparison of LLM agent scaffolds.

One model (`deepseek-flash`, thinking disabled — switched from `claude-haiku-4-5` before any
run, see `docs/DEVIATIONS.md`), one tool set, one task set. Five scaffolds — **ReAct,
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
cd agent-cost-study
uv venv --python 3.12 && source .venv/bin/activate   # skip if .venv already exists
uv pip install -e ".[dev]"
```

Put your DeepSeek key in `.env` (git-ignored; `.env.example` shows the format):

```
DEEPSEEK_API_KEY=sk-...
ACS_PROVIDER=deepseek
```

**Budget.** The spend guard stops launching runs once the *billed* amount of a run tag reaches
`--max-usd` (default `MAX_USD = 4.50`). DeepSeek bills **half price off-peak**; peak is
**01:00–04:00 and 06:00–10:00 UTC, Mon–Fri** (06:30–09:30 and 11:30–15:30 IST). Launch long
runs outside those hours. Check your balance at platform.deepseek.com before each launch.

## Run protocol

Always run from `agent-cost-study/` with the venv active.

```bash
# 0. Tests. Offline suite (free), then one live DeepSeek call + live Wikipedia (< $0.001).
pytest -m "not network"
pytest -m network

# 1. Data — already downloaded and frozen in data/samples/. Only if missing:
python -m acs.data download && python -m acs.data sample

# 2. Offline self-test of the whole pipeline (no API calls, no spend).
python -m acs.runner --run-tag selftest --benchmarks gsm8k --pilot --k 2 --r 1 --fake-llm

# 3. Pilot: 30 tasks per benchmark, pool K=3, R=1. Costs cents.
python -m acs.runner --run-tag pilot --pilot --k 3 --r 1 --max-usd 1.00
python -m acs.analysis.pilot --run-tag pilot        # -> results/pilot/PILOT_REPORT.md
```

Read `results/pilot/PILOT_REPORT.md`. Its **scope-down ladder** table prices every
configuration against the remaining budget and names the largest one that fits off-peak.
Record the decision in `docs/DEVIATIONS.md` (T12 entry), then launch that configuration.
Use the **same `--run-tag main` and the same `--limit` in every command** of a launch, and
set `--max-usd` to what is left of your balance, minus a small margin.

```bash
# 4a. As pre-registered (N=300, K=10, R=3):
python -m acs.runner --run-tag main --k 10 --r 3 --max-usd 4.00

# 4b. A scope-down configuration, e.g. "§6.3 step 4" at N=200 (K=6 both, R=2 ReWOO/Plan-Exec):
python -m acs.runner --run-tag main --limit 200 --stages pool --k 6 --max-usd 4.00
python -m acs.runner --run-tag main --limit 200 --stages others --r 3 --scaffolds reflexion_gt reflexion_self --max-usd 4.00
python -m acs.runner --run-tag main --limit 200 --stages others --r 2 --scaffolds rewoo plan_exec --max-usd 4.00
#     (§6.3 step 1 keeps HotpotQA K=10 but GSM8K K=6: run the pool stage once per benchmark,
#      `--benchmarks hotpotqa --k 10` and `--benchmarks gsm8k --k 6`.)

# 5. Every table and figure, from the JSONL logs alone.
python -m acs.analysis.report --run-tag main        # -> results/main/analysis/
```

The runner is **resumable**: re-run the identical command after any interruption; it skips run
IDs already logged with `status == "ok"` and retries errored ones. If DeepSeek reports an invalid
key (HTTP 401) or an empty balance (HTTP 402), or the guard trips, no further runs are started;
top up, then re-run the same command.

Stage `others` requires the pool runs with `r < R` to exist already, because Reflexion's
trial 1 *is* a ReAct pool trajectory.

## Layout

| Path | What |
|---|---|
| `src/acs/config.py` | every constant; nothing is hard-coded elsewhere |
| `src/acs/pricing.py` | per-model price sheets, the regimes of §2.6, and the actual bill (`usd_billed`) |
| `src/acs/llm.py` | the single gateway to the API: DeepSeek (default) and Anthropic clients, logging, spend guard, `FakeLLM` |
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
pytest -m network         # once: live Wikipedia and one live DeepSeek smoke call
```
