"""T2 — every constant of the study lives here. Never hard-code these elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
WIKI_CACHE_PATH = CACHE_DIR / "wiki.sqlite"

# --------------------------------------------------------------------------- §2.1 model
MODEL = "claude-haiku-4-5"

#: USD per 1M tokens.
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00
PRICE_CACHE_WRITE_PER_MTOK = 1.25  # 5-minute cache write, 1.25x input
PRICE_CACHE_READ_PER_MTOK = 0.10  # 0.1x input

#: Minimum cacheable prefix length. Shorter prefixes silently do not cache.
CACHE_MIN_TOKENS = {"claude-haiku-4-5": 4096}

#: Counterfactual minimum-cache-length regimes for ``usd_ideal[L]`` (§2.6).
IDEAL_CACHE_L = (0, 1024, 4096)

TEMPERATURE = 0.7  # every generative role
TEMPERATURE_JUDGE = 0.0  # the judge only

MAX_TOKENS = {
    "actor": 512,
    "planner": 1024,
    "executor": 512,
    "solver": 256,
    "worker_llm": 256,
    "reflector": 512,
    "judge": 256,
}

ROLES = tuple(MAX_TOKENS)

# --------------------------------------------------------------------------- §2.3/2.4 scaffolds
REACT_MAX_STEPS = 7
PE_MAX_PLAN_STEPS = 5
PE_MAX_EXEC_STEPS = 4
REWOO_MAX_EVIDENCE = 8
REFLEXION_MAX_TRIALS = 5
REFLEXION_MEMORY = 3
OBS_MAX_CHARS = 1500
SCRATCHPAD_MAX_CHARS = 6000  # reflector input truncation (T6-I)

SCAFFOLDS = ("react", "rewoo", "plan_exec", "reflexion_gt", "reflexion_self")
OTHER_SCAFFOLDS = ("rewoo", "plan_exec", "reflexion_gt", "reflexion_self")
BENCHMARKS = ("hotpotqa", "gsm8k")

POOL_K = 10  # ReAct pool replicates (the vote@k control)
R = 3  # replicates for every other scaffold
N = 300  # tasks per benchmark
PILOT_N = 30  # first 30 IDs of the frozen sample
SEED = 2026
BOOT_B = 2000  # cluster bootstrap resamples
VOTE_DRAWS = 200  # draws per k in the vote@k estimator (T14.3)

# --------------------------------------------------------------------------- concurrency / guards
MAX_RUNS_IN_FLIGHT = 32
LLM_CONCURRENCY = 24
WIKI_CONCURRENCY = 8
MAX_USD = 250.0
LLM_MAX_RETRIES = 8
LLM_TIMEOUT_S = 120.0

# --------------------------------------------------------------------------- tools
WIKI_ENDPOINT = "https://en.wikipedia.org/w/api.php"
WIKI_CONTACT_EMAIL = "aapkamohallamumbai@gmail.com"
WIKI_USER_AGENT = f"agent-cost-study/0.1 (academic research; contact: {WIKI_CONTACT_EMAIL})"
WIKI_MAX_RETRIES = 3

TOOLS_BY_BENCHMARK = {
    "hotpotqa": ("Search", "Lookup"),
    "gsm8k": ("Calculate",),
}
#: ReWOO additionally gets the ``LLM[...]`` worker (part of the ReWOO paradigm, §2.3).
REWOO_EXTRA_TOOLS = ("LLM",)

# --------------------------------------------------------------------------- data (§2.2)
HF_PARQUET_URLS = {
    "hotpotqa_validation": (
        "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet"
    ),
    "gsm8k_test": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet",
    "gsm8k_train": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train/0.parquet",
}


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key


def run_dir(run_tag: str) -> Path:
    return RESULTS_DIR / run_tag


def snapshot() -> dict:
    """Every module-level constant, for ``results/{run_tag}/config.json`` (§4.4)."""
    out = {}
    for name, value in sorted(globals().items()):
        if name.startswith("_") or not name.isupper():
            continue
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = {k: (str(v) if isinstance(v, Path) else v) for k, v in value.items()}
        out[name] = value
    return out
