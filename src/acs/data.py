"""T4 — dataset download, the frozen N=300 samples (§2.2), and task loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from . import config

RAW_FILES = {
    "hotpotqa_validation": "hotpotqa_distractor_validation.parquet",
    "gsm8k_test": "gsm8k_main_test.parquet",
    "gsm8k_train": "gsm8k_main_train.parquet",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: str
    benchmark: str
    type: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.type is None:
            d.pop("type")
        return d


# --------------------------------------------------------------------------- download


def _fetch_parquet(url: str, dest: Path, client: httpx.Client) -> None:
    """The HF API URL either returns a JSON list of real parquet URLs or redirects to one."""
    resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        urls = resp.json()
        if isinstance(urls, dict):  # nested {config: {split: [urls]}}
            flat: list[str] = []
            for cfg in urls.values():
                for split in cfg.values():
                    flat.extend(split)
            urls = flat
        if not urls:
            raise RuntimeError(f"HF API returned no parquet URLs for {url}")
        resp = client.get(urls[0], follow_redirects=True)
        resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def download(force: bool = False) -> dict[str, Path]:
    """Fetch the three parquet files into `data/raw/` (skip files already present)."""
    out: dict[str, Path] = {}
    headers = {"User-Agent": config.WIKI_USER_AGENT}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for key, url in config.HF_PARQUET_URLS.items():
            dest = config.RAW_DIR / RAW_FILES[key]
            if dest.exists() and not force:
                out[key] = dest
                continue
            _fetch_parquet(url, dest, client)
            out[key] = dest
    return out


def _raw(key: str) -> pd.DataFrame:
    path = config.RAW_DIR / RAW_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m acs.data download` first")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- sampling


def _exemplar_questions(benchmark: str) -> set[str]:
    from .prompts.library import exemplar_questions  # lazy: avoids an import cycle

    return {q.strip().lower() for q in exemplar_questions(benchmark)}


def _stratified_counts(proportions: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder allocation so the strata sum to exactly `n`."""
    raw = {k: p * n for k, p in proportions.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def sample_hotpotqa(n: int = config.N, seed: int = config.SEED) -> list[Task]:
    df = _raw("hotpotqa_validation")
    banned = _exemplar_questions("hotpotqa")
    df = df[~df["question"].str.strip().str.lower().isin(banned)].reset_index(drop=True)

    props = df["type"].value_counts(normalize=True).to_dict()
    counts = _stratified_counts(props, n)

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for t in sorted(counts):
        idx = np.flatnonzero((df["type"] == t).to_numpy())
        chosen = rng.choice(idx, size=counts[t], replace=False)
        picked.extend(int(i) for i in chosen)
    rng.shuffle(picked)

    return [
        Task(
            task_id=str(df.at[i, "id"]),
            question=str(df.at[i, "question"]),
            answer=str(df.at[i, "answer"]),
            benchmark="hotpotqa",
            type=str(df.at[i, "type"]),
        )
        for i in picked
    ]


def sample_gsm8k(n: int = config.N, seed: int = config.SEED) -> list[Task]:
    df = _raw("gsm8k_test")
    banned = _exemplar_questions("gsm8k")
    keep = np.flatnonzero(~df["question"].str.strip().str.lower().isin(banned).to_numpy())

    rng = np.random.default_rng(seed)
    picked = rng.choice(keep, size=n, replace=False)
    rng.shuffle(picked)

    return [
        Task(
            task_id=f"gsm8k_test_{int(i)}",
            question=str(df.at[int(i), "question"]),
            answer=str(df.at[int(i), "answer"]),
            benchmark="gsm8k",
        )
        for i in picked
    ]


def sample_path(benchmark: str, n: int = config.N) -> Path:
    return config.SAMPLES_DIR / f"{benchmark}_N{n}.json"


def make_samples(n: int = config.N, seed: int = config.SEED, force: bool = False) -> dict[str, Path]:
    """Write the frozen sample files. Refuses to overwrite (§2.2: never resample)."""
    out: dict[str, Path] = {}
    for benchmark, fn in (("hotpotqa", sample_hotpotqa), ("gsm8k", sample_gsm8k)):
        path = sample_path(benchmark, n)
        if path.exists() and not force:
            raise FileExistsError(
                f"{path} already exists — the sample is frozen and must never be resampled. "
                "Pass force=True only to rebuild a sample that was never used for a run."
            )
        tasks = fn(n=n, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([t.to_dict() for t in tasks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        out[benchmark] = path
    return out


# --------------------------------------------------------------------------- loading


def load_tasks(benchmark: str, pilot: bool = False, n: int = config.N) -> list[Task]:
    """The frozen sample; `pilot=True` returns its first `PILOT_N` IDs in file order."""
    path = sample_path(benchmark, n)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m acs.data sample` first")
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks = [
        Task(
            task_id=r["task_id"],
            question=r["question"],
            answer=r["answer"],
            benchmark=benchmark,
            type=r.get("type"),
        )
        for r in rows
    ]
    return tasks[: config.PILOT_N] if pilot else tasks


def gsm8k_train_rows(k: int = 3) -> list[dict]:
    """The first `k` GSM8K train items — the source of the ReAct/ReWOO exemplars (T6)."""
    df = _raw("gsm8k_train")
    return [
        {"question": str(df.at[i, "question"]), "answer": str(df.at[i, "answer"])}
        for i in range(k)
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m acs.data")
    ap.add_argument("command", choices=["download", "sample", "show"])
    ap.add_argument("--n", type=int, default=config.N)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "download":
        for key, path in download(force=args.force).items():
            print(f"{key}: {path} ({path.stat().st_size:,} bytes)")
    elif args.command == "sample":
        for benchmark, path in make_samples(n=args.n, force=args.force).items():
            print(f"{benchmark}: {path}")
    else:
        for benchmark in config.BENCHMARKS:
            tasks = load_tasks(benchmark, n=args.n)
            print(f"{benchmark}: {len(tasks)} tasks; first = {tasks[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
