"""T4 — the frozen samples (§2.2). Skipped when the raw parquet files are absent."""

from __future__ import annotations

import pandas as pd
import pytest

from acs import config
from acs.data import RAW_FILES, Task, load_tasks, make_samples, sample_path
from acs.prompts.library import exemplar_questions

RAW_PRESENT = all((config.RAW_DIR / f).exists() for f in RAW_FILES.values())
SAMPLES_PRESENT = all(sample_path(b).exists() for b in config.BENCHMARKS)

needs_raw = pytest.mark.skipif(not RAW_PRESENT, reason="run `python -m acs.data download`")
needs_samples = pytest.mark.skipif(
    not SAMPLES_PRESENT, reason="run `python -m acs.data sample`"
)


@needs_samples
@pytest.mark.parametrize("benchmark", config.BENCHMARKS)
def test_sample_has_n_unique_tasks(benchmark):
    tasks = load_tasks(benchmark)
    assert len(tasks) == config.N
    assert len({t.task_id for t in tasks}) == config.N
    assert all(t.question.strip() and t.answer.strip() for t in tasks)


@needs_samples
@needs_raw
def test_hotpotqa_type_proportions_match_the_split_within_2_percent():
    split = pd.read_parquet(
        config.RAW_DIR / RAW_FILES["hotpotqa_validation"]
    )["type"].value_counts(normalize=True)
    sample = pd.Series([t.type for t in load_tasks("hotpotqa")]).value_counts(normalize=True)
    for kind, p in split.items():
        assert abs(sample.get(kind, 0.0) - p) < 0.02, kind


@needs_samples
@pytest.mark.parametrize("benchmark", config.BENCHMARKS)
def test_no_sampled_question_is_a_fewshot_exemplar(benchmark):
    banned = {q.strip().lower() for q in exemplar_questions(benchmark)}
    assert not {t.question.strip().lower() for t in load_tasks(benchmark)} & banned


@needs_samples
@pytest.mark.parametrize("benchmark", config.BENCHMARKS)
def test_pilot_is_the_first_thirty_ids_in_file_order(benchmark):
    full = load_tasks(benchmark)
    pilot = load_tasks(benchmark, pilot=True)
    assert len(pilot) == config.PILOT_N
    assert [t.task_id for t in pilot] == [t.task_id for t in full[: config.PILOT_N]]


@needs_samples
def test_gsm8k_task_ids_are_row_indexed():
    for task in load_tasks("gsm8k"):
        assert task.task_id.startswith("gsm8k_test_")
        assert task.task_id.split("_")[-1].isdigit()


@needs_samples
def test_gsm8k_answers_keep_the_gold_marker():
    from acs.grading import gsm8k_gold

    for task in load_tasks("gsm8k")[:20]:
        assert "####" in task.answer
        assert gsm8k_gold(task.answer) == gsm8k_gold(task.answer)


@needs_samples
def test_sampling_refuses_to_overwrite_a_frozen_sample():
    with pytest.raises(FileExistsError):
        make_samples()


@needs_raw
@pytest.mark.parametrize("benchmark", config.BENCHMARKS)
def test_sampling_is_reproducible_from_the_seed(benchmark):
    from acs.data import sample_gsm8k, sample_hotpotqa

    fn = sample_hotpotqa if benchmark == "hotpotqa" else sample_gsm8k
    a = [t.task_id for t in fn(n=50, seed=config.SEED)]
    b = [t.task_id for t in fn(n=50, seed=config.SEED)]
    c = [t.task_id for t in fn(n=50, seed=config.SEED + 1)]
    assert a == b and a != c


def test_task_serialises_without_a_type_for_gsm8k():
    assert "type" not in Task("g", "q", "a", "gsm8k").to_dict()
    assert Task("h", "q", "a", "hotpotqa", "bridge").to_dict()["type"] == "bridge"
