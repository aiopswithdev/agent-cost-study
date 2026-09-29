"""T4 — graders (§2.5). Post hoc only; never shown to an agent, except as the stopping
signal inside `reflexion_gt`, which is the point of that condition."""

from __future__ import annotations

import re
import string
from collections import Counter

# --------------------------------------------------------------------------- HotpotQA
# Normalisation and F1 follow the official `hotpot_evaluate_v1.py`.

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_answer(s: str) -> str:
    """lowercase, strip punctuation, remove articles, collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def hotpot_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def hotpot_f1(pred: str, gold: str) -> float:
    """Official F1, including the yes/no/noanswer special case."""
    normalized_prediction = normalize_answer(pred)
    normalized_ground_truth = normalize_answer(gold)

    ZERO_METRIC = 0.0

    if (
        normalized_prediction in ("yes", "no", "noanswer")
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ("yes", "no", "noanswer")
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# --------------------------------------------------------------------------- GSM8K


def gsm8k_gold(answer_field: str) -> float:
    """Gold = the number after `####`, commas removed.

    Also accepts an already-extracted gold number, which is what `runs.jsonl` stores.
    """
    tail = answer_field.split("####")[-1].strip().replace(",", "")
    m = _NUM_RE.search(tail)
    if m is None:
        raise ValueError(f"no gold number in GSM8K answer field: {answer_field[:80]!r}")
    return float(m.group(0).replace(",", ""))


def gsm8k_pred(text: str) -> float | None:
    """Prediction = the **last** number in the answer string, commas removed."""
    matches = _NUM_RE.findall(text or "")
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def gsm8k_correct(pred_text: str, gold: str | float) -> int:
    gold_value = gsm8k_gold(gold) if isinstance(gold, str) else float(gold)
    pred = gsm8k_pred(pred_text)
    if pred is None:
        return 0
    return int(abs(pred - gold_value) < 1e-6)


# --------------------------------------------------------------------------- dispatch


def grade(benchmark: str, pred: str, gold: str) -> dict:
    """Returns `{"em": int, "f1": float, "correct": int}`. For GSM8K, em/f1 == correct."""
    pred = pred or ""
    if benchmark == "hotpotqa":
        em = hotpot_em(pred, gold)
        return {"em": em, "f1": hotpot_f1(pred, gold), "correct": em}
    if benchmark == "gsm8k":
        c = gsm8k_correct(pred, gold)
        return {"em": c, "f1": float(c), "correct": c}
    raise ValueError(f"unknown benchmark {benchmark!r}")
