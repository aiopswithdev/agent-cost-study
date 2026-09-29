import pytest

from acs.grading import (
    grade,
    gsm8k_correct,
    gsm8k_gold,
    gsm8k_pred,
    hotpot_em,
    hotpot_f1,
    normalize_answer,
)


def test_em_accepts_article_and_case_differences():
    assert hotpot_em("The Saimaa Gesture", "Saimaa Gesture") == 1
    assert hotpot_em("arthur's magazine", "Arthur's Magazine") == 1
    assert hotpot_em("First for Women", "Arthur's Magazine") == 0


def test_f1_yes_no_special_case():
    assert hotpot_f1("yes", "no") == 0
    assert hotpot_f1("yes", "yes") == 1.0
    # A yes/no gold with a verbose prediction scores zero, per the official rule.
    assert hotpot_f1("the answer is yes", "yes") == 0


def test_f1_partial_overlap():
    f1 = hotpot_f1("director screenwriter", "director, screenwriter, actor")
    assert 0 < f1 < 1


def test_normalize_strips_punctuation_and_articles():
    assert normalize_answer("The  Saimaa, Gesture!") == "saimaa gesture"


def test_gsm8k_pred_takes_the_last_number_without_commas():
    assert gsm8k_pred("The answer is $1,234.") == 1234
    assert gsm8k_pred("First 5, then 10, so 72") == 72
    assert gsm8k_pred("-3.5") == -3.5
    assert gsm8k_pred("no digits here") is None


def test_gsm8k_gold_and_correct():
    assert gsm8k_gold("reasoning\n#### 72") == 72
    assert gsm8k_gold("blah\n#### 1,234") == 1234
    assert gsm8k_correct("72", "reasoning\n#### 72") == 1
    assert gsm8k_correct("71", "reasoning\n#### 72") == 0
    assert gsm8k_correct("", "reasoning\n#### 72") == 0


def test_grade_dispatch():
    assert grade("hotpotqa", "Saimaa Gesture", "The Saimaa Gesture") == {
        "em": 1, "f1": 1.0, "correct": 1
    }
    g = grade("gsm8k", "the answer is 72", "#### 72")
    assert g["em"] == g["correct"] == 1 and g["f1"] == 1.0
    with pytest.raises(ValueError):
        grade("nope", "a", "b")
