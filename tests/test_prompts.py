import re

import pytest

from acs import config
from acs.prompts import library as lib
from acs.tools.calc import calculate

PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")
CALC_OBS_RE = re.compile(r"Calculate\[(.+?)\]\n(?:Observation \d+: )?(.+)")


@pytest.mark.parametrize("prompt_id,text", sorted(lib.all_system_prompts().items()))
def test_every_system_prompt_is_fully_built(prompt_id, text):
    assert text.strip()
    assert not PLACEHOLDER_RE.search(text), f"{prompt_id} still has a placeholder"


def test_all_system_prompt_ids_are_the_expected_set():
    expected = {"worker_llm_v1"}
    for b in config.BENCHMARKS:
        expected |= {
            f"react_{b}_v1", f"rewoo_planner_{b}_v1", f"solver_{b}_v1",
            f"pe_planner_{b}_v1", f"pe_executor_{b}_v1", f"reflector_{b}_v1",
            f"judge_{b}_v1",
        }
    assert set(lib.all_system_prompts()) == expected


def test_react_hotpotqa_contains_the_vendored_fewshot_and_answer_style():
    system = lib.react_system("hotpotqa")
    assert "Search[entity]" in system and "Lookup[keyword]" in system
    assert lib.WEBTHINK_SIMPLE6 in system
    assert "Thought 1:" in system and "Action 1: Search[Colorado orogeny]" in system
    assert system.endswith(lib.ANSWER_STYLE["hotpotqa"])
    assert "(END OF EXAMPLES)" in system


def test_react_gsm8k_prompt_shape():
    system = lib.react_system("gsm8k")
    assert "Calculate[expression]" in system
    assert "Search[" not in system
    assert system.endswith(lib.ANSWER_STYLE["gsm8k"])


@pytest.mark.parametrize("builder", [lib.react_system, lib.rewoo_planner_system])
def test_gsm8k_exemplar_calculate_observations_are_real_calculator_output(builder):
    """Every `Calculate[expr]` in a GSM8K few-shot is followed by `calculate(expr)`."""
    text = builder("gsm8k")
    lines = text.split("\n")
    checked = 0
    for i, line in enumerate(lines):
        m = re.search(r"Calculate\[(.+)\]", line)
        if not m or i + 1 >= len(lines):
            continue
        expr = m.group(1)
        if "#E" in expr:  # ReWOO evidence references are resolved at run time
            continue
        nxt = lines[i + 1]
        obs = re.match(r"Observation \d+: (.+)", nxt)
        if obs is None:
            continue
        assert obs.group(1).strip() == calculate(expr), f"{expr!r} -> {obs.group(1)!r}"
        checked += 1
    if builder is lib.react_system:
        assert checked == 4  # two in the first exemplar, one each in the other two


def test_rewoo_hotpotqa_fewshot_renames_wikipedia_to_search():
    system = lib.rewoo_planner_system("hotpotqa")
    assert "Wikipedia[" not in system
    assert "#E1 = Search[Colorado orogeny]" in system
    assert lib.REWOO_PLANNER_PREFIX in system and lib.REWOO_PLANNER_SUFFIX in system
    assert "LLM[input]:" in system and "Search[input]:" in system


def test_rewoo_gsm8k_planner_exposes_only_calculate_and_llm():
    system = lib.rewoo_planner_system("gsm8k")
    assert "Calculate[input]:" in system and "LLM[input]:" in system
    assert "Search[input]:" not in system and "Lookup[input]:" not in system


def test_solver_is_shared_and_carries_the_vendored_suffix():
    for b in config.BENCHMARKS:
        assert lib.solver_system(b).startswith(lib.REWOO_SOLVER_PREFIX)
        assert lib.solver_system(b).endswith(lib.ANSWER_STYLE[b])
    user = lib.solver_user("Q?", [("p1", "e1"), ("p2", "e2")])
    assert user.startswith("Plan: p1\nEvidence: e1\nPlan: p2\nEvidence: e2\n")
    assert lib.REWOO_SOLVER_SUFFIX in user and user.endswith("Q?")


def test_pe_executor_is_the_react_prompt_without_answer_style():
    for b in config.BENCHMARKS:
        executor = lib.pe_executor_system(b)
        assert executor.startswith(lib.react_body(b))
        assert lib.ANSWER_STYLE[b] not in executor
        assert "ONE step of a larger plan" in executor


def test_pe_executor_user_shows_completed_step_results():
    plan = ["Find A.", "Find B."]
    first = lib.pe_executor_user("Q?", plan, [], 1)
    assert "Results of completed steps:\n(none)" in first
    assert first.endswith("Current step 1: Find A.")
    second = lib.pe_executor_user("Q?", plan, ["A is 3"], 2)
    assert "Step 1 result: A is 3" in second
    assert second.endswith("Current step 2: Find B.")


def test_reflector_prompt_is_the_vendored_head_only():
    hotpot = lib.reflector_system("hotpotqa")
    assert "Previous trial:" not in hotpot
    assert "advanced reasoning agent" in hotpot
    assert lib.REFLECTIONS_FEWSHOT in hotpot
    gsm8k = lib.reflector_system("gsm8k")
    assert "Docstore" not in gsm8k and "calculator tool" in gsm8k
    assert lib.REFLECTIONS_FEWSHOT not in gsm8k


def test_reflector_user_shape_and_scratchpad_truncation():
    user = lib.reflector_user("Q?", "Thought 1: a\nAction 1: Finish[x]")
    assert user.startswith("Previous trial:\nQuestion: Q?\n")
    assert user.endswith("Reflection:")

    long_pad = "\n".join(
        ["Thought 1: short"] + [f"Observation {i}: " + "x" * 3000 for i in range(5)]
    )
    out = lib.truncate_scratchpad(long_pad)
    assert len(out) <= config.SCRATCHPAD_MAX_CHARS
    assert "Thought 1: short" in out


def test_format_reflections_matches_upstream():
    assert lib.format_reflections([]) == ""
    out = lib.format_reflections(["a", "b"])
    assert out.startswith(lib.REFLECTION_HEADER)
    assert out.endswith("Reflections:\n- a\n- b")


def test_judge_prompt_and_verdict_parsing():
    assert "You do NOT know the correct answer" in lib.JUDGE_SYSTEM
    user = lib.judge_user("Q?", "Thought 1: x", "42")
    assert user.startswith("Question: Q?") and user.endswith("Final answer: 42")

    assert lib.parse_verdict("Looks right.\nVERDICT: CORRECT") == ("CORRECT", False)
    assert lib.parse_verdict("VERDICT: CORRECT\nactually VERDICT: INCORRECT") == (
        "INCORRECT", False
    )
    assert lib.parse_verdict("no verdict at all") == ("INCORRECT", True)


def test_exemplar_questions_are_collected_for_sample_exclusion():
    hotpot = lib.exemplar_questions("hotpotqa")
    assert any("Colorado orogeny" in q for q in hotpot)
    assert any("Rome Protocols" in q for q in hotpot)  # from the Reflexion few-shot
    gsm8k = lib.exemplar_questions("gsm8k")
    assert len(gsm8k) == 3 and any("Natalia sold clips" in q for q in gsm8k)


def test_vendor_files_are_unmodified_upstream_strings():
    assert lib.REWOO_PLANNER_PREFIX.startswith("For the following tasks, make plans")
    assert lib.REWOO_SOLVER_SUFFIX.strip().endswith("Respond with the answer directly with no extra words.")
    assert lib.REFLECTION_HEADER.startswith("You have attempted to answer following question")
    assert "Wikipedia[" in lib.vendored("rewoo_fewshots.py", "HOTPOTQA_PWS_BASE")
