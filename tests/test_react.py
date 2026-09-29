import json

from acs import config
from acs.llm import CallLog
from acs.prompts import library as lib
from acs.scaffolds.base import parse_action, truncate_after_first_action
from acs.scaffolds.react import react_loop, run_react
from tests.conftest import make_ctx


def test_parse_action_variants():
    assert parse_action("Thought 1: x\nAction 1: Search[Colorado orogeny]") == (
        "Search", "Colorado orogeny"
    )
    assert parse_action("Action: lookup[eastern sector]") == ("Lookup", "eastern sector")
    assert parse_action("Action 3: Finish[Arthur's Magazine [1844]]") == (
        "Finish", "Arthur's Magazine [1844]"
    )
    assert parse_action("Action 1: Calculate[48 / 2]") == ("Calculate", "48 / 2")
    assert parse_action("Thought 2: I am just thinking") is None
    assert parse_action("Action 1: Search[unclosed") is None
    assert parse_action("") is None


def test_parse_action_takes_the_first_action_only():
    text = "Action 1: Search[A]\nObservation 1: x\nAction 2: Finish[B]"
    assert parse_action(text) == ("Search", "A")


def test_truncate_after_first_action():
    text = "Thought 1: a\nAction 1: Search[X]\nObservation 1: fake\nThought 2: b"
    assert truncate_after_first_action(text) == "Thought 1: a\nAction 1: Search[X]"
    assert truncate_after_first_action("Thought only") == "Thought only"


async def test_three_step_episode(tmp_path, fake_wiki):
    log = CallLog(tmp_path / "calls.jsonl")
    ctx = make_ctx(
        [
            "Thought 1: search the first.\nAction 1: Search[Arthur's Magazine]",
            "Thought 2: search the second.\nAction 2: Search[First for Women]",
            "Thought 3: 1844 < 1989.\nAction 3: Finish[Arthur's Magazine]",
        ],
        wiki=fake_wiki,
        call_log=log,
    )
    result = await run_react(ctx)

    assert result.final_answer == "Arthur's Magazine"
    assert result.halted is False and result.n_steps == 3
    assert result.n_llm_calls == 3
    assert result.n_tool_calls["Search"] == 2 and result.n_tool_calls["Lookup"] == 0

    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["turn_index"] for r in rows] == [0, 1, 2]
    assert {r["conversation_id"] for r in rows} == {f"{ctx.run_id}:react"}
    assert {r["role"] for r in rows} == {"actor"}
    assert result.tokens_in == sum(r["input_tokens"] for r in rows)
    assert result.tokens_out == sum(r["output_tokens"] for r in rows)


async def test_conversation_is_append_only(fake_wiki):
    """Call t's messages must be call t-1's messages plus two new turns (§4.1)."""
    ctx = make_ctx(
        [
            "Thought 1: a\nAction 1: Search[Arthur's Magazine]",
            "Thought 2: b\nAction 2: Search[First for Women]",
            "Thought 3: c\nAction 3: Finish[x]",
        ],
        wiki=fake_wiki,
    )
    await run_react(ctx)
    sent = [r["messages"] for r in ctx.llm.requests]
    assert [len(m) for m in sent] == [1, 3, 5]
    for prev, cur in zip(sent, sent[1:]):
        prev_text = [_text(m) for m in prev]
        cur_text = [_text(m) for m in cur]
        assert cur_text[: len(prev_text)] == prev_text


def _text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content)


async def test_invalid_action_produces_the_error_observation():
    ctx = make_ctx(
        ["Thought 1: just thinking", "Thought 2: ok\nAction 2: Finish[42]"],
    )
    outcome = await react_loop(
        ctx,
        system=lib.react_system("hotpotqa"),
        system_prompt_id="react_hotpotqa_v1",
        first_user_msg="Question: q",
        conversation_label="react",
    )
    assert outcome.answer == "42"
    assert "Invalid action. Valid actions are Search[...], Lookup[...], Finish[...]." in (
        outcome.transcript
    )
    # The invalid action still consumes a step, as in ReAct's own loop.
    assert outcome.n_steps == 2


async def test_halting_after_max_steps():
    ctx = make_ctx(["Thought: no action"] * config.REACT_MAX_STEPS)
    result = await run_react(ctx)
    assert result.halted is True
    assert result.final_answer == ""
    assert result.n_steps == config.REACT_MAX_STEPS
    assert result.n_llm_calls == config.REACT_MAX_STEPS


async def test_multi_action_output_is_truncated_to_the_first_action(fake_wiki):
    ctx = make_ctx(
        [
            "Thought 1: a\nAction 1: Search[Arthur's Magazine]\n"
            "Observation 1: hallucinated\nThought 2: b\nAction 2: Finish[wrong]",
            "Thought 2: real\nAction 2: Finish[right]",
        ],
        wiki=fake_wiki,
    )
    result = await run_react(ctx)
    assert result.final_answer == "right"
    stored = ctx.llm.requests[1]["messages"][1]["content"]
    assert stored == "Thought 1: a\nAction 1: Search[Arthur's Magazine]"


async def test_empty_model_output_is_replaced_by_a_placeholder():
    ctx = make_ctx(["", "Thought: x\nAction: Finish[1]"])
    result = await run_react(ctx)
    assert result.final_answer == "1"
    assert "(no output)" in ctx.llm.requests[1]["messages"][1]["content"]


async def test_gsm8k_react_uses_the_calculator():
    ctx = make_ctx(
        ["Thought 1: divide.\nAction 1: Calculate[48 / 2]",
         "Thought 2: done.\nAction 2: Finish[24]"],
        benchmark="gsm8k",
    )
    result = await run_react(ctx)
    assert result.final_answer == "24"
    assert result.n_tool_calls["Calculate"] == 1
    assert "Observation 1: 24" in result.extra["transcript"]
