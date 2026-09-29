import json

import pytest

from acs.llm import (
    CallLog,
    FakeLLM,
    SpendLimitExceeded,
    build_request,
    validate_messages,
)
from tests.conftest import make_ctx


def _conversation(n_turns: int) -> list[dict]:
    msgs = [{"role": "user", "content": "Question: q"}]
    for i in range(n_turns - 1):
        msgs.append({"role": "assistant", "content": f"Action {i}: Search[x]"})
        msgs.append({"role": "user", "content": f"Observation {i}: obs"})
    return msgs


def test_cache_control_on_system_and_last_message_only():
    system_blocks, msgs = build_request("SYS", _conversation(5))
    assert system_blocks == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    # Exactly one breakpoint in the message list, on the last block of the last message.
    breakpoints = [
        (i, j)
        for i, m in enumerate(msgs)
        if isinstance(m["content"], list)
        for j, b in enumerate(m["content"])
        if "cache_control" in b
    ]
    assert breakpoints == [(len(msgs) - 1, 0)]
    # Earlier messages stay plain strings so breakpoints never accumulate.
    assert all(isinstance(m["content"], str) for m in msgs[:-1])
    assert len(breakpoints) + len(system_blocks) <= 4


def test_build_request_does_not_mutate_the_caller_messages():
    original = _conversation(3)
    snapshot = json.loads(json.dumps(original))
    build_request("SYS", original)
    assert original == snapshot


def test_build_request_strips_inherited_cache_control():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
    ]}]
    _, out = build_request("SYS", msgs)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "user", "content": "   "}],
    ],
)
def test_validate_messages_rejects_bad_shapes(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


async def test_fake_llm_round_trip_writes_a_correct_row(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl", max_usd=100.0)
    ctx = make_ctx(["hello there"], call_log=log)
    res = await ctx.call(
        role="actor",
        conversation_id=f"{ctx.run_id}:react",
        turn_index=0,
        system_prompt_id="react_hotpotqa_v1",
        system="SYS",
        messages=[{"role": "user", "content": "Question: q"}],
    )
    assert res.text == "hello there"

    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == ctx.run_id
    assert row["benchmark"] == "hotpotqa" and row["task_id"] == "t1"
    assert row["scaffold"] == "react" and row["replicate"] == 0
    assert row["role"] == "actor" and row["turn_index"] == 0
    assert row["system_prompt_id"] == "react_hotpotqa_v1"
    assert row["temperature"] == 0.7 and row["max_tokens"] == 512
    assert row["output_tokens"] > 0 and row["error"] is None
    assert set(row) >= {
        "call_id", "conversation_id", "model", "input_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens", "stop_reason",
        "latency_s", "ts", "attempts",
    }


async def test_judge_role_uses_temperature_zero(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    ctx = make_ctx(["VERDICT: CORRECT"], call_log=log)
    await ctx.call(
        role="judge",
        conversation_id=f"{ctx.run_id}:judge1",
        turn_index=0,
        system_prompt_id="judge_hotpotqa_v1",
        system="SYS",
        messages=[{"role": "user", "content": "q"}],
    )
    row = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert row["temperature"] == 0.0 and row["max_tokens"] == 256


async def test_fake_llm_honours_stop_sequences():
    llm = FakeLLM(["Thought 1: a\nAction 1: Search[x]\nObservation 1: leaked"])
    res = await llm.call(
        role="actor", conversation_id="c", turn_index=0, system_prompt_id="p",
        system="SYS", messages=[{"role": "user", "content": "q"}],
        max_tokens=512, temperature=0.7, stop=["\nObservation"],
    )
    assert "leaked" not in res.text


async def test_spend_guard_trips_and_blocks_further_calls(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl", max_usd=1e-9)
    ctx = make_ctx(["x" * 400, "y"], call_log=log)
    await ctx.call(
        role="actor", conversation_id="c", turn_index=0, system_prompt_id="p",
        system="SYS" * 500, messages=[{"role": "user", "content": "q"}],
    )
    assert log.stop is True and ctx.llm.stopped is True
    with pytest.raises(SpendLimitExceeded):
        await ctx.call(
            role="actor", conversation_id="c", turn_index=1, system_prompt_id="p",
            system="SYS", messages=[{"role": "user", "content": "q"}],
        )


async def test_spend_total_is_restored_from_an_existing_log(tmp_path):
    path = tmp_path / "calls.jsonl"
    log = CallLog(path, max_usd=100.0)
    ctx = make_ctx(["hello"], call_log=log)
    await ctx.call(
        role="actor", conversation_id="c", turn_index=0, system_prompt_id="p",
        system="SYS" * 100, messages=[{"role": "user", "content": "q"}],
    )
    resumed = CallLog(path, max_usd=100.0)
    assert resumed.n_calls == 1
    assert abs(resumed.total_usd_nocache - log.total_usd_nocache) < 1e-12


@pytest.mark.network
async def test_live_smoke_call_logs_usage(tmp_path):
    from acs import config
    from acs.llm import LLM

    log = CallLog(tmp_path / "calls.jsonl", max_usd=1.0)
    llm = LLM(call_log=log, concurrency=1)
    try:
        res = await llm.call(
            role="actor", conversation_id="smoke", turn_index=0,
            system_prompt_id="smoke_v1", system="You are terse.",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=16, temperature=0.7,
        )
    finally:
        await llm.aclose()
    assert res.text
    assert res.usage["input_tokens"] > 0 and res.usage["output_tokens"] > 0
    row = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert row["model"] == config.MODEL and row["error"] is None
