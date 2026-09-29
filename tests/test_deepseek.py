"""The DeepSeek adapter, offline: the API client is replaced by a stub."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from acs import config
from acs.llm import (
    DEEPSEEK_EXTRA_BODY,
    CallLog,
    DeepSeekLLM,
    SpendLimitExceeded,
    ThinkingModeError,
    make_llm,
    to_openai_messages,
    usage_from_openai,
)


def _usage(prompt=1000, hit=800, miss=200, completion=50, reasoning=0):
    return openai.types.CompletionUsage.model_validate({
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    })


def _response(text="Thought 1: x\nAction 1: Search[y]", finish="stop", usage=None, reasoning=None):
    message = SimpleNamespace(content=text, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=usage if usage is not None else _usage(),
    )


class StubCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _llm(tmp_path, responses, max_usd=100.0):
    llm = DeepSeekLLM(call_log=CallLog(tmp_path / "calls.jsonl", max_usd=max_usd),
                      model="deepseek-flash", api_key="test-key")
    stub = StubCompletions(responses)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return llm, stub


async def _call(llm, stop=None):
    return await llm.call(
        role="actor", conversation_id="c", turn_index=0, system_prompt_id="p",
        system="SYS", messages=[{"role": "user", "content": "Question: q"}],
        max_tokens=512, temperature=0.7, stop=stop,
    )


def test_to_openai_messages_flattens_blocks_and_drops_cache_control():
    system_blocks = [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "Question: q"},
        {"role": "assistant", "content": "Action 1: Search[x]"},
        {"role": "user", "content": [{"type": "text", "text": "Observation 1: o",
                                      "cache_control": {"type": "ephemeral"}}]},
    ]
    assert to_openai_messages(system_blocks, messages) == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Question: q"},
        {"role": "assistant", "content": "Action 1: Search[x]"},
        {"role": "user", "content": "Observation 1: o"},
    ]


def test_usage_maps_hits_and_misses():
    assert usage_from_openai(_usage(1000, 800, 200, 50)) == {
        "input_tokens": 200, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 800, "output_tokens": 50,
    }


def test_usage_falls_back_to_prompt_tokens_details():
    u = openai.types.CompletionUsage.model_validate({
        "prompt_tokens": 1000, "completion_tokens": 5, "total_tokens": 1005,
        "prompt_tokens_details": {"cached_tokens": 600},
    })
    got = usage_from_openai(u)
    assert got["cache_read_input_tokens"] == 600 and got["input_tokens"] == 400


async def test_request_disables_thinking_and_passes_sampling_params(tmp_path):
    llm, stub = _llm(tmp_path, [_response()])
    res = await _call(llm, stop=["\nObservation"])
    req = stub.requests[0]
    assert req["extra_body"] == DEEPSEEK_EXTRA_BODY == {"thinking": {"type": "disabled"}}
    assert req["model"] == "deepseek-flash"
    assert req["temperature"] == 0.7 and req["max_tokens"] == 512
    assert req["stop"] == ["\nObservation"]
    assert req["messages"][0] == {"role": "system", "content": "SYS"}
    assert res.text.startswith("Thought 1") and res.stop_reason == "end_turn"
    row = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert row["model"] == "deepseek-flash"
    assert (row["input_tokens"], row["cache_read_input_tokens"], row["output_tokens"]) == (200, 800, 50)


async def test_length_finish_maps_to_max_tokens(tmp_path):
    llm, _ = _llm(tmp_path, [_response(finish="length")])
    assert (await _call(llm)).stop_reason == "max_tokens"


@pytest.mark.parametrize("kw", [{"reasoning": "hidden thoughts"}, {"usage": _usage(reasoning=12)}])
async def test_reasoning_output_fails_loudly(tmp_path, kw):
    llm, _ = _llm(tmp_path, [_response(**kw)])
    with pytest.raises(ThinkingModeError):
        await _call(llm)


@pytest.mark.parametrize("status", [401, 402])
async def test_bad_key_or_no_credit_stops_the_study(tmp_path, status):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    err = openai.APIStatusError("nope", response=httpx.Response(status, request=request), body=None)
    llm, _ = _llm(tmp_path, [err])
    with pytest.raises(SpendLimitExceeded):
        await _call(llm)
    assert llm.stopped is True
    with pytest.raises(SpendLimitExceeded):  # nothing further is attempted
        await _call(llm)


async def test_count_tokens_subtracts_the_template_overhead(tmp_path):
    llm, stub = _llm(tmp_path, [
        _response(usage=_usage(prompt=11, hit=0, miss=11)),    # "." + "x" -> overhead 10
        _response(usage=_usage(prompt=1510, hit=0, miss=1510)),
    ])
    assert await llm.count_tokens("A long system prompt", []) == 1500
    assert all(r["max_tokens"] == 1 for r in stub.requests)
    assert not (tmp_path / "calls.jsonl").exists()  # study log holds study calls only


async def test_spend_guard_watches_the_billed_amount(tmp_path):
    # 1M hit + 1M miss + 1M output at peak list price = 0.006 + 0.30 + 1.20 = 1.506 USD.
    big = _usage(prompt=2_000_000, hit=1_000_000, miss=1_000_000, completion=1_000_000)
    llm, _ = _llm(tmp_path, [_response(usage=big), _response()], max_usd=0.5)
    await _call(llm)
    assert llm.call_log.total_usd_billed >= 0.5 * 1.506 - 1e-9
    assert llm.stopped
    with pytest.raises(SpendLimitExceeded):
        await _call(llm)


def test_make_llm_picks_the_configured_provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    llm = make_llm(provider="deepseek")
    assert isinstance(llm, DeepSeekLLM) and llm.model == config.DEFAULT_MODEL["deepseek"]
    with pytest.raises(ValueError):
        make_llm(provider="nope")


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        config.api_key("deepseek")
