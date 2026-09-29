"""Wikipedia 429 handling, and billing a retried run only for its final attempt."""

import json

import httpx
import pandas as pd
import pytest

from acs import config
from acs.analysis.load import live_calls, load
from acs.tools import wiki as W


def _client(tmp_path, responses):
    """A WikiClient whose HTTP layer replays `responses` (status, headers) in order."""
    seq = list(responses)
    seen = []

    def handler(request):
        seen.append(request)
        status, headers = seq.pop(0)
        body = {"query": {"pages": []}} if status == 200 else {}
        return httpx.Response(status, headers=headers, json=body)

    client = W.WikiClient(cache=W.SqliteCache(tmp_path / "wiki.sqlite"))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, seen


async def test_429s_are_retried_honouring_retry_after_and_then_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_backoff", lambda attempt: 0.0)
    client, seen = _client(tmp_path, [(429, {"retry-after": "0"}), (503, {}), (200, {})])
    params = {"action": "query", "titles": "X", "format": "json"}
    assert await client._get_json(params) == {"query": {"pages": []}}
    assert client.throttled == 2 and len(seen) == 3
    assert await client._get_json(params) == {"query": {"pages": []}}  # from the disk cache
    assert len(seen) == 3


async def test_a_429_sets_a_cooldown_for_every_request(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_backoff", lambda attempt: 0.0)
    client, _ = _client(tmp_path, [(429, {"retry-after": "0.05"}), (200, {})])
    await client._get_json({"action": "query", "titles": "Y", "format": "json"})
    assert client._cooldown_until > 0


async def test_retries_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_backoff", lambda attempt: 0.0)
    monkeypatch.setattr(config, "WIKI_MAX_RETRIES", 3)
    client, _ = _client(tmp_path, [(429, {"retry-after": "0"})] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        await client._get_json({"action": "query", "titles": "Z", "format": "json"})


def test_retry_after_parsing():
    assert W._parse_retry_after("7") == 7.0
    assert W._parse_retry_after("100000") == config.WIKI_BACKOFF_CAP_S
    assert W._parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert W._parse_retry_after(None) is None


def _call(run_id, attempt, call_id, conv_turn=0):
    return {"call_id": call_id, "run_id": run_id, "attempt_id": attempt,
            "conversation_id": f"{run_id}:c", "turn_index": conv_turn}


def test_live_calls_keep_only_the_final_attempt():
    calls = pd.DataFrame([
        _call("r1", "a-old", "c1"),       # errored first attempt of r1
        _call("r1", "a-new", "c2"),       # final attempt of r1
        _call("r2", None, "c3"),          # legacy row (no attempt ids at all)
        _call("r3", "x", "c4"),           # run that is not in runs.jsonl
    ])
    runs = [{"run_id": "r1", "attempt_id": "a-new"}, {"run_id": "r2"}]
    assert live_calls(calls, runs)["call_id"].tolist() == ["c2", "c3"]


def test_load_bills_a_retried_run_once(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    base = {"benchmark": "gsm8k", "task_id": "t1", "scaffold": "react", "replicate": 0,
            "role": "actor", "system_prompt_id": "p", "model": config.MODEL,
            "input_tokens": 100, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "output_tokens": 10, "ts": "2026-09-29T16:00:00+00:00"}
    run_id = "gsm8k:t1:react:0"
    calls = [
        {**base, **_call(run_id, "a1", "c1")},
        {**base, **_call(run_id, "a2", "c2")},
    ]
    run_common = {"run_id": run_id, "benchmark": "gsm8k", "task_id": "t1", "scaffold": "react",
                  "replicate": 0, "final_answer": "1", "gold": "1", "em": 1, "f1": 1.0,
                  "correct": 1, "n_llm_calls": 1, "n_tool_calls": {}, "n_steps": 1,
                  "halted": False, "tokens_in": 100, "tokens_out": 10, "usd_nocache": 0.0,
                  "usd_empirical": 0.0, "wall_s": 1.0, "extra": {}, "trace": []}
    runs = [{**run_common, "status": "error", "error": "429", "attempt_id": "a1"},
            {**run_common, "status": "ok", "error": None, "attempt_id": "a2"}]
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    ds = load("t", run_dir=d)
    assert ds.calls["call_id"].tolist() == ["c2"]
    assert len(ds.calls_all) == 2  # both attempts cost money
    assert ds.ownership["call_id"].tolist() == ["c2"]


def test_holm_keeps_small_p_values_small():
    from acs.analysis.stats import holm
    assert holm([0.0005, 0.04, 0.3]) == [0.0015, 0.08, 0.3]


def test_bootstrap_p_is_never_zero():
    import numpy as np
    from acs.analysis.stats import two_sided_p
    p = two_sided_p(np.ones(2000))  # every resample on one side
    assert 0 < p <= 2 / 2001 + 1e-15
