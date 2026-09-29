import pytest

from acs import config
from acs.scaffolds.reflexion import run_reflexion
from acs.store import PoolRecord
from tests.conftest import make_ctx, make_task

GOLD = "Arthur's Magazine"


def pool_record(answer: str = "First for Women", run_id: str = "hotpotqa:t1:react:0") -> PoolRecord:
    return PoolRecord(
        run_id=run_id,
        final_answer=answer,
        halted=False,
        n_steps=3,
        n_llm_calls=3,
        n_tool_calls={"Search": 2, "Lookup": 0, "Calculate": 0, "LLM": 0},
        tokens_in=1000,
        tokens_out=100,
        usd_nocache=0.0015,
        usd_empirical=0.0012,
        transcript="Thought 1: a\nAction 1: Search[x]\nObservation 1: y\n"
                   "Action 2: Finish[First for Women]",
        trace=[{"kind": "llm", "role": "actor", "call_id": "c0", "text": "trial one"}],
    )


def trial_script(answers: list[str], reflection: str = "I should search both pages."):
    """One actor turn plus one reflector turn per extra trial."""
    out = []
    for a in answers:
        out.append(f"Thought 1: retry.\nAction 1: Finish[{a}]")
        out.append(reflection)
    return out


async def test_gt_stops_at_the_first_gold_correct_trial():
    ctx = make_ctx(
        ["a reflection"] + trial_script([GOLD]),
        scaffold="reflexion_gt",
    )
    result = await run_reflexion(ctx, "gt", pool_record())
    assert result.final_answer == GOLD
    assert result.extra["n_trials"] == 2
    assert result.extra["trial_answers"] == ["First for Women", GOLD]
    assert result.extra["trial_correct"] == [0, 1]
    assert result.extra["judge_verdicts"] == [None, None]
    assert [r["role"] for r in ctx.llm.requests] == ["reflector", "actor"]


async def test_gt_runs_all_trials_when_never_correct():
    script = ["reflection"]
    for _ in range(config.REFLEXION_MAX_TRIALS - 1):
        script += ["Thought 1: x\nAction 1: Finish[wrong]", "reflection"]
    ctx = make_ctx(script, scaffold="reflexion_gt")
    result = await run_reflexion(ctx, "gt", pool_record())
    assert result.extra["n_trials"] == config.REFLEXION_MAX_TRIALS
    assert result.final_answer == "wrong"
    assert sum(result.extra["trial_correct"]) == 0


async def test_self_stops_when_the_judge_says_correct_even_if_wrong():
    ctx = make_ctx(["Looks fine.\nVERDICT: CORRECT"], scaffold="reflexion_self")
    result = await run_reflexion(ctx, "self", pool_record(answer="First for Women"))
    assert result.extra["n_trials"] == 1
    assert result.final_answer == "First for Women"
    assert result.extra["trial_correct"] == [0]  # graded wrong, but the judge stopped it
    assert result.extra["judge_verdicts"] == ["CORRECT"]
    assert [r["role"] for r in ctx.llm.requests] == ["judge"]


async def test_self_continues_when_the_judge_says_incorrect_even_if_right():
    ctx = make_ctx(
        [
            "Hmm.\nVERDICT: INCORRECT",
            "a reflection",
            f"Thought 1: x\nAction 1: Finish[{GOLD}]",
            "Seems right.\nVERDICT: CORRECT",
        ],
        scaffold="reflexion_self",
    )
    result = await run_reflexion(ctx, "self", pool_record(answer=GOLD))
    assert result.extra["n_trials"] == 2
    assert result.extra["trial_correct"] == [1, 1]
    assert result.extra["judge_verdicts"] == ["INCORRECT", "CORRECT"]


async def test_unparsable_verdict_counts_as_incorrect_and_is_recorded():
    ctx = make_ctx(
        ["I cannot tell.", "a reflection", "Thought: x\nAction: Finish[z]",
         "VERDICT: CORRECT"],
        scaffold="reflexion_self",
    )
    result = await run_reflexion(ctx, "self", pool_record())
    assert result.extra["judge_parse_fail"] == 1
    assert result.extra["judge_verdicts"][0] == "INCORRECT"


async def test_empty_answer_triggers_no_judge_call():
    ctx = make_ctx(
        ["a reflection", f"Thought 1: x\nAction 1: Finish[{GOLD}]", "VERDICT: CORRECT"],
        scaffold="reflexion_self",
    )
    result = await run_reflexion(ctx, "self", pool_record(answer=""))
    assert result.extra["judge_verdicts"][0] is None
    assert [r["role"] for r in ctx.llm.requests] == ["reflector", "actor", "judge"]


async def test_reflector_never_sees_the_gold_answer_or_the_judge_text():
    judge_text = "This is wrong because the magazine dates differ.\nVERDICT: INCORRECT"
    # A gold token that does not occur in the question, so any match is a real leak.
    task = make_task(question="Which documentary is about Finnish rock groups?",
                     answer="ZZGOLDZZ")
    ctx = make_ctx(
        [judge_text, "a reflection", "Thought: x\nAction: Finish[still wrong]",
         "VERDICT: CORRECT"],
        scaffold="reflexion_self",
        task=task,
    )
    await run_reflexion(ctx, "self", pool_record(answer="First for Women"))
    reflector_requests = [r for r in ctx.llm.requests if r["role"] == "reflector"]
    assert reflector_requests
    for req in reflector_requests:
        blob = req["system"][0]["text"] + "".join(
            m["content"] if isinstance(m["content"], str)
            else "".join(b["text"] for b in m["content"])
            for m in req["messages"]
        )
        assert "ZZGOLDZZ" not in blob
        assert "because the magazine dates differ" not in blob


async def test_memory_holds_at_most_three_reflections():
    script = ["reflection 1"]
    for i in range(2, config.REFLEXION_MAX_TRIALS + 1):
        script.append("Thought: x\nAction: Finish[wrong]")
        if i < config.REFLEXION_MAX_TRIALS:
            script.append(f"reflection {i}")
    ctx = make_ctx(script, scaffold="reflexion_gt")
    result = await run_reflexion(ctx, "gt", pool_record())

    assert len(result.extra["reflections"]) == config.REFLEXION_MAX_TRIALS - 1
    last_actor = [r for r in ctx.llm.requests if r["role"] == "actor"][-1]
    header = last_actor["messages"][0]["content"][0]["text"]
    assert header.count("\n- ") == config.REFLEXION_MEMORY
    assert "reflection 1" not in header  # the oldest reflection has fallen out


async def test_trial1_cost_is_counted_in_full():
    rec = pool_record()
    ctx = make_ctx(["Looks right.\nVERDICT: CORRECT"], scaffold="reflexion_self")
    result = await run_reflexion(ctx, "self", rec)

    judge_row = ctx.trace[-1]
    assert result.tokens_in > rec.tokens_in
    assert result.usd_nocache > rec.usd_nocache
    assert result.extra["trial1_source_run_id"] == rec.run_id
    assert result.extra["trial1_tokens_in"] == rec.tokens_in
    assert result.extra["trial1_usd_nocache"] == rec.usd_nocache
    assert result.n_tool_calls["Search"] == 2  # inherited from trial 1
    assert ctx.trace[0]["text"] == "trial one"  # trial 1's trace is carried over
    assert judge_row["role"] == "judge"


async def test_unknown_mode_is_rejected():
    ctx = make_ctx([], scaffold="reflexion_gt")
    with pytest.raises(ValueError):
        await run_reflexion(ctx, "oracle", pool_record())
