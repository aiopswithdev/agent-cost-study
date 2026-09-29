from acs import config
from acs.scaffolds.plan_exec import FALLBACK_STEP, parse_plan, run_plan_exec
from tests.conftest import make_ctx


def test_parse_numbered_plan():
    steps, ok = parse_plan("Plan:\n1. Find A.\n2. Find B.\n3) Find C.")
    assert steps == ["Find A.", "Find B.", "Find C."] and ok is True


def test_plan_is_capped_at_five_steps():
    steps, ok = parse_plan("\n".join(f"{i}. step {i}" for i in range(1, 9)))
    assert len(steps) == config.PE_MAX_PLAN_STEPS and ok is True


def test_unparsable_plan_falls_back_to_one_step():
    steps, ok = parse_plan("I will just answer it.")
    assert steps == [FALLBACK_STEP] and ok is False


async def test_two_step_plan_runs_two_executor_conversations(fake_wiki):
    ctx = make_ctx(
        [
            "1. Find when Arthur's Magazine started.\n2. Find when First for Women started.",
            "Thought 1: search.\nAction 1: Search[Arthur's Magazine]",
            "Thought 2: found.\nAction 2: Finish[1844]",
            "Thought 1: search.\nAction 1: Search[First for Women]",
            "Thought 2: found.\nAction 2: Finish[1989]",
            "Arthur's Magazine",
        ],
        scaffold="plan_exec",
        wiki=fake_wiki,
    )
    result = await run_plan_exec(ctx)

    assert result.final_answer == "Arthur's Magazine"
    assert result.extra["plan_parse_ok"] is True
    assert result.extra["plan_steps"] == [
        "Find when Arthur's Magazine started.", "Find when First for Women started."
    ]
    assert result.extra["step_results"] == ["1844", "1989"]
    assert [r["role"] for r in ctx.llm.requests] == [
        "planner", "executor", "executor", "executor", "executor", "solver"
    ]

    # Distinct conversations, each with turn_index restarting at 0.
    rows = [e for e in ctx.trace if e["kind"] == "llm"]
    assert len(rows) == 6
    assert result.n_tool_calls["Search"] == 2

    # Step 2's first message carries step 1's result; step 1's says "(none)".
    step1_first = ctx.llm.requests[1]["messages"][0]["content"][0]["text"]
    step2_first = ctx.llm.requests[3]["messages"][0]["content"][0]["text"]
    assert "Results of completed steps:\n(none)" in step1_first
    assert "Step 1 result: 1844" in step2_first
    assert step2_first.rstrip().endswith("Current step 2: Find when First for Women started.")


async def test_toolbox_and_lookup_state_are_shared_across_steps(fake_wiki):
    ctx = make_ctx(
        [
            "1. Search the page.\n2. Look inside it.",
            "Thought 1: go.\nAction 1: Search[Arthur's Magazine]",
            "Thought 2: ok.\nAction 2: Finish[found]",
            "Thought 1: look.\nAction 1: Lookup[Philadelphia]",
            "Thought 2: ok.\nAction 2: Finish[Philadelphia]",
            "Philadelphia",
        ],
        scaffold="plan_exec",
        wiki=fake_wiki,
    )
    result = await run_plan_exec(ctx)
    # Lookup in step 2 resolves against the page Search loaded in step 1.
    lookup_obs = [e for e in ctx.trace if e.get("tool") == "Lookup"][0]["observation"]
    assert lookup_obs.startswith("(Result 1 /")
    assert result.extra["step_results"] == ["found", "Philadelphia"]


async def test_halted_step_yields_no_result():
    ctx = make_ctx(
        ["1. Do a thing."]
        + ["Thought: no action"] * config.PE_MAX_EXEC_STEPS
        + ["fallback answer"],
        scaffold="plan_exec",
    )
    result = await run_plan_exec(ctx)
    assert result.extra["step_results"] == ["No result."]
    assert result.halted is True
    assert result.final_answer == "fallback answer"
