from acs import config
from acs.prompts import library as lib
from acs.scaffolds.rewoo import parse_plan, run_rewoo, substitute_evidence
from tests.conftest import make_ctx


def test_parses_the_vendored_fewshots_first_example():
    first = lib.HOTPOTQA_PWS_BASE.split("\n\n")[0]
    steps = parse_plan(first)
    assert [s.index for s in steps] == [1, 2, 3, 4]
    assert [s.tool for s in steps] == ["Search", "LLM", "Search", "LLM"]
    assert steps[0].tool_input == "Colorado orogeny"
    assert steps[0].plan == "Search for more information about Colorado orogeny."
    assert steps[2].tool_input == "#E2"


def test_plan_is_capped_at_max_evidence():
    text = "\n".join(
        f"Plan: step {i}\n#E{i} = Search[e{i}]" for i in range(1, config.REWOO_MAX_EVIDENCE + 4)
    )
    assert len(parse_plan(text)) == config.REWOO_MAX_EVIDENCE


def test_no_evidence_lines_parse_to_nothing():
    assert parse_plan("Plan: think hard.\nThen answer.") == []


def test_substitution_is_longest_index_first():
    evidence = {1: "ONE", 10: "TEN", 2: "TWO"}
    assert substitute_evidence("#E10 and #E1 and #E2", evidence) == "TEN and ONE and TWO"
    assert substitute_evidence("Given context: #E1", {1: "ctx"}) == "Given context: ctx"


async def test_full_run_executes_workers_then_solves(fake_wiki):
    plan = (
        "Plan: Search for Arthur's Magazine.\n"
        "#E1 = Search[Arthur's Magazine]\n"
        "Plan: Search for First for Women.\n"
        "#E2 = Search[First for Women]\n"
        "Plan: Compare the two.\n"
        "#E3 = LLM[Which was started first? Given context: #E1, #E2]\n"
    )
    ctx = make_ctx(
        [plan, "Arthur's Magazine was started in 1844, earlier.", "Arthur's Magazine"],
        scaffold="rewoo",
        wiki=fake_wiki,
        with_llm_worker=True,
    )
    result = await run_rewoo(ctx)

    assert result.final_answer == "Arthur's Magazine"
    assert result.extra["plan_parse_ok"] is True
    assert result.extra["n_evidence"] == 3
    assert result.n_tool_calls["Search"] == 2 and result.n_tool_calls["LLM"] == 1
    assert result.n_llm_calls == 3  # planner + one LLM worker + solver

    roles = [r["role"] for r in ctx.llm.requests]
    assert roles == ["planner", "worker_llm", "solver"]
    # The worker saw the substituted evidence, not the raw `#E` references.
    worker_msg = ctx.llm.requests[1]["messages"][0]["content"][0]["text"]
    assert "#E1" not in worker_msg and "American literary periodical" in worker_msg
    # The solver saw one Plan/Evidence pair per step.
    solver_msg = ctx.llm.requests[2]["messages"][0]["content"][0]["text"]
    assert solver_msg.count("Plan: ") == 3 and solver_msg.count("Evidence: ") == 3


async def test_unparsable_plan_still_calls_the_solver():
    ctx = make_ctx(
        ["I will just answer directly.", "Arthur's Magazine"],
        scaffold="rewoo",
        with_llm_worker=True,
    )
    result = await run_rewoo(ctx)
    assert result.extra["plan_parse_ok"] is False
    assert result.extra["n_evidence"] == 0
    assert result.final_answer == "Arthur's Magazine"
    assert [r["role"] for r in ctx.llm.requests] == ["planner", "solver"]
    solver_msg = ctx.llm.requests[1]["messages"][0]["content"][0]["text"]
    assert "Plan: " not in solver_msg


async def test_unknown_tool_becomes_the_evidence_string():
    ctx = make_ctx(
        ["Plan: use a tool.\n#E1 = Translate[hi]\n", "42"],
        scaffold="rewoo",
        with_llm_worker=True,
    )
    result = await run_rewoo(ctx)
    solver_msg = ctx.llm.requests[1]["messages"][0]["content"][0]["text"]
    assert "Error: unknown tool Translate" in solver_msg
    assert result.final_answer == "42"


async def test_solver_finish_wrapper_is_unwrapped():
    ctx = make_ctx(
        ["Plan: p\n#E1 = Search[Arthur's Magazine]\n", "Finish[Arthur's Magazine]"],
        scaffold="rewoo",
        with_llm_worker=True,
    )
    result = await run_rewoo(ctx)
    assert result.final_answer == "Arthur's Magazine"
