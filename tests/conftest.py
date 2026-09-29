"""Shared fixtures. Everything here is offline; network tests are marked `network`."""

from __future__ import annotations

import pytest

from acs.data import Task
from acs.llm import CallLog, FakeLLM
from acs.scaffolds.base import RunContext, make_run_id
from acs.tools.toolbox import ToolBox


class FakeWiki:
    """Stands in for `WikiClient`: a dict of entity -> (observation, page_text)."""

    def __init__(self, pages: dict[str, tuple[str, str | None]] | None = None) -> None:
        self.pages = pages or {}
        self.queries: list[str] = []

    async def search(self, entity: str) -> tuple[str, str | None]:
        self.queries.append(entity)
        if entity in self.pages:
            return self.pages[entity]
        return f"Could not find {entity}. Similar: [].", None


def make_task(benchmark: str = "hotpotqa", **kw) -> Task:
    defaults = dict(
        task_id="t1",
        question="Which magazine was started first Arthur's Magazine or First for Women?",
        answer="Arthur's Magazine",
        benchmark=benchmark,
    )
    if benchmark == "gsm8k":
        defaults.update(task_id="gsm8k_test_0", question="What is 48 / 2?", answer="24")
    defaults.update(kw)
    return Task(**defaults)


def make_ctx(
    responses=None,
    *,
    benchmark: str = "hotpotqa",
    scaffold: str = "react",
    replicate: int = 0,
    task: Task | None = None,
    wiki: FakeWiki | None = None,
    call_log: CallLog | None = None,
    with_llm_worker: bool = False,
) -> RunContext:
    task = task or make_task(benchmark)
    llm = FakeLLM(responses, call_log=call_log)
    trace: list[dict] = []
    toolbox = ToolBox(
        benchmark,
        wiki=wiki if wiki is not None else FakeWiki(),
        trace=trace,
        with_llm_worker=with_llm_worker,
    )
    return RunContext(
        run_id=make_run_id(benchmark, task.task_id, scaffold, replicate),
        benchmark=benchmark,
        task=task,
        scaffold=scaffold,
        replicate=replicate,
        llm=llm,
        toolbox=toolbox,
        trace=trace,
    )


@pytest.fixture
def fake_wiki() -> FakeWiki:
    return FakeWiki(
        {
            "Arthur's Magazine": (
                "Arthur's Magazine (1844-1846) was an American literary periodical.",
                "Arthur's Magazine (1844-1846) was an American literary periodical. "
                "It was published in Philadelphia.",
            ),
            "First for Women": (
                "First for Women is a woman's magazine started in 1989.",
                "First for Women is a woman's magazine started in 1989.",
            ),
        }
    )
