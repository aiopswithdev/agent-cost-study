"""T5 — per-run tool state and dispatch. One `ToolBox` per scaffold run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .. import config
from .calc import calculate
from .wiki import LookupState, WikiClient, default_client, lookup

if TYPE_CHECKING:  # pragma: no cover
    from ..llm import BaseLLM


def valid_tools(benchmark: str, with_llm_worker: bool = False) -> tuple[str, ...]:
    tools = config.TOOLS_BY_BENCHMARK[benchmark]
    return tools + config.REWOO_EXTRA_TOOLS if with_llm_worker else tools


def valid_actions_str(benchmark: str, with_llm_worker: bool = False) -> str:
    """The text appended to the "Invalid action." observation inside `react_loop`."""
    names = valid_tools(benchmark, with_llm_worker) + ("Finish",)
    return ", ".join(f"{n}[...]" for n in names) + "."


class ToolBox:
    """Holds the Lookup state, the tool-call counters and the run's trace."""

    def __init__(
        self,
        benchmark: str,
        llm_for_worker: "BaseLLM | None" = None,
        wiki: WikiClient | None = None,
        trace: list[dict] | None = None,
        with_llm_worker: bool = False,
        worker_call: Callable[[str], Any] | None = None,
    ) -> None:
        if benchmark not in config.TOOLS_BY_BENCHMARK:
            raise ValueError(f"unknown benchmark {benchmark!r}")
        self.benchmark = benchmark
        self.llm_for_worker = llm_for_worker
        self.with_llm_worker = with_llm_worker or llm_for_worker is not None
        self._worker_call = worker_call
        self.wiki = wiki if wiki is not None else (default_client() if benchmark == "hotpotqa" else None)
        self.trace: list[dict] = trace if trace is not None else []
        self.lookup_state = LookupState()
        self.n_tool_calls: dict[str, int] = {"Search": 0, "Lookup": 0, "Calculate": 0, "LLM": 0}

    @property
    def valid(self) -> tuple[str, ...]:
        return valid_tools(self.benchmark, self.with_llm_worker)

    def valid_actions_str(self) -> str:
        return valid_actions_str(self.benchmark, self.with_llm_worker)

    def _unknown(self, tool_name: str) -> str:
        return f"Error: unknown tool {tool_name}. Valid tools: {list(self.valid)}"

    async def run(self, tool_name: str, arg: str) -> str:
        """Dispatch, truncate to `OBS_MAX_CHARS`, count the call, append a trace entry."""
        arg = arg if arg is not None else ""
        if tool_name not in self.valid:
            obs = self._unknown(tool_name)
        elif tool_name == "Search":
            observation, page_text = await self.wiki.search(arg)
            if page_text is not None:
                self.lookup_state.page_text = page_text
                self.lookup_state.keyword = None
                self.lookup_state.results = []
                self.lookup_state.cnt = 0
            obs = observation
        elif tool_name == "Lookup":
            obs = lookup(self.lookup_state.page_text, arg, self.lookup_state)
        elif tool_name == "Calculate":
            obs = calculate(arg)
        elif tool_name == "LLM":
            obs = await self._run_worker(arg)
        else:  # pragma: no cover - `valid` is exhaustive
            obs = self._unknown(tool_name)

        obs = (obs or "")[: config.OBS_MAX_CHARS]
        if tool_name in self.n_tool_calls:
            self.n_tool_calls[tool_name] += 1
        self.trace.append({"kind": "tool", "tool": tool_name, "input": arg, "observation": obs})
        return obs

    async def _run_worker(self, arg: str) -> str:
        if self._worker_call is None:
            return "Error: LLM worker is not available in this run."
        return await self._worker_call(arg)
