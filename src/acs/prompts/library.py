"""T6 — every prompt string in the study, each with a stable `system_prompt_id`.

Vendored upstream prompts are read from `prompts/vendor/` and their string literals are
extracted with `ast` at import time (the vendored `.py` files import langchain and must
never themselves be imported). Authored text is written out in full below.

Every deviation from a vendored string is listed in `docs/DEVIATIONS.md`.
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path

from .. import config

VENDOR = Path(__file__).resolve().parent / "vendor"


# --------------------------------------------------------------------------- vendor access


@lru_cache(maxsize=None)
def _vendor_strings(filename: str) -> dict[str, str]:
    """Top-level `NAME = "literal"` assignments of a vendored module, without importing it."""
    tree = ast.parse((VENDOR / filename).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            out[target.id] = value
    return out


@lru_cache(maxsize=None)
def _react_json() -> dict[str, str]:
    return json.loads((VENDOR / "react_prompts_naive.json").read_text(encoding="utf-8"))


def vendored(filename: str, name: str) -> str:
    strings = _vendor_strings(filename)
    if name not in strings:
        raise KeyError(f"{name!r} not found in vendored {filename}")
    return strings[name]


WEBTHINK_SIMPLE6 = _react_json()["webthink_simple6"]
REFLECT_INSTRUCTION = vendored("reflexion_prompts.py", "REFLECT_INSTRUCTION")
REFLECTION_HEADER = vendored("reflexion_prompts.py", "REFLECTION_HEADER")
REFLECTIONS_FEWSHOT = vendored("reflexion_fewshots.py", "REFLECTIONS")
REWOO_PLANNER_PREFIX = vendored("rewoo_planner.py", "DEFAULT_PREFIX")
REWOO_PLANNER_SUFFIX = vendored("rewoo_planner.py", "DEFAULT_SUFFIX")
REWOO_SOLVER_PREFIX = vendored("rewoo_solver.py", "DEFAULT_PREFIX")
REWOO_SOLVER_SUFFIX = vendored("rewoo_solver.py", "DEFAULT_SUFFIX")
#: DEVIATION (logged): the only edit to this vendored string is the tool rename
#: `Wikipedia[` -> `Search[`, so the few-shot names the tool this study exposes.
HOTPOTQA_PWS_BASE = vendored("rewoo_fewshots.py", "HOTPOTQA_PWS_BASE").replace(
    "Wikipedia[", "Search["
)


# --------------------------------------------------------------------------- A. answer style

ANSWER_STYLE = {
    "hotpotqa": (
        "Your final answer must be as short as possible: a name, entity, date, number, "
        "or yes/no — no explanation."
    ),
    "gsm8k": "Your final answer must be a single number only.",
}


# --------------------------------------------------------------------------- B/C. ReAct

REACT_INSTRUCTION = {
    "hotpotqa": (
        "Solve a question answering task with interleaving Thought, Action, Observation "
        "steps. Thought can reason about the current situation, and Action can be three "
        "types:\n"
        "(1) Search[entity], which searches the exact entity on Wikipedia and returns the "
        "first paragraph if it exists. If not, it will return some similar entities to "
        "search.\n"
        "(2) Lookup[keyword], which returns the next sentence containing keyword in the "
        "last passage successfully found by Search.\n"
        "(3) Finish[answer], which returns the answer and finishes the task.\n"
        "Here are some examples.\n"
    ),
    "gsm8k": (
        "Solve a math word problem with interleaving Thought, Action, Observation steps. "
        "Thought can reason about the current situation, and Action can be two types:\n"
        "(1) Calculate[expression], which evaluates an arithmetic expression (numbers, "
        "+ - * / // % ** and parentheses) and returns the result.\n"
        "(2) Finish[answer], which returns the answer and finishes the task.\n"
        "Here are some examples.\n"
    ),
}

#: GSM8K train items 0, 1, 2. Every `Observation` is a real `calculate()` output
#: (enforced by `tests/test_prompts.py`).
GSM8K_REACT_EXEMPLARS = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Thought 1: In May she sold half of 48 clips, so I need 48 / 2.
Action 1: Calculate[48 / 2]
Observation 1: 24
Thought 2: She sold 24 clips in May. The total is 48 + 24.
Action 2: Calculate[48 + 24]
Observation 2: 72
Thought 3: Natalia sold 72 clips altogether.
Action 3: Finish[72]

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Thought 1: She earns 12 dollars per 60 minutes, so for 50 minutes she earns 12 / 60 * 50.
Action 1: Calculate[12 / 60 * 50]
Observation 1: 10
Thought 2: She earned 10 dollars.
Action 2: Finish[10]

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Thought 1: Betty has 100 / 2. Her parents give 15 and her grandparents give 15 * 2. She still needs 100 - 100 / 2 - 15 - 15 * 2.
Action 1: Calculate[100 - 100 / 2 - 15 - 15 * 2]
Observation 1: 5
Thought 2: Betty needs 5 more dollars.
Action 2: Finish[5]
"""

REACT_EXEMPLARS = {"hotpotqa": WEBTHINK_SIMPLE6, "gsm8k": GSM8K_REACT_EXEMPLARS}

END_OF_EXAMPLES = "\n(END OF EXAMPLES)\n"


def react_body(benchmark: str) -> str:
    """The ReAct system prompt **without** ANSWER_STYLE (the Plan-Exec executor base)."""
    return REACT_INSTRUCTION[benchmark] + REACT_EXEMPLARS[benchmark] + END_OF_EXAMPLES


def react_system(benchmark: str) -> str:
    return react_body(benchmark) + ANSWER_STYLE[benchmark]


# --------------------------------------------------------------------------- D. ReWOO planner

TOOL_DESCRIPTIONS = {
    "Search": (
        "Searches the exact entity on Wikipedia and returns the first paragraph if it "
        "exists; otherwise returns similar entities. Input should be an entity name."
    ),
    "Lookup": (
        "Returns the next sentence containing the keyword in the page most recently found "
        "by Search. Input should be a keyword."
    ),
    "Calculate": (
        "Evaluates an arithmetic expression (numbers, + - * / // % ** and parentheses). "
        "Input should be a mathematical expression."
    ),
    "LLM": (
        "A pretrained LLM like yourself. Useful when you need to act with general world "
        "knowledge and common sense. Prioritize it when you are confident in solving the "
        "problem yourself. Input can be any instruction."
    ),
}


def tool_list_block(benchmark: str, with_llm_worker: bool = True) -> str:
    names = list(config.TOOLS_BY_BENCHMARK[benchmark])
    if with_llm_worker:
        names += list(config.REWOO_EXTRA_TOOLS)
    lines = "".join(f"{n}[input]: {TOOL_DESCRIPTIONS[n]}\n" for n in names)
    return f"Tools can be one of the following:\n{lines}\n"


GSM8K_PWS_FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Plan: Compute the number of clips sold in May, which is half of the April sales.
#E1 = Calculate[48 / 2]
Plan: Add the April and May sales.
#E2 = Calculate[48 + #E1]

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Plan: Compute the earnings for 50 minutes at 12 dollars per 60 minutes.
#E1 = Calculate[12 / 60 * 50]

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Plan: Compute how much money Betty already has.
#E1 = Calculate[100 / 2]
Plan: Compute the grandparents' gift, which is twice the parents' 15 dollars.
#E2 = Calculate[15 * 2]
Plan: Compute how much more money Betty needs.
#E3 = Calculate[100 - #E1 - 15 - #E2]

"""

REWOO_FEWSHOT = {"hotpotqa": HOTPOTQA_PWS_BASE, "gsm8k": GSM8K_PWS_FEWSHOT}


def rewoo_planner_system(benchmark: str) -> str:
    return (
        REWOO_PLANNER_PREFIX
        + tool_list_block(benchmark)
        + REWOO_FEWSHOT[benchmark]
        + REWOO_PLANNER_SUFFIX
    )


def rewoo_planner_user(question: str) -> str:
    return f"Question: {question}"


# --------------------------------------------------------------------------- E. shared solver


def solver_system(benchmark: str) -> str:
    """Shared by ReWOO and Plan-Exec so only planning/execution differs between them."""
    return REWOO_SOLVER_PREFIX + ANSWER_STYLE[benchmark]


def solver_user(question: str, pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"Plan: {plan}\nEvidence: {evidence}\n" for plan, evidence in pairs)
    return body + REWOO_SOLVER_SUFFIX + question


# --------------------------------------------------------------------------- F. ReWOO worker

WORKER_LLM_SYSTEM = (
    "You are a helpful assistant. Answer the request concisely, using the given context "
    "when provided."
)


# --------------------------------------------------------------------------- G. Plan-Exec planner

PE_HOTPOTQA_EXAMPLES = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Plan:
1. Find the area that the eastern sector of the Colorado orogeny extends into.
2. Find the elevation range of that area.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Plan:
1. Find who Matt Groening named the character Milhouse after.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Plan:
1. Find what the documentary Adam Clayton Powell is about.
2. Find what The Saimaa Gesture is about.
3. Decide which of the two is about Finnish rock groups.
"""

PE_GSM8K_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Plan:
1. Compute the number of clips sold in May, which is half of 48.
2. Add the April and May sales.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Plan:
1. Compute the earnings for 50 minutes at 12 dollars per hour.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Plan:
1. Compute how much money Betty already has.
2. Compute the total gifts from her parents and grandparents.
3. Compute how much more money Betty needs.
"""

PE_EXAMPLES = {"hotpotqa": PE_HOTPOTQA_EXAMPLES, "gsm8k": PE_GSM8K_EXAMPLES}


def pe_tool_names_block(benchmark: str) -> str:
    names = config.TOOLS_BY_BENCHMARK[benchmark]
    return "; ".join(f"{n}[input] ({TOOL_DESCRIPTIONS[n]})" for n in names)


def pe_planner_system(benchmark: str) -> str:
    return (
        "Let's first understand the problem and devise a plan to solve it. Write a short "
        f"step-by-step plan of at most {config.PE_MAX_PLAN_STEPS} steps. Each step must be "
        "a sub-task that can be completed with these tools: "
        f"{pe_tool_names_block(benchmark)}. Do not solve the steps yourself. Output only "
        'the numbered plan, one step per line, in the form "1. …".\n\n'
        "Here are some examples.\n"
        f"{PE_EXAMPLES[benchmark]}"
        "(END OF EXAMPLES)"
    )


def pe_planner_user(question: str) -> str:
    return f"Question: {question}\nPlan:"


# --------------------------------------------------------------------------- H. Plan-Exec executor

PE_EXECUTOR_SUFFIX = (
    "\nYou are executing ONE step of a larger plan. Use Finish[result] to return the "
    "result of the CURRENT step only (a short fact or number), not necessarily the final "
    "answer to the question."
)


def pe_executor_system(benchmark: str) -> str:
    return react_body(benchmark) + PE_EXECUTOR_SUFFIX


def pe_executor_user(
    question: str, plan: list[str], results: list[str], step_index: int
) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
    done = (
        "\n".join(f"Step {i + 1} result: {r}" for i, r in enumerate(results))
        if results
        else "(none)"
    )
    return (
        f"Question: {question}\n"
        f"Plan:\n{numbered}\n"
        f"Results of completed steps:\n{done}\n"
        f"Current step {step_index}: {plan[step_index - 1]}"
    )


# --------------------------------------------------------------------------- I/J. Reflexion

_REFLECT_SPLIT = "Previous trial:"


def reflector_system(benchmark: str) -> str:
    head = REFLECT_INSTRUCTION.split(_REFLECT_SPLIT)[0]
    if benchmark == "hotpotqa":
        return head.replace("{examples}", REFLECTIONS_FEWSHOT)
    #: DEVIATION (logged): GSM8K has no Docstore and the vendored few-shots are HotpotQA
    #: trajectories, so the environment is renamed and `{examples}` is left empty.
    head = head.replace("an Docstore API environment", "a calculator tool")
    return head.replace("{examples}", "")


def truncate_scratchpad(scratchpad: str, max_chars: int = config.SCRATCHPAD_MAX_CHARS) -> str:
    """Shorten the longest `Observation` lines to 200 chars until the whole fits."""
    lines = scratchpad.split("\n")
    observations = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("Observation")]
    observations.sort(key=lambda p: len(p[1]), reverse=True)
    for i, line in observations:
        if len("\n".join(lines)) <= max_chars:
            break
        lines[i] = line[:200] + ("…" if len(line) > 200 else "")
    return "\n".join(lines)


def reflector_user(question: str, scratchpad: str) -> str:
    return (
        f"{_REFLECT_SPLIT}\n"
        f"Question: {question}\n"
        f"{truncate_scratchpad(scratchpad)}\n\n"
        "Reflection:"
    )


def format_reflections(reflections: list[str], header: str = REFLECTION_HEADER) -> str:
    """Exactly Reflexion's `format_reflections`."""
    if not reflections:
        return ""
    return header + "Reflections:\n- " + "\n- ".join(r.strip() for r in reflections)


# --------------------------------------------------------------------------- K. judge

JUDGE_SYSTEM = (
    "You are a strict verifier. You will see a question, an agent's reasoning trajectory "
    "including tool observations, and the agent's final answer. You do NOT know the "
    "correct answer. Decide whether the final answer is correct, judging only from the "
    "question, the evidence in the observations, and your own knowledge. Write one "
    'sentence of justification, then on the last line write exactly "VERDICT: CORRECT" or '
    '"VERDICT: INCORRECT".'
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)")


def judge_user(question: str, scratchpad: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Trajectory:\n{truncate_scratchpad(scratchpad)}\n\n"
        f"Final answer: {answer}"
    )


def parse_verdict(text: str) -> tuple[str, bool]:
    """Returns `(verdict, parse_failed)`; a missing verdict counts as INCORRECT."""
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return "INCORRECT", True
    return matches[-1], False


# --------------------------------------------------------------------------- registry

SYSTEM_PROMPT_BUILDERS = {
    "react_{b}_v1": react_system,
    "rewoo_planner_{b}_v1": rewoo_planner_system,
    "solver_{b}_v1": solver_system,
    "pe_planner_{b}_v1": pe_planner_system,
    "pe_executor_{b}_v1": pe_executor_system,
    "reflector_{b}_v1": reflector_system,
}


def system_prompt_id(kind: str, benchmark: str) -> str:
    """`kind` is one of react / rewoo_planner / solver / pe_planner / pe_executor /
    reflector / judge / worker_llm."""
    if kind == "worker_llm":
        return "worker_llm_v1"
    return f"{kind}_{benchmark}_v1"


@lru_cache(maxsize=None)
def all_system_prompts() -> dict[str, str]:
    """Every distinct system prompt in the study, keyed by `system_prompt_id` (§4.3)."""
    out: dict[str, str] = {"worker_llm_v1": WORKER_LLM_SYSTEM}
    for benchmark in config.BENCHMARKS:
        for template, builder in SYSTEM_PROMPT_BUILDERS.items():
            out[template.format(b=benchmark)] = builder(benchmark)
        out[f"judge_{benchmark}_v1"] = JUDGE_SYSTEM
    return out


def get_system_prompt(system_prompt_id_: str) -> str:
    return all_system_prompts()[system_prompt_id_]


# --------------------------------------------------------------------------- exemplar questions

_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.M)


@lru_cache(maxsize=None)
def exemplar_questions(benchmark: str) -> tuple[str, ...]:
    """Every question appearing in a few-shot, so §2.2 can exclude them from the sample."""
    sources = [REACT_EXEMPLARS[benchmark], REWOO_FEWSHOT[benchmark], PE_EXAMPLES[benchmark]]
    if benchmark == "hotpotqa":
        sources.append(REFLECTIONS_FEWSHOT)
    seen: dict[str, None] = {}
    for src in sources:
        for q in _QUESTION_RE.findall(src):
            seen.setdefault(q.strip(), None)
    return tuple(seen)
