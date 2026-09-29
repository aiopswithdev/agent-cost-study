# Vendored prompt sources (verbatim, read-only)

Do NOT edit these files. Do NOT import the `.py` ones (upstream imports langchain);
`src/acs/prompts/library.py` extracts their string literals with `ast` at build time.

| File | Repo | Commit SHA | Upstream path |
|---|---|---|---|
| `react_prompts_naive.json` | `ysymyth/ReAct` | `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9` | `prompts/prompts_naive.json` |
| `reflexion_prompts.py` | `noahshinn/reflexion` | `218cf0ef1df84b05ce379dd4a8e47f17766733a0` | `hotpotqa_runs/prompts.py` |
| `reflexion_fewshots.py` | `noahshinn/reflexion` | `218cf0ef1df84b05ce379dd4a8e47f17766733a0` | `hotpotqa_runs/fewshots.py` |
| `rewoo_fewshots.py` | `billxbf/ReWOO` | `9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `prompts/fewshots.py` |
| `rewoo_planner.py` | `billxbf/ReWOO` | `9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `prompts/planner.py` |
| `rewoo_solver.py` | `billxbf/ReWOO` | `9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `prompts/solver.py` |

Raw URL pattern: `https://raw.githubusercontent.com/{repo}/{sha}/{path}`
