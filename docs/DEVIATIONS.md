# DEVIATIONS from `docs/IMPLEMENTATION_PLAN.md`

Every change from the plan (and from `PREREGISTRATION.md`) is logged here with its date and reason.
The pre-registration itself is never edited.

---

## 2026-09-29 — T6: ReWOO HotpotQA few-shot tool rename

Plan §T6-D mandates it, logged here as required: the vendored ReWOO few-shot
`HOTPOTQA_PWS_BASE` uses the tool name `Wikipedia[...]`. Every occurrence of `Wikipedia[`
is replaced by `Search[` so the few-shot names the tool this study actually exposes
(§2.3 harmonised tools). This is the **only** edit made to that vendored string.

**Reason:** all scaffolds must share one tool set; ReWOO's original `Wikipedia` worker
returned longer page content, which would confound control flow with tool quality
(threat to validity §8.1).

## 2026-09-29 — T6: Reflexion reflector prompt adapted to GSM8K

The vendored `REFLECT_INSTRUCTION` says the agent had "access to an Docstore API
environment". For GSM8K that phrase is replaced by "a calculator tool" and `{examples}` is
filled with the empty string (the vendored `REFLECTIONS` few-shots are HotpotQA
trajectories and would be off-distribution for arithmetic). Mandated by plan §T6-I.

## 2026-09-29 — T0: Rate limits and credit balance

<!-- FILL IN BEFORE LAUNCHING T13. Blocking check per plan §T0.2. -->
- Organisation RPM for `claude-haiku-4-5`: _TBD_
- Organisation ITPM: _TBD_
- Organisation OTPM: _TBD_
- Credit balance at launch: _TBD_
- Scope-down ladder (§6.3) steps applied, if any: _none yet_

## 2026-09-29 — T12: Pilot go/no-go decision

<!-- FILL IN AFTER THE PILOT. -->
- Decision: _TBD_
- Evidence: `results/pilot/PILOT_REPORT.md`

## 2026-09-29 — T5/T8: tool names are canonicalised, not taken verbatim

`parse_action` (T7) is specified to Title-case the tool name. A bare `.title()` turns
`LLM` into `Llm`, which would break ReWOO's `#Ek = LLM[...]` worker, so both parsers now
snap the Title-cased name to the exact spelling of a known tool (`Search`, `Lookup`,
`Calculate`, `Finish`, `LLM`) case-insensitively, and fall through unchanged otherwise.

**Reason:** identical behaviour to the spec for every tool the spec covers, and it stops a
purely cosmetic casing difference in the model's output from being scored as an unknown
tool. Unknown tools still produce the unknown-tool error string.

## 2026-09-29 — T11: `--fake-llm` self-test flag on the runner

The runner accepts `--fake-llm`, which drives the whole pipeline with scripted replies and
makes no API calls. It exists only to exercise runner → JSONL → analysis offline (the T11
acceptance criterion) and is never used for a real run; output written under such a run tag
carries a warning banner in the log.
