"""Action parsing against the output habits observed in the pilot (docs/DEVIATIONS.md)."""

import pytest

from acs.scaffolds.base import normalize_labels, parse_action, truncate_after_first_action


def parse(text):
    return parse_action(truncate_after_first_action(text))


@pytest.mark.parametrize(
    "text, expected",
    [
        # the original, labelled format
        ("Thought 1: x\nAction 1: Search[Raymond Carver]", ("Search", "Raymond Carver")),
        # pilot habit 1: a bare tool call with no `Action N:` label (HotpotQA)
        ("Search[Enrico Cocozza]", ("Search", "Enrico Cocozza")),
        ("I apologize for the formatting error. Let me try again.\n\nSearch[Foo (film)]",
         ("Search", "Foo (film)")),
        ("Finish[Arthur's Magazine [1844]]", ("Finish", "Arthur's Magazine [1844]")),
        ("lookup[eastern sector]", ("Lookup", "eastern sector")),
        # pilot habit 2: markdown around the labels (GSM8K)
        ("Let's go step by step.\n\n**Thought 1:** Rosie runs.\n**Action 1:** Calculate[10 * 3]",
         ("Calculate", "10 * 3")),
        ("**Thought 1**: a\n**Action 1**: Calculate[2**3]", ("Calculate", "2**3")),
    ],
)
def test_accepted_forms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Thought 3: The passage says Bass Rock is in the Firth of Forth.",  # no action at all
        "Some text mentioning Search[x] mid-line",  # not at the start of a line
        "Unknowntool[x]",  # bare lines are accepted for known tools only
        "Let's break it down step by step.\n**Step 1 – Wheels** 57 × 4 = 228",
    ],
)
def test_rejected_forms(text):
    assert parse(text) is None


def test_labelled_action_wins_over_a_tool_mentioned_in_the_thought():
    text = "Thought 1: I will Search[x] soon\nAction 1: Lookup[y]"
    assert parse(text) == ("Lookup", "y")


def test_only_the_first_action_is_kept_and_stored_text_is_normalised():
    text = "**Thought 1:** a\n**Action 1:** Search[x]\nObservation 1: fake\nAction 2: Search[y]"
    stored = truncate_after_first_action(text)
    assert stored == "Thought 1: a\nAction 1: Search[x]"
    assert truncate_after_first_action(stored) == stored  # idempotent


def test_normalize_labels_leaves_arithmetic_alone():
    assert normalize_labels("Calculate[2**3 + 4**2]") == "Calculate[2**3 + 4**2]"
    assert normalize_labels("**Thought 1:**\nI think") == "Thought 1: \nI think"
