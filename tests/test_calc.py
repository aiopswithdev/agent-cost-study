import pytest

from acs.tools.calc import ERROR, calculate


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("48 / 2", "24"),
        ("48 + 24", "72"),
        ("12 / 60 * 50", "10"),
        ("100 - 100 / 2 - 15 - 15 * 2", "5"),
        ("100 / 2", "50"),
        ("15 * 2", "30"),
        ("100 - 50 - 15 - 30", "5"),
    ],
)
def test_exemplar_observations(expr, expected):
    """These are the observations printed in the GSM8K few-shots (T6-C, T6-D)."""
    assert calculate(expr) == expected


def test_formatting():
    assert calculate("7/3") == "2.333333"
    assert calculate("1/8") == "0.125"
    assert calculate("2 ** 10") == "1024"
    assert calculate("-3 - 4") == "-7"


def test_cleaning():
    assert calculate("$1,200 + 3") == "1203"
    assert calculate(" 50% ") == "0.5"
    assert calculate("  48\t/ 2 ") == "24"


@pytest.mark.parametrize(
    "expr",
    ["__import__('os')", "os.system('ls')", "1/0", "2**1000", "10**20", "", "abc",
     "[1,2]", "lambda: 1", "True", "1 if 2 else 3"],
)
def test_rejects_unsafe_or_invalid(expr):
    assert calculate(expr) == ERROR
