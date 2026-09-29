"""T5 — `Calculate[expression]`: a safe arithmetic evaluator (the only GSM8K tool)."""

from __future__ import annotations

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

MAX_EXPONENT = 100
MAX_ABS_RESULT = 1e15
ERROR = "Error: invalid expression"


class _Unsafe(Exception):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _Unsafe("non-numeric constant")
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise _Unsafe("unsupported unary operator")
        return op(_eval(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise _Unsafe("unsupported binary operator")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise _Unsafe("exponent too large")
        result = op(left, right)
        if isinstance(result, complex):
            raise _Unsafe("complex result")
        return result
    raise _Unsafe(f"unsupported node {type(node).__name__}")


def _format(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def calculate(expr: str) -> str:
    """Evaluate `expr`; any problem returns the single fixed error string."""
    if not isinstance(expr, str):
        return ERROR
    s = expr.strip().replace("$", "").replace(",", "")
    s = "".join(s.split())
    percent = s.endswith("%")
    if percent:
        s = s[:-1]
    if not s:
        return ERROR
    try:
        tree = ast.parse(s, mode="eval")
        value = _eval(tree)
        if percent:
            value = value / 100
        value = float(value)
    except (_Unsafe, SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError,
            RecursionError, MemoryError):
        return ERROR
    if value != value or value in (float("inf"), float("-inf")):
        return ERROR
    if abs(value) > MAX_ABS_RESULT:
        return ERROR
    return _format(value)
