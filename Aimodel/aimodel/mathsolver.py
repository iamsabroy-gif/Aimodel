"""Arithmetic: exact calculation instead of guessing.

A small language model cannot reliably add or multiply, so maths is done by
code. Questions like "what is 12 * (3 + 4)?", "15% of 200", "square root of
144" or "twelve plus five" are turned into an expression, checked, and
evaluated exactly with a safe evaluator (no `eval`). The steps are returned
so the answer can show its work.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from fractions import Fraction

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
_NUMWORD = re.compile(r"\b(?:" + "|".join(list(_ONES) + list(_TENS) + list(_SCALES)) + r")(?:[\s-]+(?:and\s+)?(?:"
                      + "|".join(list(_ONES) + list(_TENS) + list(_SCALES)) + r"))*\b")

_PHRASES = [  # (pattern, replacement) applied in order to the lower-cased text
    (r"\bto the power of\b|\braised to\b|\bpower\b", "**"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
    (r"\bmultiplied by\b|\btimes\b|\bmultiply\b|\bx\b|×", "*"),
    (r"\bdivided by\b|\bover\b|÷|\bdivide\b", "/"),
    (r"\bplus\b|\badded to\b|\band\b(?=\s*\d)", "+"),
    (r"\bminus\b|\bsubtract(?:ed)?(?: by)?\b|\bless\b", "-"),
    (r"\bmod(?:ulo)?\b|\bremainder of\b", "%%"),
]
_NOISE = re.compile(r"\b(what|whats|what's|is|are|equals?|equal to|calculate|compute|solve|evaluate|"
                    r"find|the|value|of|result|answer|tell|me|please|how|much|many|give|a|an)\b|[?=]|\bto\b")

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_FUNCS = {"sqrt": math.sqrt, "cbrt": lambda v: math.copysign(abs(v) ** (1 / 3), v), "abs": abs,
          "fact": lambda v: math.factorial(int(v)), "ln": math.log, "log": math.log10,
          "round": round, "floor": math.floor, "ceil": math.ceil}


def words_to_numbers(text: str) -> str:
    """'twenty one' -> '21', 'two hundred and five' -> '205'."""
    def convert(m: re.Match) -> str:
        total = current = 0
        for w in re.split(r"[\s-]+", m.group(0)):
            if w == "and":
                continue
            if w in _ONES:
                current += _ONES[w]
            elif w in _TENS:
                current += _TENS[w]
            elif w == "hundred":
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * _SCALES[w]
                current = 0
        return str(total + current)
    return _NUMWORD.sub(convert, text)


def _normalise(text: str) -> str:
    t = text.lower().replace(",", "").replace("’", "'")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = words_to_numbers(t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s*of\s*", r"(\1/100)*", t)
    t = re.sub(r"\bhalf of\s*", "(1/2)*", t)
    t = re.sub(r"\b(?:square root|sqrt)(?: of)?\s*", "sqrt ", t)
    t = re.sub(r"\bcube root(?: of)?\s*", "cbrt ", t)
    t = re.sub(r"\bfactorial of\s*(\d+)|\b(\d+)\s*!", lambda m: f"fact({m.group(1) or m.group(2)})", t)
    t = re.sub(r"\b(sqrt|cbrt|ln|log|abs)\s+(\d+(?:\.\d+)?)", r"\1(\2)", t)
    t = re.sub(r"\b(?:sum|total) of\s*([\d.\s+and]+)", lambda m: "+".join(re.findall(r"\d+(?:\.\d+)?", m.group(1))), t)
    t = re.sub(r"\bproduct of\s*([\d.\s+and]+)", lambda m: "*".join(re.findall(r"\d+(?:\.\d+)?", m.group(1))), t)
    t = re.sub(r"\b(?:average|mean) of\s*([\d.\s,+and]+)",
               lambda m: "(" + "+".join(n := re.findall(r"\d+(?:\.\d+)?", m.group(1))) + f")/{len(n)}", t)
    for pat, rep in _PHRASES:
        t = re.sub(pat, rep, t)
    t = t.replace("^", "**").replace("%%", "%")
    t = _NOISE.sub(" ", t)
    return re.sub(r"\s+", "", t)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b) > 1000 or abs(a) > 10**6:
                return Fraction(float(a) ** float(b)) if abs(b) <= 1000 else _fail()
            return a ** int(b)
        if type(node.op) in _BIN:
            if isinstance(node.op, (ast.Div, ast.Mod, ast.FloorDiv)) and b == 0:
                raise ZeroDivisionError
            return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
            and len(node.args) == 1 and not node.keywords:
        v = _FUNCS[node.func.id](float(_eval(node.args[0])))
        return Fraction(v).limit_denominator(10**9) if not float(v).is_integer() else Fraction(int(v))
    _fail()


def _fail():
    raise ValueError("unsupported")


def _format(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator:,}".replace(",", "") if abs(value) < 10**15 else str(value.numerator)
    s = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return s


def solve(text: str):
    """Return {"answer", "expression", "steps"} if `text` is a calculation, else None."""
    t = text.strip()
    if not re.search(r"\d|\b(?:" + "|".join(list(_ONES) + list(_TENS)) + r")\b", t.lower()):
        return None
    pct = re.fullmatch(r".*?(\d+(?:\.\d+)?)\s*(?:is\s+)?what\s+(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)\??\s*", t.lower())
    if pct and Fraction(pct.group(2)) != 0:
        v = Fraction(pct.group(1)) / Fraction(pct.group(2)) * 100
        return {"answer": _format(v) + "%", "expression": f"{pct.group(1)} / {pct.group(2)} * 100",
                "steps": [f"{pct.group(1)} / {pct.group(2)} * 100 = {_format(v)}"]}
    expr = _normalise(t)
    # Needs a real operation: an operator, a function, or a bracket - not just a lone number.
    if not expr or not re.search(r"[-+*/%()]|fact|sqrt|cbrt|ln|log|abs", expr):
        return None
    if re.search(r"[^\d.+\-*/%()a-z]", expr) or re.search(r"[a-z]+", re.sub(r"sqrt|cbrt|fact|ln|log|abs", "", expr)):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
    except ZeroDivisionError:
        return {"answer": None, "expression": expr, "steps": ["division by zero is undefined"],
                "error": "Dividing by zero is undefined."}
    except (SyntaxError, ValueError, OverflowError, TypeError, RecursionError):
        return None
    pretty = re.sub(r"(?<=[\d)])([-+*/%])(?=[\d(])", r" \1 ", expr).replace("* *", "**")
    steps = _steps(tree)
    return {"answer": _format(value), "expression": pretty, "steps": steps}


def _steps(tree) -> list[str]:
    """Evaluate bottom-up, recording each operation: (3 + 4) * 5 -> 3 + 4 = 7; 7 * 5 = 35."""
    out: list[str] = []
    sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.Pow: "^",
           ast.FloorDiv: "//"}

    def go(n):
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp):
            v = go(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right)
            v = _eval(ast.BinOp(ast.Constant(float(a) if a.denominator != 1 else int(a)), n.op,
                                ast.Constant(float(b) if b.denominator != 1 else int(b))))
            out.append(f"{_format(a)} {sym[type(n.op)]} {_format(b)} = {_format(v)}")
            return v
        if isinstance(n, ast.Call):
            a = go(n.args[0])
            v = _eval(ast.Call(n.func, [ast.Constant(float(a))], []))
            out.append(f"{n.func.id}({_format(a)}) = {_format(v)}")
            return v
        _fail()
    go(tree)
    return out
