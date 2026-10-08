"""Unit conversion: "convert 5 km to miles", "how many meters in 3 km?", "100 F in celsius".

Exact factors, no guessing. Only fires when both units are known and measure the same kind of thing.
"""

from __future__ import annotations

import re

from .mathsolver import words_to_numbers

# unit name -> (kind, how many base units). Base units: metre, gram, litre, second.
_UNITS: dict[str, tuple[str, float]] = {}


def _add(kind: str, factor: float, *names: str) -> None:
    for n in names:
        _UNITS[n] = (kind, factor)


_add("length", 1, "m", "meter", "meters", "metre", "metres")
_add("length", 1000, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_add("length", 0.01, "cm", "centimeter", "centimeters", "centimetre", "centimetres")
_add("length", 0.001, "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_add("length", 1609.344, "mile", "miles", "mi")
_add("length", 0.3048, "foot", "feet", "ft")
_add("length", 0.0254, "inch", "inches", "in")
_add("length", 0.9144, "yard", "yards", "yd")
_add("mass", 1000, "kg", "kilogram", "kilograms", "kilo", "kilos")
_add("mass", 1, "g", "gram", "grams")
_add("mass", 0.001, "mg", "milligram", "milligrams")
_add("mass", 453.59237, "lb", "lbs", "pound", "pounds")
_add("mass", 28.349523125, "oz", "ounce", "ounces")
_add("mass", 1_000_000, "tonne", "tonnes", "metric ton", "metric tons")
_add("volume", 1, "l", "liter", "liters", "litre", "litres")
_add("volume", 0.001, "ml", "milliliter", "milliliters", "millilitre", "millilitres")
_add("volume", 3.785411784, "gallon", "gallons", "gal")
_add("volume", 0.2365882365, "cup", "cups")
_add("time", 1, "s", "sec", "second", "seconds")
_add("time", 60, "min", "minute", "minutes")
_add("time", 3600, "h", "hr", "hour", "hours")
_add("time", 86400, "day", "days")
_add("time", 604800, "week", "weeks")
_add("time", 31_557_600, "year", "years")
_TEMP = {"c": "c", "celsius": "c", "centigrade": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_NAME = r"[a-z]+(?: [a-z]+)?"
_PATTERNS = [
    re.compile(r"^(?:please )?(?:convert|change)\s+(?P<n>" + _NUM + r")\s*°?\s*(?P<a>" + _NAME + r")\s+(?:to|into|in)\s+°?(?P<b>[a-z]+(?: [a-z]+)?)$"),
    re.compile(r"^(?:how many|what is|what's)\s+(?P<b>[a-z]+)\s+(?:are |is )?in\s+(?P<n>" + _NUM + r")\s*°?\s*(?P<a>[a-z]+(?: [a-z]+)?)$"),
    re.compile(r"^(?:what is|what's|how many)?\s*(?P<n>" + _NUM + r")\s*°?\s*(?P<a>" + _NAME + r")\s+(?:in|to|into|equals? how many)\s+°?(?P<b>[a-z]+(?: [a-z]+)?)$"),
]


def _fmt(v: float) -> str:
    return f"{v:.6g}" if abs(v) >= 1e-4 or v == 0 else f"{v:.3e}"


def _find(name: str):
    name = name.strip()
    return _UNITS.get(name) or _UNITS.get(name.rstrip("s")) or None


def convert(text: str):
    """{"answer", "expression", "steps"} for a unit conversion, else None."""
    t = words_to_numbers(text.strip().lower().rstrip("?.! ")).replace("degrees ", "").replace("degree ", "")
    t = re.sub(r"\s+", " ", t)
    for pattern in _PATTERNS:
        m = pattern.match(t)
        if not m:
            continue
        n = float(m.group("n").replace(",", ""))
        a, b = m.group("a").strip(), m.group("b").strip()
        ta, tb = _TEMP.get(a), _TEMP.get(b)
        if ta and tb:
            c = {"c": n, "f": (n - 32) * 5 / 9, "k": n - 273.15}[ta]
            out = {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[tb]
            return {"answer": f"{_fmt(out)} {b}", "expression": f"{_fmt(n)} {a} in {b}",
                    "steps": [f"{_fmt(n)} {a} = {_fmt(out)} {b}"]}
        ua, ub = _find(a), _find(b)
        if ua and ub and ua[0] == ub[0]:
            out = n * ua[1] / ub[1]
            return {"answer": f"{_fmt(out)} {b}", "expression": f"{_fmt(n)} {a} in {b}",
                    "steps": [f"{_fmt(n)} {a} x {_fmt(ua[1])} / {_fmt(ub[1])} = {_fmt(out)} {b}"]}
    return None
