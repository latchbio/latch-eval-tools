import math


def is_finite_number(value: object) -> bool:
    """Return whether a JSON number is finite and representable as a float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        parsed = float(value)
    except OverflowError:
        return False
    return math.isfinite(parsed)
