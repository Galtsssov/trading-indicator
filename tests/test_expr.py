import pandas as pd
import pytest

from qlab import expr

DF = pd.DataFrame({"a": [1.0, -2.0, float("nan"), 4.0], "s": ["x", "y", "x", "z"]})


def test_basic_logic_and_nan_is_false():
    assert expr.evaluate("a > 0 and s == 'x'", DF).tolist() == [True, False, False, False]
    assert expr.evaluate("abs(a) >= 2 or s in ['z']", DF).tolist() == [False, True, False, True]
    assert expr.evaluate("not (a < 0)", DF).tolist() == [True, False, True, True]  # NaN < 0 → False → not → True
    assert expr.evaluate("0 < a * 2 <= 4", DF).tolist() == [True, False, False, False]
    assert expr.evaluate("s not in ['x']", DF).tolist() == [False, True, False, True]


@pytest.mark.parametrize("bad", ["__import__('os')", "a.real > 0", "open('x')", "[c for c in a]", "lambda: 1"])
def test_unsafe_constructs_rejected(bad):
    with pytest.raises((ValueError, SyntaxError)):
        expr.evaluate(bad, DF)


def test_unknown_column():
    with pytest.raises(ValueError, match="неизвестный признак"):
        expr.evaluate("b > 0", DF)
    assert expr.names("abs(a) > 1 and s == 'x'") == {"a", "s"}
