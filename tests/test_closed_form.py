"""闭式解求值的锁定测试：初值、ODE 右端、k2=k1 特解、时间-河程换算。"""

import math

import pytest

from app import closed_form as cf


REF = dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)


def test_initial_values_match_boundary_conditions():
    # D(0) = D0，L(0) = L0，DO(0) = Csat - D0
    assert cf.deficit(0.0, REF["d0"], REF["l0"], REF["k1"], REF["k2"]) == pytest.approx(2.0)
    assert cf.bod(0.0, REF["l0"], REF["k1"]) == pytest.approx(20.0)
    assert cf.dissolved_oxygen(
        0.0, REF["d0"], REF["l0"], REF["k1"], REF["k2"], REF["csat"]
    ) == pytest.approx(8.0)


def test_closed_form_satisfies_ode():
    # 闭式解必须满足 dD/dt = k1 L - k2 D（对一般分支用中心差分数值核验）
    d0, l0, k1, k2 = REF["d0"], REF["l0"], REF["k1"], REF["k2"]
    h = 1e-6
    for t in (0.1, 0.5, 1.7, 4.0):
        numerical_deriv = (
            cf.deficit(t + h, d0, l0, k1, k2) - cf.deficit(t - h, d0, l0, k1, k2)
        ) / (2 * h)
        rhs = k1 * cf.bod(t, l0, k1) - k2 * cf.deficit(t, d0, l0, k1, k2)
        assert numerical_deriv == pytest.approx(rhs, rel=1e-5, abs=1e-7)


def test_equal_rates_uses_special_formula():
    # k2 = k1 = k：特解 D(t) = (D0 + k L0 t) e^{-kt}，且与一般分支的极限连续
    d0, l0, k = 1.0, 15.0, 0.4
    for t in (0.0, 0.3, 2.5):
        expected = (d0 + k * l0 * t) * math.exp(-k * t)
        assert cf.deficit(t, d0, l0, k, k) == pytest.approx(expected)

    # 一般公式在 k2 -> k1 时应收敛到特解，证明特解不是另一套模型
    near = cf.deficit(1.0, d0, l0, k, k + 1e-9)
    special = cf.deficit(1.0, d0, l0, k, k)
    assert near == pytest.approx(special, rel=1e-6)


def test_equal_rates_satisfies_ode():
    d0, l0, k = 1.5, 12.0, 0.5
    h = 1e-6
    for t in (0.2, 1.0, 3.0):
        numerical_deriv = (
            cf.deficit(t + h, d0, l0, k, k) - cf.deficit(t - h, d0, l0, k, k)
        ) / (2 * h)
        rhs = k * cf.bod(t, l0, k) - k * cf.deficit(t, d0, l0, k, k)
        assert numerical_deriv == pytest.approx(rhs, rel=1e-5, abs=1e-7)


def test_distance_evaluation_converts_time_before_formula():
    # D(x) 必须先做 t = x/U，绝不允许把公里数直接代入
    x_km = 45.0
    via_distance = cf.deficit_at_distance(
        x_km, REF["d0"], REF["l0"], REF["k1"], REF["k2"], REF["u"]
    )
    via_time = cf.deficit(
        cf.time_from_distance(x_km, REF["u"]),
        REF["d0"], REF["l0"], REF["k1"], REF["k2"],
    )
    assert via_distance == pytest.approx(via_time)

    # 45 km / 30 km/day = 1.5 day；直接把 45 当时间会得到完全不同的错误结果
    assert cf.time_from_distance(45.0, 30.0) == pytest.approx(1.5)
    wrong = cf.deficit(45.0, REF["d0"], REF["l0"], REF["k1"], REF["k2"])
    assert abs(wrong - via_distance) > 1.0
