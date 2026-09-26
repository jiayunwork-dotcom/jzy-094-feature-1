"""临界点定位锁定测试：
1) 只调大复氧系数 k2，临界亏氧变小、氧垂变浅；
2) k2 = k1 走特解而非一般公式；
3) 流速加倍，临界时刻不变、临界距离按比例加倍；
另含 L0 加倍、无临界点与非法输入的覆盖。
"""

import pytest

from app import closed_form as cf
from app.critical import find_critical, find_critical_numeric
from app.validation import ParameterError, validate_model_inputs


REF = dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)


def test_reference_case_has_clear_sag_at_positive_distance():
    cp = find_critical(**REF)
    assert cp.exists is True
    assert cp.branch == "general"
    assert cp.t_c_day == pytest.approx(1.959, abs=0.01)
    assert cp.x_c_km == pytest.approx(58.8, abs=1.0)
    assert cp.x_c_km > 0
    # 氧垂明显：临界亏氧显著高于初始亏氧，最低 DO 掉到 5 mg/L 以下
    assert cp.d_c > REF["d0"]
    assert cp.d_c == pytest.approx(5.56, abs=0.05)
    assert cp.do_c < 5.0
    # 临界点处 dD/dt 必须为零
    assert cf.deficit_rate(cp.t_c_day, REF["d0"], REF["l0"], REF["k1"], REF["k2"]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_increasing_reaeration_shallows_the_sag():
    # 锁定规律：只把 k2 明显调大，临界亏氧单调变小、最低 DO 升高、氧垂变浅
    deficits = []
    do_mins = []
    x_crit = []
    for k2 in (0.45, 0.6, 0.9, 1.5):
        cp = find_critical(REF["d0"], REF["l0"], REF["k1"], k2, REF["u"], REF["csat"])
        assert cp.exists is True
        deficits.append(cp.d_c)
        do_mins.append(cp.do_c)
        x_crit.append(cp.x_c_km)

    for a, b in zip(deficits, deficits[1:]):
        assert b < a
    for a, b in zip(do_mins, do_mins[1:]):
        assert b > a
    # 临界点随复氧增强向上游移动
    for a, b in zip(x_crit, x_crit[1:]):
        assert b < a


def test_doubling_velocity_doubles_critical_distance_only():
    # 锁定规律：临界时刻由 k1,k2,D0,L0 决定，与流速无关；河程 x_c = U t_c
    cp1 = find_critical(**REF)
    faster = {**REF, "u": REF["u"] * 2}
    cp2 = find_critical(**faster)
    assert cp2.t_c_day == pytest.approx(cp1.t_c_day)
    assert cp2.d_c == pytest.approx(cp1.d_c)
    assert cp2.x_c_km == pytest.approx(2.0 * cp1.x_c_km)

    # 同一时刻对应的河程随流速加倍
    t = 1.0
    assert cf.distance_from_time(t, faster["u"]) == pytest.approx(
        2.0 * cf.distance_from_time(t, REF["u"])
    )


def test_equal_rates_uses_special_branch_formula():
    # k2 = k1：必须走特解 t_c = 1/k - D0/(k L0)，不能套一般分母公式
    case = dict(d0=2.0, l0=20.0, k1=0.4, k2=0.4, u=30.0, csat=10.0)
    cp = find_critical(**case)
    assert cp.exists is True
    assert cp.branch == "equal_rates"

    k, d0, l0 = case["k1"], case["d0"], case["l0"]
    expected_t = 1.0 / k - d0 / (k * l0)
    assert cp.t_c_day == pytest.approx(expected_t)
    assert cp.x_c_km == pytest.approx(expected_t * case["u"])

    # 特解临界点处导数为零
    assert cf.deficit_rate(cp.t_c_day, d0, l0, k, k) == pytest.approx(0.0, abs=1e-9)

    # 与一般公式趋近 k2=k1 的极限一致（证明特解正确而不是另立公式）
    cp_near = find_critical(d0, l0, k, k + 1e-8, case["u"], case["csat"])
    assert cp_near.branch == "general"
    assert cp_near.t_c_day == pytest.approx(cp.t_c_day, rel=1e-5)

    # 数值求根/黄金分割（标准库手写）复核解析特解
    numeric = find_critical_numeric(**case)
    assert numeric["exists"] is True
    assert numeric["t_c_day"] == pytest.approx(cp.t_c_day, abs=1e-7)
    assert numeric["critical_deficit_mg_l"] == pytest.approx(cp.d_c, abs=1e-7)


def test_doubling_initial_bod_raises_maximum_deficit():
    cp1 = find_critical(**REF)
    cp2 = find_critical(REF["d0"], REF["l0"] * 2, REF["k1"], REF["k2"], REF["u"], REF["csat"])
    assert cp2.exists is True
    assert cp2.d_c > cp1.d_c
    assert cp2.do_c < cp1.do_c


def test_monotonic_reaeration_reports_no_critical_point():
    # k1 L0 <= k2 D0：初始亏氧就在下降，曲线单调复氧——明确报告无临界点，
    # 绝不能给出负河程
    case = dict(d0=3.0, l0=5.0, k1=0.2, k2=0.9, u=30.0, csat=10.0)
    assert REF["k1"] * case["l0"] <= case["k2"] * case["d0"]
    cp = find_critical(**case)
    assert cp.exists is False
    assert cp.t_c_day is None
    assert cp.x_c_km is None
    assert cp.d_c is None
    assert cp.do_c is None
    # 整条下游曲线确实不增
    for t in (0.01, 0.5, 2.0, 10.0):
        assert (
            cf.deficit(t, case["d0"], case["l0"], case["k1"], case["k2"])
            <= case["d0"] + 1e-12
        )

    # 零负荷同样没有氧垂
    cp_zero = find_critical(2.0, 0.0, 0.3, 0.6, 30.0, 10.0)
    assert cp_zero.exists is False


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d0=2.0, l0=20.0, k1=0.0, k2=0.6, u=30.0, csat=10.0),      # k1 为零
        dict(d0=2.0, l0=20.0, k1=0.3, k2=-0.6, u=30.0, csat=10.0),     # k2 为负
        dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=0.0, csat=10.0),       # 流速为零
        dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=0.0),       # Csat 为零
        dict(d0=2.0, l0=-1.0, k1=0.3, k2=0.6, u=30.0, csat=10.0),      # L0 为负
        dict(d0=12.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0),     # D0 > Csat
        dict(d0=2.0, l0=20.0, k1=float("nan"), k2=0.6, u=30.0, csat=10.0),
        dict(d0=2.0, l0=20.0, k1=0.3, k2=float("inf"), u=30.0, csat=10.0),
        dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=True, csat=10.0),      # bool 冒充数值
    ],
)
def test_invalid_coefficients_raise(kwargs):
    with pytest.raises(ParameterError):
        validate_model_inputs(**kwargs)
