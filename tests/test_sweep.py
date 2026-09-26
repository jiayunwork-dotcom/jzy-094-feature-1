"""区间批量扫参测试：k2 增大临界亏氧减小、L0 加倍升高、跨无临界点区间、
以及跨 k2=k1 时两侧样本各自选对分支。"""

import pytest

from app.sweep import SweepParams, sweep
from app.validation import ParameterError, validate_sweep_bounds


REF = dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)
BASE = SweepParams(**REF)


def test_sweep_over_k2_reports_shallower_sag_and_decreasing_trend():
    result = sweep(BASE, "k2", 0.45, 1.2, 8)
    assert result["parameter"] == "k2"
    assert len(result["points"]) == 8
    assert result["trend"] == "decreasing"

    deficits = [row["critical_deficit_mg_l"] for row in result["points"]]
    assert all(d is not None for d in deficits)
    for a, b in zip(deficits, deficits[1:]):
        assert b < a  # 只调大复氧系数，临界亏氧变小、氧垂变浅

    # 临界距离同步向上游移动
    xs = [row["x_c_km"] for row in result["points"]]
    for a, b in zip(xs, xs[1:]):
        assert b < a


def test_sweep_over_l0_reports_increasing_deficit():
    result = sweep(BASE, "l0", 10.0, 40.0, 7)
    assert result["trend"] == "increasing"
    deficits = [row["critical_deficit_mg_l"] for row in result["points"]]
    for a, b in zip(deficits, deficits[1:]):
        assert b > a


def test_sweep_into_monotonic_reaeration_marks_missing_critical_points():
    # k2 很大时 k1·L0 <= k2·D0，氧垂消失，必须显式标 exists=false，不得给负河程
    result = sweep(BASE, "k2", 0.6, 6.0, 6)
    rows = result["points"]
    assert rows[0]["exists"] is True
    assert rows[-1]["exists"] is False
    assert rows[-1]["x_c_km"] is None
    assert rows[-1]["critical_deficit_mg_l"] is None
    # 其余存在临界点的行，临界亏氧总体仍随 k2 下降
    valid = [r["critical_deficit_mg_l"] for r in rows if r["exists"]]
    assert len(valid) >= 2
    assert valid[-1] < valid[0]


def test_sweep_d0_across_equal_rates_picks_correct_branches():
    # 扫 k1 跨越 k2=0.3：两侧样本分别落入 general / equal_rates 分支且都给出正距离
    base = SweepParams(**{**REF, "k1": 0.3, "k2": 0.3})
    result = sweep(base, "k1", 0.15, 0.6, 4)  # 不含 0.3 网格点
    for row in result["points"]:
        assert row["exists"] is True
        assert row["x_c_km"] > 0
        assert row["branch"] in ("general", "equal_rates")

    # 单独验证恰好在 k1=k2 的点走特解
    on_point = sweep(SweepParams(**{**REF, "k2": 0.45}), "k1", 0.45, 0.45 + 0.2, 3)
    assert on_point["points"][0]["branch"] == "equal_rates"
    assert on_point["points"][0]["exists"] is True


def test_sweep_velocity_only_scales_critical_distance():
    # 扫流速：临界时刻与临界亏氧不变，临界距离随 U 线性拉长
    result = sweep(BASE, "u", 20.0, 40.0, 5)
    for row in result["points"]:
        assert row["exists"] is True
        assert row["t_c_day"] == pytest.approx(result["points"][0]["t_c_day"])
        assert row["critical_deficit_mg_l"] == pytest.approx(
            result["points"][0]["critical_deficit_mg_l"]
        )
        assert row["x_c_km"] == pytest.approx(row["t_c_day"] * row["value"])


def test_invalid_sweep_bounds_raise():
    with pytest.raises(ParameterError):
        validate_sweep_bounds("k2", 0.9, 0.5, 5)       # lo >= hi
    with pytest.raises(ParameterError):
        validate_sweep_bounds("k2", 0.0, 0.5, 5)       # k2 下界非正
    with pytest.raises(ParameterError):
        validate_sweep_bounds("u", 10.0, 20.0, 1)      # 点数不足
    # L0 允许下界为 0
    lo, hi, n = validate_sweep_bounds("l0", 0.0, 10.0, 4)
    assert (lo, hi, n) == (0.0, 10.0, 4)
    # D0 上界不能越过 Csat
    with pytest.raises(ParameterError):
        validate_sweep_bounds("d0", 0.0, 12.0, 4, csat=10.0)
