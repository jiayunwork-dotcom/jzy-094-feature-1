"""沿程扫描测试：时间-河程一致、网格最优点逼近解析临界点、窗口边界处理。"""

import pytest

from app import closed_form as cf
from app.critical import find_critical
from app.scanning import (
    analytic_within_window,
    grid_argmax,
    refine_window_max,
    scan_profile,
)


REF = dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)


def test_profile_carries_both_time_and_distance_with_t_equals_x_over_u():
    t_max = 4.0
    points = scan_profile(**REF, t_max_day=t_max, n_points=401)
    assert len(points) == 401
    for pt in points:
        assert pt.x_km == pytest.approx(pt.t_day * REF["u"])
        assert pt.do_mg_l == pytest.approx(REF["csat"] - pt.deficit_mg_l)
        assert pt.l_mg_l == pytest.approx(cf.bod(pt.t_day, REF["l0"], REF["k1"]))

    assert points[0].t_day == 0.0
    assert points[0].x_km == 0.0
    assert points[-1].t_day == pytest.approx(t_max)
    assert points[-1].x_km == pytest.approx(t_max * REF["u"])


def test_grid_peak_and_refinement_land_on_analytic_critical_point():
    cp = find_critical(**REF)
    t_max = 4.0
    points = scan_profile(**REF, t_max_day=t_max, n_points=401)
    idx = grid_argmax(points)
    # 网格峰点应在解析临界点附近（网格间距 0.01 day）
    assert abs(points[idx].t_day - cp.t_c_day) <= 0.011

    refined = refine_window_max(**REF, t_max_day=t_max, n_points=401)
    assert refined["at_window_boundary"] is False
    assert refined["t_day"] == pytest.approx(cp.t_c_day, abs=1e-6)
    assert refined["x_km"] == pytest.approx(cp.x_c_km, abs=1e-4)
    assert refined["deficit_mg_l"] == pytest.approx(cp.d_c, abs=1e-7)

    assert analytic_within_window(cp, t_max) is True
    assert analytic_within_window(cp, cp.t_c_day - 0.5) is False


def test_window_too_small_reports_peak_at_boundary():
    # 临界点约在 1.96 day；只扫到 0.5 day 时峰贴在窗口右端
    refined = refine_window_max(**REF, t_max_day=0.5, n_points=201)
    assert refined["at_window_boundary"] is True
    cp = find_critical(**REF)
    assert refined["t_day"] < cp.t_c_day
