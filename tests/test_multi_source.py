"""多排污口叠加内核测试。

四条必须锁死的自洽规律：
1. 单口退化：只放一个排口时，多口接口的曲线/临界点/临界亏氧必须与原
   单口闭式接口在数值上完全对得上；
2. 同位合并：两个位置完全相同的排口 == 一个负荷翻倍的单口；
3. 下游因果：任一河程点只统计位置不晚于它的排口；下游新插入口子不能
   改动它上游已经算出的曲线与临界点；
4. 顺序不敏感：提交顺序打乱、位置负荷不变，曲线与全部临界点逐位不变。

另含双峰/三峰发现、无临界点如实报告、k2=k1 特解段、分段重组式与直和式
逐位相等、自动窗口包得住所有峰等锁定。
"""

import math

import pytest

from app import closed_form as cf
from app.critical import find_critical
from app.multi_critical import (
    default_scan_end,
    find_critical_multi,
    segment_deficit,
    segment_rate,
)
from app.multi_source import (
    build_river,
    river_bod,
    river_deficit,
    river_deficit_at_distance,
    river_rate,
    scan_multi_profile,
)
from app.validation import MAX_OUTFALLS, ParameterError, validate_multi_inputs

K1, K2, U, CSAT = 0.3, 0.6, 30.0, 10.0
BASE_KW = dict(k1=K1, k2=K2, u=U, csat=CSAT)
D0 = 2.0


# --------------------------------------------------------------------------
# 规律 1：单口退化一致（最容易被偷懒实现破坏，重点锁）
# --------------------------------------------------------------------------


def test_single_outfall_degenerate_matches_closed_form_critical_point():
    river = build_river(D0, **BASE_KW, outfalls=[(0.0, 20.0)])
    multi = find_critical_multi(river)
    single = find_critical(D0, 20.0, K1, K2, U, CSAT)

    assert multi.exists is True
    assert len(multi.points) == 1
    gp = multi.global_point
    assert gp.is_global is True
    # 数值搜索 vs 解析公式：位置容差取数值方法的实际精度水平
    assert gp.t_c_day == pytest.approx(single.t_c_day, abs=1e-6)
    assert gp.x_c_km == pytest.approx(single.x_c_km, abs=3e-5)
    # 临界亏氧是同一闭式函数直接取值，必须到机器精度
    assert gp.d_c == pytest.approx(single.d_c, abs=1e-12)
    assert gp.do_c == pytest.approx(single.do_c, abs=1e-12)
    # 参考工况的教科书数值
    assert gp.x_c_km == pytest.approx(58.8, abs=1.0)
    assert gp.d_c == pytest.approx(5.56, abs=0.01)


def test_single_outfall_degenerate_matches_closed_form_along_whole_profile():
    river = build_river(D0, **BASE_KW, outfalls=[(0.0, 20.0)])
    for t in (0.0, 0.1, 0.5, 1.7, 1.96, 4.0, 12.0):
        assert river_deficit(river, t) == pytest.approx(
            cf.deficit(t, D0, 20.0, K1, K2), abs=1e-12
        )
        assert river_bod(river, t) == pytest.approx(cf.bod(t, 20.0, K1), abs=1e-12)
        assert river_rate(river, t) == pytest.approx(
            cf.deficit_rate(t, D0, 20.0, K1, K2), abs=1e-12
        )


def test_single_outfall_at_nonzero_position_matches_shifted_closed_form():
    # 排口不在河头：x>0 之前只有本底衰减，之后曲线就是把单口闭式解时间
    # 原点挪到该口子所在河程
    x_i, l_i = 45.0, 17.0
    river = build_river(D0, **BASE_KW, outfalls=[(x_i, l_i)])
    t_i = cf.time_from_distance(x_i, U)
    for x in (0.0, 10.0, 44.9, 45.0, 60.0, 200.0):
        t = cf.time_from_distance(x, U)
        expected_bg = cf.deficit(t, D0, 0.0, K1, K2)
        if t >= t_i:
            expected = expected_bg + cf.deficit(t - t_i, 0.0, l_i, K1, K2)
        else:
            expected = expected_bg
        assert river_deficit(river, t) == pytest.approx(expected, abs=1e-12)


def test_single_outfall_equal_rates_matches_special_branch():
    k = 0.4
    river = build_river(D0, k1=k, k2=k, u=U, csat=CSAT, outfalls=[(0.0, 20.0)])
    multi = find_critical_multi(river)
    single = find_critical(D0, 20.0, k, k, U, CSAT)
    assert multi.exists is True
    assert multi.global_point.t_c_day == pytest.approx(single.t_c_day, abs=1e-6)
    assert multi.global_point.d_c == pytest.approx(single.d_c, abs=1e-12)
    for t in (0.0, 0.3, 2.5):
        assert river_deficit(river, t) == pytest.approx(
            (D0 + k * 20.0 * t) * math.exp(-k * t), abs=1e-12
        )


# --------------------------------------------------------------------------
# 规律 2：同位排口负荷相加
# --------------------------------------------------------------------------


def test_coincident_outfalls_equal_one_doubled_load_outfall():
    pair = build_river(D0, **BASE_KW, outfalls=[(30.0, 10.0), (30.0, 10.0)])
    single = build_river(D0, **BASE_KW, outfalls=[(30.0, 20.0)])
    assert len(pair.outfalls) == 1
    assert pair.outfalls[0].l0 == pytest.approx(20.0)
    assert pair.n_coincident_merges == 1

    for t in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
        # 线性叠加的直接推论：逐位相等，不是“接近”
        assert river_deficit(pair, t) == river_deficit(single, t)
        assert river_bod(pair, t) == river_bod(single, t)

    a = find_critical_multi(pair)
    b = find_critical_multi(single)
    assert a.global_point.t_c_day == b.global_point.t_c_day
    assert a.global_point.d_c == b.global_point.d_c
    assert len(a.points) == len(b.points) == 1


def test_three_way_coincident_loads_sum_with_math_fsum_accuracy():
    loads = [7.1, 8.3, 4.6]
    river = build_river(D0, **BASE_KW, outfalls=[(12.0, l) for l in loads])
    assert len(river.outfalls) == 1
    assert river.outfalls[0].l0 == pytest.approx(math.fsum(loads))
    direct = build_river(D0, **BASE_KW, outfalls=[(12.0, math.fsum(loads))])
    for t in (0.5, 2.0, 8.0):
        assert river_deficit(river, t) == river_deficit(direct, t)


# --------------------------------------------------------------------------
# 规律 3：下游因果——后面的口子不能改动上游答案
# --------------------------------------------------------------------------


def test_downstream_outfall_does_not_change_upstream_profile_or_peaks():
    base = build_river(D0, **BASE_KW, outfalls=[(0.0, 15.0), (40.0, 12.0)])
    extended = build_river(
        D0, **BASE_KW, outfalls=[(0.0, 15.0), (40.0, 12.0), (200.0, 25.0)]
    )
    t_window = 100.0 / U  # 新口子在窗口之外

    p1 = scan_multi_profile(base, t_window, 337)
    p2 = scan_multi_profile(extended, t_window, 337)
    for a, b in zip(p1, p2):
        assert a.deficit_mg_l == b.deficit_mg_l
        assert a.l_mg_l == b.l_mg_l

    c1 = find_critical_multi(base, t_window)
    c2 = find_critical_multi(extended, t_window)
    assert [p.t_c_day for p in c1.points] == [p.t_c_day for p in c2.points]
    assert [p.d_c for p in c1.points] == [p.d_c for p in c2.points]
    # 新口子自己的峰在窗口外，窗口内峰数不变
    assert len(c1.points) == len(c2.points)


def test_only_outfalls_not_later_than_point_contribute():
    river = build_river(
        D0, **BASE_KW, outfalls=[(10.0, 5.0), (50.0, 9.0), (120.0, 14.0)]
    )
    for idx, o in enumerate(river.outfalls):
        t_inject = cf.time_from_distance(o.x_km, U)
        t_just_before = t_inject - 1e-12
        t_just_after = t_inject + 1e-12
        d_before = river_deficit(river, t_just_before)
        d_after = river_deficit(river, t_just_after)
        # 注入瞬间新增 k1·L 的耗氧源，亏氧连续但斜率上跳；
        # 该口对注入前一刻的贡献必须为 0
        assert d_after == pytest.approx(d_before, abs=1e-11)
        # 注入前的贡献等于去掉该口后的曲线
        others = build_river(
            D0,
            **BASE_KW,
            outfalls=[(oo.x_km, oo.l0) for j, oo in enumerate(river.outfalls) if j != idx],
        )
        assert river_deficit(river, t_just_before) == pytest.approx(
            river_deficit(others, t_just_before), abs=1e-11
        )


# --------------------------------------------------------------------------
# 规律 4：顺序不敏感（逐位相等，不是容差近似）
# --------------------------------------------------------------------------


def test_shuffled_submission_order_gives_bit_identical_profile_and_peaks():
    outfalls = [(0.0, 10.0), (50.0, 15.0), (20.0, 8.0), (90.0, 6.0), (50.0, 5.0)]
    shuffled = [outfalls[i] for i in (3, 0, 4, 2, 1)]

    a = build_river(D0, **BASE_KW, outfalls=outfalls)
    b = build_river(D0, **BASE_KW, outfalls=shuffled)

    # 合并后的规范化排口完全一致
    assert a.outfalls == b.outfalls

    t_end = 6.0
    pa = scan_multi_profile(a, t_end, 777)
    pb = scan_multi_profile(b, t_end, 777)
    for x, y in zip(pa, pb):
        assert x.t_day == y.t_day
        assert x.x_km == y.x_km
        assert x.deficit_mg_l == y.deficit_mg_l
        assert x.l_mg_l == y.l_mg_l
        assert x.do_mg_l == y.do_mg_l

    ca = find_critical_multi(a, t_end)
    cb = find_critical_multi(b, t_end)
    assert len(ca.points) == len(cb.points)
    for x, y in zip(ca.points, cb.points):
        assert x.t_c_day == y.t_c_day
        assert x.x_c_km == y.x_c_km
        assert x.d_c == y.d_c
        assert x.is_global == y.is_global
        assert x.rank == y.rank
    assert ca.global_point.t_c_day == cb.global_point.t_c_day


# --------------------------------------------------------------------------
# 多峰发现与全局/次级标注
# --------------------------------------------------------------------------


def test_two_widely_separated_outfalls_produce_two_local_maxima():
    # 两个口子隔得够远：先鼓一个峰、衰减后被第二口顶起第二个更深的峰
    river = build_river(
        D0, k1=0.35, k2=0.6, u=U, csat=CSAT, outfalls=[(0.0, 18.0), (80.0, 18.0)]
    )
    res = find_critical_multi(river)
    assert res.exists is True
    assert len(res.points) == 2

    peaks = sorted(res.points, key=lambda p: p.x_c_km)
    first, second = peaks
    assert not first.is_global and first.rank == 2
    assert second.is_global and second.rank == 1
    # 两个峰分别位于各自排口下游的正河程处
    assert 0.0 < first.x_c_km < 80.0
    assert second.x_c_km > 80.0
    assert second.d_c > first.d_c

    # 每个峰处 dD/dt ≈ 0，且两侧符号为 +、-
    for p in peaks:
        h = 1e-4
        assert river_rate(river, p.t_c_day - h) > 0
        assert river_rate(river, p.t_c_day + h) < 0


def test_three_outfalls_produce_three_ranked_peaks_and_global_is_deepest():
    river = build_river(
        D0,
        k1=0.35,
        k2=0.7,
        u=U,
        csat=CSAT,
        outfalls=[(0.0, 16.0), (60.0, 16.0), (120.0, 16.0)],
    )
    res = find_critical_multi(river)
    assert len(res.points) == 3
    assert {p.rank for p in res.points} == {1, 2, 3}
    globals_ = [p for p in res.points if p.is_global]
    assert len(globals_) == 1
    assert res.global_point is globals_[0]
    # 全局最深峰就是临界亏氧最大的那个
    assert res.global_point.d_c == max(p.d_c for p in res.points)
    # 自动窗口终点已在下降段，保证峰没有落在窗口外
    assert res.still_rising_at_end is False
    assert river_rate(river, res.t_scan_max_day) <= 1e-9


def test_auto_scan_window_always_covers_all_peaks():
    river = build_river(
        D0, k1=0.1, k2=0.15, u=10.0, csat=CSAT, outfalls=[(0.0, 30.0)]
    )
    t_end = default_scan_end(river)
    res = find_critical_multi(river)
    for p in res.points:
        assert p.t_c_day < t_end
    # 自动窗口终点导数确实 <= 0
    assert segment_rate(river, len(river.outfalls) - 1, t_end) <= 0.0


# --------------------------------------------------------------------------
# 无临界点：不能凑正数
# --------------------------------------------------------------------------


def test_no_critical_point_when_reaeration_always_dominates():
    # 所有口子负荷都很小：起点与每个注入点后的导数都不抬头
    river = build_river(
        d0=1.0, **BASE_KW, outfalls=[(10.0, 0.5), (60.0, 0.3), (120.0, 0.2)]
    )
    res = find_critical_multi(river)
    assert res.exists is False
    assert res.points == ()
    assert res.global_point is None
    # 沿全程（含每个注入点之后）亏氧确实不上升
    t_end = res.t_scan_max_day
    positions = river.t_positions
    for t_i in [0.0] + list(positions):
        for dt in (1e-6, 1e-3, 1e-1):
            t = t_i + dt
            if t < t_end:
                assert river_rate(river, t) <= 1e-9


def test_zero_load_outfalls_and_pure_background_report_no_critical_point():
    river = build_river(D0, **BASE_KW, outfalls=[(0.0, 0.0), (50.0, 0.0)])
    res = find_critical_multi(river)
    assert res.exists is False
    # 纯本底 D0·e^{-k2 t} 单调衰减
    for t in (0.01, 1.0, 5.0):
        assert river_deficit(river, t) == pytest.approx(D0 * math.exp(-K2 * t))


def test_manual_window_ending_mid_rise_flags_possible_peak_beyond():
    # 给一个故意截得很短的窗口：终点仍在上升时必须显式标注，不假装全了
    river = build_river(D0, **BASE_KW, outfalls=[(0.0, 20.0)])
    res = find_critical_multi(river, t_max_day=0.5)
    assert res.still_rising_at_end is True
    # 窗口内没有完整的峰
    assert res.exists is False


# --------------------------------------------------------------------------
# 分段重组式 vs 闭式项直和式：只能换括号，不能换公式
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k1,k2", [(0.3, 0.6), (0.6, 0.3), (0.1, 0.1000001), (0.5, 0.5)])
def test_segment_regrouped_expressions_equal_direct_closed_form_sum(k1, k2):
    river = build_river(
        D0,
        k1=k1,
        k2=k2,
        u=U,
        csat=CSAT,
        outfalls=[(0.0, 12.0), (30.0, 8.0), (90.0, 5.0)],
    )
    n = len(river.outfalls)
    positions = river.t_positions
    test_ts = [0.0]
    for t_i in positions:
        test_ts += [t_i + 1e-7, t_i + 0.5, t_i + 2.0]
    for through in range(-1, n - 1):
        # through 对应段：只可在该段活动集合稳定的时间上比较
        lo_t = 0.0 if through < 0 else positions[through]
        hi_t = positions[through + 1]
        for t in [lo_t, (lo_t + hi_t) / 2, hi_t - 1e-9]:
            assert segment_deficit(river, through, t) == pytest.approx(
                river_deficit(river, t), abs=1e-10
            )
            assert segment_rate(river, through, t) == pytest.approx(
                river_rate(river, t, through_index=through), abs=1e-8
            )
    # 最后一段
    through = n - 1
    lo_t = positions[-1]
    for t in [lo_t, lo_t + 0.5, lo_t + 5.0]:
        assert segment_deficit(river, through, t) == pytest.approx(
            river_deficit(river, t), abs=1e-10
        )
        assert segment_rate(river, through, t) == pytest.approx(
            river_rate(river, t), abs=1e-8
        )


# --------------------------------------------------------------------------
# 河程换算与曲线输出
# --------------------------------------------------------------------------


def test_profile_points_respect_x_equals_u_times_t_and_do_equals_csat_minus_d():
    river = build_river(D0, **BASE_KW, outfalls=[(0.0, 10.0), (60.0, 8.0)])
    points = scan_multi_profile(river, 5.0, 501)
    assert len(points) == 501
    assert points[0].x_km == 0.0
    assert points[0].deficit_mg_l == pytest.approx(D0)
    for pt in points:
        assert pt.x_km == pytest.approx(pt.t_day * U)
        assert pt.do_mg_l == pytest.approx(CSAT - pt.deficit_mg_l)
    assert points[-1].t_day == pytest.approx(5.0)


def test_deficit_at_distance_converts_time_before_formula():
    river = build_river(D0, **BASE_KW, outfalls=[(15.0, 12.0)])
    for x in (0.0, 30.0, 75.0, 200.0):
        assert river_deficit(
            river, cf.time_from_distance(x, U)
        ) == pytest.approx(river_deficit_at_distance(river, x), abs=1e-14)


# --------------------------------------------------------------------------
# 非法输入：必须在真正计算之前挡住，且能定位到具体口子
# --------------------------------------------------------------------------


def _validate(outfalls, **overrides):
    kw = dict(d0=D0, k1=K1, k2=K2, u=U, csat=CSAT)
    kw.update(overrides)
    return validate_multi_inputs(outfalls=outfalls, **kw)


def test_empty_outfall_list_rejected_before_computation():
    with pytest.raises(ParameterError, match="不能为空"):
        _validate([])


def test_non_list_outfalls_rejected():
    with pytest.raises(ParameterError):
        _validate({"x_km": 0.0, "l0": 10.0})
    with pytest.raises(ParameterError):
        _validate(None)


def test_negative_position_reported_with_index():
    with pytest.raises(ParameterError, match=r"排污口\[2\]") as exc:
        _validate([(0.0, 5.0), (10.0, 5.0), (-3.0, 5.0)])
    assert "x_km" in str(exc.value)


def test_negative_load_reported_with_index():
    with pytest.raises(ParameterError, match=r"排污口\[0\].*l0"):
        _validate([(0.0, -1.0)])


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        True,
        "abc",
        None,
    ],
)
def test_non_finite_or_non_numeric_position_and_load_rejected(bad):
    with pytest.raises(ParameterError, match=r"排污口\[1\]"):
        _validate([(0.0, 5.0), (bad, 5.0)])
    with pytest.raises(ParameterError, match=r"排污口\[1\]"):
        _validate([(0.0, 5.0), (10.0, bad)])


def test_missing_field_rejected():
    with pytest.raises(ParameterError, match=r"排污口\[0\]"):
        _validate([{"x_km": 1.0}])


def test_too_many_outfalls_rejected():
    too_many = [(float(i), 1.0) for i in range(MAX_OUTFALLS + 1)]
    with pytest.raises(ParameterError, match=str(MAX_OUTFALLS + 1)):
        _validate(too_many)
    # 上限本身合法（空/越界之外不额外限制真实规模）
    result = _validate([(0.0, 1.0), (1.0, 0.0)])
    assert result["n_submitted"] == 2


def test_shared_coefficients_still_validated():
    with pytest.raises(ParameterError):
        _validate([(0.0, 5.0)], k1=0.0)
    with pytest.raises(ParameterError):
        _validate([(0.0, 5.0)], u=-1.0)
    with pytest.raises(ParameterError):
        _validate([(0.0, 5.0)], d0=99.0)  # D0 > Csat
