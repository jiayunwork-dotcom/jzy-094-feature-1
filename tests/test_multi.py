"""多排污口叠加工况测试。

锁死的四条叠加规律：
1) 单口退化：只放一个排污口（x=0）时，多口接口给出的曲线、临界点、
   临界亏氧必须与单口接口数值一致——对不上就是叠加逻辑错了；
2) 同位合并：两个位置完全相同的口子 ≡ 一个负荷为两者之和的口子，
   验证叠加不漏算、不重复算；
3) 因果性：下游新增口子不改变它上游已算出的曲线与临界点；
4) 顺序不敏感：口子提交顺序打乱，整条曲线与全部临界点逐位不变。

另覆盖：多峰识别与全局/次级标注、无临界点如实报告、叠加合成的物理
正确性（本底衰减 + 已过口子贡献）、k1=k2 特解路径、非法输入拦截。
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import closed_form as cf
from app.critical import find_critical
from app.main import app
from app.multi_critical import find_all_peaks
from app.multi_profile import (
    prepare_outfalls,
    scan_multi_profile,
    total_bod,
    total_deficit,
    total_deficit_rate,
)
from app.validation import (
    MAX_OUTFALLS,
    ParameterError,
    validate_outfalls,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


SHARED = dict(d0=2.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)
REF_SINGLE = dict(SHARED, l0=20.0)


def multi_body(outfalls, **over):
    body = dict(SHARED)
    body["outfalls"] = outfalls
    body.update(over)
    return body


def make_outfalls(*pairs, u=SHARED["u"]):
    return prepare_outfalls(
        validate_outfalls([{"x_km": x, "l0": load} for x, load in pairs]), u
    )


# --------------------------------------------------------------------------
# 规律 1：单口退化一致
# --------------------------------------------------------------------------


def test_degenerate_curve_matches_closed_form_bitwise():
    ofs = make_outfalls((0.0, 20.0))
    for t in (0.0, 0.3, 1.0, 1.9592888830070636, 4.0, 10.0):
        assert total_deficit(t, SHARED["d0"], ofs, 0.3, 0.6) == cf.deficit(
            t, 2.0, 20.0, 0.3, 0.6
        )
        assert total_bod(t, ofs, 0.3) == cf.bod(t, 20.0, 0.3)


def test_degenerate_critical_matches_analytic():
    ofs = make_outfalls((0.0, 20.0))
    mc = find_all_peaks(SHARED["d0"], ofs, 0.3, 0.6, 30.0, 10.0)
    cp = find_critical(**REF_SINGLE)
    assert cp.exists and mc.exists and mc.n_peaks == 1
    g = mc.global_peak
    assert g.is_global is True
    assert g.t_day == pytest.approx(cp.t_c_day, abs=1e-9)
    assert g.x_km == pytest.approx(cp.x_c_km, abs=1e-8)
    assert g.deficit_mg_l == pytest.approx(cp.d_c, abs=1e-10)
    assert g.do_mg_l == pytest.approx(cp.do_c, abs=1e-10)


def test_api_degenerate_matches_single_endpoints(client):
    body = multi_body([{"x_km": 0.0, "l0": 20.0}], x_max_km=120.0, n_points=401)
    r = client.post("/multi/profile", json=body)
    assert r.status_code == 200
    multi = r.json()

    r1 = client.post("/profile", json=dict(REF_SINGLE, x_max_km=120.0, n_points=401))
    single = r1.json()
    assert len(multi["profile"]) == len(single["profile"]) == 401
    for pm, ps in zip(multi["profile"], single["profile"]):
        assert pm["t_day"] == ps["t_day"]
        assert pm["x_km"] == ps["x_km"]
        assert pm["deficit_mg_l"] == ps["deficit_mg_l"]  # 逐位一致
        assert pm["do_mg_l"] == ps["do_mg_l"]
        assert pm["bod_mg_l"] == ps["bod_mg_l"]

    cp = client.post("/critical", json=REF_SINGLE).json()["critical"]
    assert multi["critical"]["n_peaks"] == 1
    g = multi["critical"]["global"]
    assert g["t_c_day"] == pytest.approx(cp["t_c_day"], abs=1e-9)
    assert g["x_c_km"] == pytest.approx(cp["x_c_km"], abs=1e-8)
    assert g["critical_deficit_mg_l"] == pytest.approx(
        cp["critical_deficit_mg_l"], abs=1e-10
    )
    assert g["critical_do_mg_l"] == pytest.approx(cp["critical_do_mg_l"], abs=1e-10)


# --------------------------------------------------------------------------
# 规律 2：同位置口子合并 ≡ 负荷相加的单口
# --------------------------------------------------------------------------


def test_duplicate_positions_equal_summed_single_outfall(client):
    dup = multi_body(
        [{"x_km": 10.0, "l0": 10.0}, {"x_km": 10.0, "l0": 10.0}],
        x_max_km=150.0, n_points=301,
    )
    summed = multi_body([{"x_km": 10.0, "l0": 20.0}], x_max_km=150.0, n_points=301)
    rd = client.post("/multi/profile", json=dup)
    rs = client.post("/multi/profile", json=summed)
    assert rd.status_code == rs.status_code == 200
    d, s = rd.json(), rs.json()
    # 合并策略在回显中可见
    assert d["input"]["n_outfalls_submitted"] == 2
    assert d["input"]["n_outfalls"] == 1
    assert d["input"]["outfalls"][0]["n_merged"] == 2
    # 曲线与全部临界点逐位一致：叠加没有漏算或重复算
    assert d["profile"] == s["profile"]
    assert d["critical"] == s["critical"]


def test_validate_outfalls_canonicalizes_order_and_duplicates():
    a = validate_outfalls(
        [{"x_km": 5.0, "l0": 2.0}, {"x_km": 1.0, "l0": 3.0}, {"x_km": 5.0, "l0": 4.0}]
    )
    b = validate_outfalls(
        [{"x_km": 5.0, "l0": 4.0}, {"x_km": 5.0, "l0": 2.0}, {"x_km": 1.0, "l0": 3.0}]
    )
    assert a == b
    assert [o["x_km"] for o in a] == [1.0, 5.0]
    assert a[1]["l0"] == 6.0 and a[1]["n_merged"] == 2


# --------------------------------------------------------------------------
# 规律 3：因果性——下游新口子不改上游答案
# --------------------------------------------------------------------------


def test_downstream_outfall_does_not_change_upstream_results(client):
    base = multi_body([{"x_km": 5.0, "l0": 18.0}], x_max_km=300.0, n_points=601)
    extended = multi_body(
        [{"x_km": 5.0, "l0": 18.0}, {"x_km": 150.0, "l0": 25.0}],
        x_max_km=300.0, n_points=601,
    )
    b = client.post("/multi/profile", json=base).json()
    e = client.post("/multi/profile", json=extended).json()

    # 新口子（150 km）上游的曲线逐位不变
    b_up = [p for p in b["profile"] if p["x_km"] < 150.0]
    e_up = [p for p in e["profile"] if p["x_km"] < 150.0]
    assert len(b_up) > 100
    assert b_up == e_up

    # 上游已定位的临界点位置与数值逐位不变；新口子只在下游追加峰
    b_peaks = b["critical"]["peaks"]
    assert all(p["x_km"] < 150.0 for p in b_peaks)
    e_peaks_up = [p for p in e["critical"]["peaks"] if p["x_km"] < 150.0]
    keys = ("t_day", "x_km", "deficit_mg_l", "do_mg_l")
    assert [[p[k] for k in keys] for p in e_peaks_up] == [
        [p[k] for k in keys] for p in b_peaks
    ]
    assert e["critical"]["n_peaks"] > b["critical"]["n_peaks"]

    # 全局标注是全河性质：下游鼓出更深的包后，上游峰从全局降为次级，
    # 这是正确语义而非上游结果被改
    assert b_peaks[0]["is_global"] is True
    assert e_peaks_up[0]["is_global"] is False
    assert e["critical"]["global"]["x_c_km"] > 150.0


# --------------------------------------------------------------------------
# 规律 4：顺序不敏感
# --------------------------------------------------------------------------


def test_outfall_order_does_not_change_results(client):
    o1 = [
        {"x_km": 50.0, "l0": 12.0},
        {"x_km": 5.0, "l0": 18.0},
        {"x_km": 150.0, "l0": 25.0},
        {"x_km": 50.0, "l0": 3.0},
    ]
    o2 = [o1[2], o1[3], o1[0], o1[1]]  # 打乱，含一对重复位置
    r1 = client.post("/multi/profile", json=multi_body(o1, x_max_km=300.0, n_points=401))
    r2 = client.post("/multi/profile", json=multi_body(o2, x_max_km=300.0, n_points=401))
    assert r1.status_code == r2.status_code == 200
    # 规范化回显、整条曲线、全部临界点逐位一致
    assert r1.json() == r2.json()


# --------------------------------------------------------------------------
# 多峰识别：全部局部峰 + 全局最深标注 + 与独立密集扫描互验
# --------------------------------------------------------------------------


def test_two_distant_outfalls_produce_two_peaks_global_is_deepest(client):
    body = multi_body(
        [{"x_km": 5.0, "l0": 18.0}, {"x_km": 150.0, "l0": 25.0}],
        x_max_km=300.0, n_points=601,
    )
    d = client.post("/multi/profile", json=body).json()
    crit = d["critical"]
    assert crit["exists"] is True
    assert crit["n_peaks"] == 2

    p1, p2 = crit["peaks"]
    assert p1["x_km"] < p2["x_km"]  # 峰按河程升序汇报
    assert p1["is_global"] is False and p2["is_global"] is True
    assert p2["deficit_mg_l"] > p1["deficit_mg_l"]
    assert crit["global"]["x_c_km"] == p2["x_km"]
    assert crit["global"]["critical_deficit_mg_l"] == p2["deficit_mg_l"]

    ofs = make_outfalls((5.0, 18.0), (150.0, 25.0))
    # 每个峰都是局部极大：两侧邻近处亏氧严格更小
    for p in (p1, p2):
        for dt in (-0.02, 0.02):
            assert (
                total_deficit(p["t_day"] + dt, SHARED["d0"], ofs, 0.3, 0.6)
                < p["deficit_mg_l"]
            )

    # 与独立密集扫描互验全局峰（20001 点网格，不经过峰值搜索代码）
    horizon = crit["search_horizon_day"]
    grid = [i * horizon / 20000 for i in range(20001)]
    vals = [total_deficit(t, SHARED["d0"], ofs, 0.3, 0.6) for t in grid]
    i_max = max(range(len(vals)), key=lambda i: vals[i])
    assert crit["global"]["critical_deficit_mg_l"] >= vals[i_max] - 1e-9
    step_x = (horizon / 20000) * SHARED["u"]
    assert abs(crit["global"]["x_c_km"] - grid[i_max] * SHARED["u"]) < 2 * step_x


def test_first_peak_can_be_global_when_deeper():
    # 前口负荷大、后口负荷小：全局临界点是第一个峰，第二个作为次级峰汇报
    ofs = make_outfalls((5.0, 25.0), (150.0, 10.0))
    mc = find_all_peaks(SHARED["d0"], ofs, 0.3, 0.6, 30.0, 10.0)
    assert mc.exists and mc.n_peaks == 2
    first, second = mc.peaks
    assert first.is_global is True and second.is_global is False
    assert mc.global_peak.x_km == first.x_km
    assert first.deficit_mg_l > second.deficit_mg_l


def test_three_outfalls_need_not_mean_three_peaks():
    # 中间的口子落在第一个峰的上升段上，不单独鼓包：三个口子只出两个峰
    ofs = make_outfalls((5.0, 18.0), (60.0, 5.0), (150.0, 25.0))
    mc = find_all_peaks(SHARED["d0"], ofs, 0.3, 0.6, 30.0, 10.0)
    assert mc.exists and mc.n_peaks == 2
    xs = [p.x_km for p in mc.peaks]
    assert xs == sorted(xs)


# --------------------------------------------------------------------------
# 无临界点：如实报告，绝不凑数
# --------------------------------------------------------------------------


def test_no_rising_segment_reports_no_critical_point(client):
    body = multi_body(
        [{"x_km": 10.0, "l0": 0.5}, {"x_km": 40.0, "l0": 0.4}],
        d0=1.0, k1=0.2, k2=2.0,
    )
    r = client.post("/multi/profile", json=body)
    assert r.status_code == 200
    crit = r.json()["critical"]
    assert crit["exists"] is False
    assert crit["n_peaks"] == 0
    assert crit["peaks"] == []
    assert crit["global"] is None
    # 全程确实单调不增（复氧一直压得住）
    profile = r.json()["profile"]
    for pa, pb in zip(profile, profile[1:]):
        assert pb["deficit_mg_l"] <= pa["deficit_mg_l"] + 1e-12


def test_zero_loads_everywhere_reports_no_critical_point():
    ofs = make_outfalls((0.0, 0.0), (10.0, 0.0))
    mc = find_all_peaks(0.0, ofs, 0.3, 0.6, 30.0, 10.0)
    assert mc.exists is False
    assert mc.n_peaks == 0
    assert mc.global_peak is None


# --------------------------------------------------------------------------
# 叠加合成的物理正确性
# --------------------------------------------------------------------------


def test_total_deficit_is_background_plus_passed_outfalls_only():
    loads = ((10.0, 8.0), (30.0, 12.0), (60.0, 5.0))
    ofs = make_outfalls(*loads)
    k1, k2, u, d0 = 0.3, 0.6, 30.0, 2.0

    # 第一个口子上游：只有本底亏氧在衰减
    for x in (0.0, 3.0, 9.9):
        t = cf.time_from_distance(x, u)
        assert total_deficit(t, d0, ofs, k1, k2) == cf.deficit(t, d0, 0.0, k1, k2)

    # 任意点：本底残余 + 位置不晚于该点的口子贡献，手工逐项对账
    for x in (10.0, 25.0, 30.0, 45.0, 60.0, 100.0):
        t = cf.time_from_distance(x, u)
        expected = cf.deficit(t, d0, 0.0, k1, k2)
        for ox, ol in loads:
            ti = cf.time_from_distance(ox, u)
            if t >= ti:
                expected += cf.deficit(t - ti, 0.0, ol, k1, k2)
        assert total_deficit(t, d0, ofs, k1, k2) == expected


def test_single_outfall_downstream_with_zero_background_is_shifted_single():
    # d0 = 0 时，x0 处的单口 ≡ 单口曲线整体平移 x0
    k1, k2, u, csat = 0.3, 0.6, 30.0, 10.0
    ofs = make_outfalls((30.0, 20.0))
    mc = find_all_peaks(0.0, ofs, k1, k2, u, csat)
    cp = find_critical(0.0, 20.0, k1, k2, u, csat)
    assert mc.exists and mc.n_peaks == 1
    assert mc.global_peak.x_km == pytest.approx(30.0 + cp.x_c_km, abs=1e-8)
    assert mc.global_peak.deficit_mg_l == pytest.approx(cp.d_c, abs=1e-10)
    # 口子上游亏氧恒为 0（本底为 0、尚无口子经过）
    for p in scan_multi_profile(0.0, ofs, k1, k2, u, csat, 1.0, 11):
        assert p.deficit_mg_l == 0.0


def test_equal_rates_multi_degenerate_matches_special_branch():
    case = dict(d0=2.0, k1=0.4, k2=0.4, u=30.0, csat=10.0)
    ofs = prepare_outfalls(
        validate_outfalls([{"x_km": 0.0, "l0": 20.0}]), case["u"]
    )
    mc = find_all_peaks(case["d0"], ofs, case["k1"], case["k2"], case["u"], case["csat"])
    cp = find_critical(case["d0"], 20.0, case["k1"], case["k2"], case["u"], case["csat"])
    assert cp.branch == "equal_rates"
    assert mc.exists and mc.n_peaks == 1
    assert mc.global_peak.t_day == pytest.approx(cp.t_c_day, abs=1e-9)
    assert mc.global_peak.deficit_mg_l == pytest.approx(cp.d_c, abs=1e-10)


def test_search_horizon_covers_all_peaks_and_tail_decays():
    ofs = make_outfalls((5.0, 18.0), (150.0, 25.0))
    mc = find_all_peaks(SHARED["d0"], ofs, 0.3, 0.6, 30.0, 10.0)
    # 所有峰都在搜索上界之内，且上界处曲线已在衰减
    assert all(p.t_day < mc.t_end_day for p in mc.peaks)
    assert total_deficit_rate(mc.t_end_day, SHARED["d0"], ofs, 0.3, 0.6) < 0.0


# --------------------------------------------------------------------------
# 窗口与扫描行为
# --------------------------------------------------------------------------


def test_api_window_by_distance_default_and_conflict(client):
    r = client.post(
        "/multi/profile",
        json=multi_body([{"x_km": 5.0, "l0": 18.0}], x_max_km=120.0, n_points=241),
    )
    d = r.json()
    assert r.status_code == 200
    assert d["window"]["t_max_day"] == pytest.approx(4.0)  # 120 km / 30 km/day
    assert len(d["profile"]) == 241
    for pt in d["profile"]:
        assert pt["x_km"] == pytest.approx(pt["t_day"] * 30.0)
        assert pt["do_mg_l"] == pytest.approx(10.0 - pt["deficit_mg_l"])

    # 缺省窗口 = 临界点搜索上界（覆盖全部峰与衰减尾部）
    d2 = client.post(
        "/multi/profile", json=multi_body([{"x_km": 5.0, "l0": 18.0}])
    ).json()
    assert d2["window"]["t_max_day"] == pytest.approx(
        d2["critical"]["search_horizon_day"]
    )

    # 两种窗口单位同时给 → 400
    r3 = client.post(
        "/multi/profile",
        json=multi_body([{"x_km": 5.0, "l0": 18.0}], t_max_day=4.0, x_max_km=120.0),
    )
    assert r3.status_code == 400


# --------------------------------------------------------------------------
# 非法输入：计算前挡回，错误信息定位到具体口子
# --------------------------------------------------------------------------


def test_validate_outfalls_rejects_bad_inputs():
    with pytest.raises(ParameterError, match="不能为空"):
        validate_outfalls([])
    with pytest.raises(ParameterError, match="第 1 个排污口.*不能为负"):
        validate_outfalls([{"x_km": 1.0, "l0": 5.0}, {"x_km": -2.0, "l0": 5.0}])
    with pytest.raises(ParameterError, match="第 0 个排污口.*不能为负"):
        validate_outfalls([{"x_km": 1.0, "l0": -0.5}])
    with pytest.raises(ParameterError, match="有限数值"):
        validate_outfalls([{"x_km": float("nan"), "l0": 5.0}])
    with pytest.raises(ParameterError, match="有限数值"):
        validate_outfalls([{"x_km": 1.0, "l0": float("inf")}])
    with pytest.raises(ParameterError, match="必须是数值"):
        validate_outfalls([{"x_km": True, "l0": 5.0}])
    with pytest.raises(ParameterError, match="上限"):
        validate_outfalls(
            [{"x_km": float(i), "l0": 1.0} for i in range(MAX_OUTFALLS + 1)]
        )
    with pytest.raises(ParameterError, match="必须同时给出"):
        validate_outfalls([{"x_km": 1.0}])
    with pytest.raises(ParameterError, match="必须是数组"):
        validate_outfalls("not-a-list")
    # 恰好上限数量合法（且同位置全部合并）
    merged = validate_outfalls([{"x_km": 1.0, "l0": 1.0}] * MAX_OUTFALLS)
    assert merged == [{"x_km": 1.0, "l0": float(MAX_OUTFALLS), "n_merged": MAX_OUTFALLS}]


def test_api_empty_outfall_list_rejected(client):
    r = client.post("/multi/profile", json=multi_body([]))
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_parameter"
    assert "不能为空" in body["detail"]


@pytest.mark.parametrize(
    "outfalls, fragment",
    [
        ([{"x_km": 1.0, "l0": 5.0}, {"x_km": -2.0, "l0": 5.0}], "第 1 个排污口"),
        ([{"x_km": 1.0, "l0": -0.5}], "第 0 个排污口"),
        ([{"x_km": 3.0, "l0": 1.0}, {"x_km": 2.0, "l0": 1.0}, {"x_km": -1.0, "l0": 1.0}], "第 2 个排污口"),
    ],
)
def test_api_outfall_value_errors_carry_index(client, outfalls, fragment):
    r = client.post("/multi/profile", json=multi_body(outfalls))
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_parameter"
    assert fragment in r.json()["detail"]


def test_api_too_many_outfalls_rejected(client):
    body = multi_body([{"x_km": float(i % 50), "l0": 1.0} for i in range(501)])
    r = client.post("/multi/profile", json=body)
    assert r.status_code == 400
    assert "上限" in r.json()["detail"]


def test_api_nan_and_inf_outfall_fields_rejected(client):
    raw = json.dumps(multi_body([{"x_km": float("nan"), "l0": 5.0}]))
    r = client.post(
        "/multi/profile", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code in (400, 422)

    raw = json.dumps(multi_body([{"x_km": 1.0, "l0": float("inf")}]))
    r = client.post(
        "/multi/profile", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code in (400, 422)


def test_api_outfall_structural_errors_return_422(client):
    # 缺字段
    r = client.post("/multi/profile", json=multi_body([{"x_km": 1.0}]))
    assert r.status_code == 422
    # 额外字段
    r = client.post("/multi/profile", json=multi_body([{"x_km": 1.0, "l0": 2.0, "foo": 1}]))
    assert r.status_code == 422
    # 布尔冒充数值
    r = client.post("/multi/profile", json=multi_body([{"x_km": True, "l0": 2.0}]))
    assert r.status_code == 422
    # 列表本身不是数组
    r = client.post("/multi/profile", json=multi_body({"x_km": 1.0, "l0": 2.0}))
    assert r.status_code == 422


def test_api_shared_params_still_validated(client):
    # d0 超过 csat
    r = client.post("/multi/profile", json=multi_body([{"x_km": 1.0, "l0": 5.0}], d0=11.0))
    assert r.status_code == 400
    # 负复氧系数
    r = client.post("/multi/profile", json=multi_body([{"x_km": 1.0, "l0": 5.0}], k2=-0.5))
    assert r.status_code == 400
    # 零流速
    r = client.post("/multi/profile", json=multi_body([{"x_km": 1.0, "l0": 5.0}], u=0.0))
    assert r.status_code == 400
    # 顶层多余字段
    r = client.post(
        "/multi/profile", json=multi_body([{"x_km": 1.0, "l0": 5.0}], l0=99.0)
    )
    assert r.status_code == 422
