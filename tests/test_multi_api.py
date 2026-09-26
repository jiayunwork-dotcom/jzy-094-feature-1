"""多排污口叠加接口 POST /multi/critical 的端到端测试。

重点锁：
- 单口退化时与原 /critical 接口给出的临界点数值一致；
- 双峰工况明确汇报两个峰并标注 global/secondary；
- 提交顺序打乱，整条曲线与所有临界点逐位不变；
- 同位排口合并、下游插入口不改上游答案（接口层同样成立）；
- 无临界点如实 exists=false，不凑正数；
- 各类非法输入（空列表、负河程、非有限、负荷为负、越界数量等）在计算
  前被挡回，错误信息能定位到具体口子，沿用既有错误响应风格。
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


COMMON = dict(d0=2.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)


def _body(outfalls, **overrides):
    body = dict(COMMON)
    body.update(overrides)
    body["outfalls"] = [{"x_km": x, "l0": l} for x, l in outfalls]
    return body


def test_health_still_ok(client):
    assert client.get("/healthz").status_code == 200


def test_single_outfall_matches_single_source_endpoint(client):
    body = _body([(0.0, 20.0)])
    r_multi = client.post("/multi/critical", json=body)
    r_single = client.post("/critical", json=dict(COMMON, l0=20.0))
    assert r_multi.status_code == 200
    assert r_single.status_code == 200

    mc = r_multi.json()["critical"]
    sc = r_single.json()["critical"]
    assert mc["exists"] is True
    assert mc["n_local_maxima"] == 1
    assert mc["global"]["is_global"] is True
    assert mc["global"]["kind"] == "global"
    assert mc["global"]["t_c_day"] == pytest.approx(sc["t_c_day"], abs=1e-6)
    assert mc["global"]["x_c_km"] == pytest.approx(sc["x_c_km"], abs=3e-5)
    assert mc["global"]["critical_deficit_mg_l"] == pytest.approx(
        sc["critical_deficit_mg_l"], abs=1e-12
    )
    # 唯一的峰同时出现在 peaks 列表与 global 字段
    assert len(mc["peaks"]) == 1
    assert mc["peaks"][0]["t_c_day"] == mc["global"]["t_c_day"]


def test_single_outfall_profile_matches_single_source_profile(client):
    r_multi = client.post(
        "/multi/critical", json=_body([(0.0, 20.0)], x_max_km=120.0, n_points=401)
    )
    r_single = client.post(
        "/profile", json=dict(COMMON, l0=20.0, x_max_km=120.0, n_points=401)
    )
    assert r_multi.status_code == 200
    mp = r_multi.json()["profile"]
    sp = r_single.json()["profile"]
    assert len(mp) == len(sp) == 401
    for a, b in zip(mp, sp):
        assert a["t_day"] == pytest.approx(b["t_day"])
        assert a["x_km"] == pytest.approx(b["x_km"])
        assert a["deficit_mg_l"] == pytest.approx(b["deficit_mg_l"], abs=1e-12)
        assert a["do_mg_l"] == pytest.approx(b["do_mg_l"], abs=1e-12)
        assert a["bod_mg_l"] == pytest.approx(b["bod_mg_l"], abs=1e-12)


def test_two_peaks_are_both_reported_with_global_marked(client):
    body = _body([(0.0, 18.0), (80.0, 18.0)], k1=0.35)
    r = client.post("/multi/critical", json=body)
    assert r.status_code == 200
    crit = r.json()["critical"]
    assert crit["exists"] is True
    assert crit["n_local_maxima"] == 2
    peaks = crit["peaks"]
    assert sorted(p["x_c_km"] for p in peaks) == [p["x_c_km"] for p in peaks]
    global_peaks = [p for p in peaks if p["is_global"]]
    assert len(global_peaks) == 1
    assert crit["global"]["x_c_km"] == global_peaks[0]["x_c_km"]
    assert crit["global"]["critical_deficit_mg_l"] == max(
        p["critical_deficit_mg_l"] for p in peaks
    )
    assert {p["kind"] for p in peaks} == {"global", "secondary"}
    # 第一峰在第一口与第二口之间，第二峰在第二口下游
    assert peaks[0]["x_c_km"] < 80.0 < peaks[1]["x_c_km"]
    # 自动窗口包住两个峰且终点已下降
    assert crit["still_rising_at_end"] is False
    assert peaks[-1]["x_c_km"] < crit["x_scan_max_km"]


def test_shuffled_order_gives_identical_response(client):
    outfalls = [(0.0, 10.0), (50.0, 15.0), (20.0, 8.0), (90.0, 6.0), (50.0, 5.0)]
    order_a = outfalls
    order_b = [outfalls[i] for i in (3, 0, 4, 2, 1)]
    ra = client.post(
        "/multi/critical", json=_body(order_a, x_max_km=300.0, n_points=833)
    )
    rb = client.post(
        "/multi/critical", json=_body(order_b, x_max_km=300.0, n_points=833)
    )
    assert ra.status_code == rb.status_code == 200
    ja, jb = ra.json(), rb.json()
    assert ja["profile"] == jb["profile"]
    assert ja["critical"] == jb["critical"]
    assert ja["merged_outfalls"] == jb["merged_outfalls"]


def test_coincident_outfalls_merge_strategy_reported(client):
    r = client.post("/multi/critical", json=_body([(30.0, 10.0), (30.0, 10.0)]))
    assert r.status_code == 200
    data = r.json()
    merged = data["merged_outfalls"]
    assert merged["n_submitted"] == 2
    assert merged["n_merged"] == 1
    assert merged["n_coincident_merges"] == 1
    assert merged["outfalls"] == [{"x_km": 30.0, "l0": 20.0}]

    # 与负荷翻倍的单口请求结果一致（临界曲线）
    r2 = client.post("/multi/critical", json=_body([(30.0, 20.0)]))
    assert r2.json()["critical"] == data["critical"]


def test_adding_downstream_outfall_keeps_upstream_answer(client):
    upstream = _body([(0.0, 15.0), (40.0, 12.0)], x_max_km=100.0, n_points=501)
    extended = _body(
        [(0.0, 15.0), (40.0, 12.0), (200.0, 25.0)], x_max_km=100.0, n_points=501
    )
    a = client.post("/multi/critical", json=upstream).json()
    b = client.post("/multi/critical", json=extended).json()
    assert a["profile"] == b["profile"]
    assert a["critical"] == b["critical"]


def test_no_critical_point_reported_honestly(client):
    body = _body([(10.0, 0.5), (60.0, 0.3)], d0=1.0)
    r = client.post("/multi/critical", json=body)
    assert r.status_code == 200
    crit = r.json()["critical"]
    assert crit["exists"] is False
    assert crit["peaks"] == []
    assert crit["global"] is None
    assert crit["n_local_maxima"] == 0
    # 曲线仍然正常返回，且全程亏氧不超过初始本底
    for pt in r.json()["profile"]:
        assert pt["deficit_mg_l"] <= 1.0 + 1e-9


def test_distance_window_is_converted_via_x_over_u(client):
    r = client.post(
        "/multi/critical", json=_body([(0.0, 10.0)], x_max_km=90.0)
    )
    assert r.status_code == 200
    window = r.json()["window"]
    assert window["t_max_day"] == pytest.approx(3.0)
    assert window["x_max_km"] == pytest.approx(90.0)
    assert window["auto"] is False


def test_both_window_units_rejected(client):
    body = _body([(0.0, 10.0)], t_max_day=3.0, x_max_km=90.0)
    r = client.post("/multi/critical", json=body)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_parameter"


# --------------------------------------------------------------------------
# 非法输入
# --------------------------------------------------------------------------


def test_empty_outfall_list_returns_400(client):
    r = client.post("/multi/critical", json=dict(COMMON, outfalls=[]))
    assert r.status_code == 400
    err = r.json()
    assert err["error"] == "invalid_parameter"
    assert "不能为空" in err["detail"]


@pytest.mark.parametrize(
    "outfalls,index",
    [
        ([(0.0, 5.0), (-1.0, 5.0)], 1),       # 负河程
        ([(0.0, -2.0)], 0),                    # 负负荷
        ([(0.0, 5.0), (10.0, 5.0), (-3.0, 1.0)], 2),
    ],
)
def test_invalid_outfall_values_return_400_with_index(client, outfalls, index):
    r = client.post("/multi/critical", json=_body(outfalls))
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert f"排污口[{index}]" in detail


def test_non_numeric_field_returns_422_or_400(client):
    body = dict(COMMON, outfalls=[{"x_km": "abc", "l0": 5.0}])
    r = client.post("/multi/critical", json=body)
    assert r.status_code in (400, 422)


def test_bool_field_rejected(client):
    body = dict(COMMON, outfalls=[{"x_km": True, "l0": 5.0}])
    r = client.post("/multi/critical", json=body)
    assert r.status_code in (400, 422)


def test_nan_position_rejected(client):
    raw = json.dumps(
        dict(COMMON, outfalls=[{"x_km": float("nan"), "l0": 5.0}])
    )
    r = client.post(
        "/multi/critical", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code in (400, 422)


def test_inf_load_rejected(client):
    raw = json.dumps(
        dict(COMMON, outfalls=[{"x_km": 0.0, "l0": float("inf")}])
    )
    r = client.post(
        "/multi/critical", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code in (400, 422)


def test_missing_outfall_field_returns_422(client):
    body = dict(COMMON, outfalls=[{"x_km": 1.0}])
    r = client.post("/multi/critical", json=body)
    assert r.status_code == 422


def test_missing_outfalls_key_returns_422(client):
    r = client.post("/multi/critical", json=dict(COMMON))
    assert r.status_code == 422


def test_extra_field_rejected(client):
    body = _body([(0.0, 5.0)])
    body["outfalls"][0]["unexpected"] = 1
    r = client.post("/multi/critical", json=body)
    assert r.status_code == 422


def test_shared_bad_coefficient_returns_400(client):
    r = client.post("/multi/critical", json=_body([(0.0, 5.0)], k2=-0.6))
    assert r.status_code == 400
    r = client.post("/multi/critical", json=_body([(0.0, 5.0)], u=0.0))
    assert r.status_code == 400
    r = client.post("/multi/critical", json=_body([(0.0, 5.0)], d0=11.0))
    assert r.status_code == 400


def test_absurd_outfall_count_rejected(client):
    # 不真的构造上万个元素：直接调内核校验路径锁定上限与错误信息
    from app.validation import MAX_OUTFALLS, ParameterError, validate_multi_inputs

    with pytest.raises(ParameterError, match=str(MAX_OUTFALLS + 1)):
        validate_multi_inputs(
            d0=2.0,
            k1=0.3,
            k2=0.6,
            u=30.0,
            csat=10.0,
            outfalls=[(float(i), 1.0) for i in range(MAX_OUTFALLS + 1)],
        )
