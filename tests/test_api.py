"""HTTP 接口层测试：健康检查、参考工况、三个计算端点的编排与 400 校验。"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


REF_BODY = dict(d0=2.0, l0=20.0, k1=0.3, k2=0.6, u=30.0, csat=10.0)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_reference_shows_positive_critical_distance(client):
    r = client.get("/reference")
    assert r.status_code == 200
    body = r.json()
    cp = body["critical"]
    assert cp["exists"] is True
    assert cp["x_c_km"] > 0
    assert cp["x_c_km"] == pytest.approx(58.8, abs=1.0)


def test_critical_endpoint_reference_case(client):
    r = client.post("/critical", json=REF_BODY)
    assert r.status_code == 200
    body = r.json()
    cp = body["critical"]
    assert cp["exists"] is True
    assert cp["branch"] == "general"
    assert cp["t_c_day"] == pytest.approx(1.959, abs=0.01)
    assert cp["x_c_km"] == pytest.approx(58.8, abs=1.0)
    # 数值核验与解析解一致
    nc = body["numeric_check"]
    assert nc["exists"] is True
    assert nc["t_c_day"] == pytest.approx(cp["t_c_day"], abs=1e-6)


def test_critical_endpoint_equal_rates_special_branch(client):
    body = dict(REF_BODY, k1=0.4, k2=0.4)
    r = client.post("/critical", json=body)
    assert r.status_code == 200
    cp = r.json()["critical"]
    assert cp["exists"] is True
    assert cp["branch"] == "equal_rates"
    assert cp["x_c_km"] > 0


def test_critical_endpoint_no_critical_point(client):
    body = dict(REF_BODY, d0=3.0, l0=5.0, k1=0.2, k2=0.9)
    r = client.post("/critical", json=body)
    assert r.status_code == 200
    cp = r.json()["critical"]
    assert cp["exists"] is False
    assert cp["x_c_km"] is None


def test_profile_endpoint_window_by_distance_converts_to_time(client):
    body = dict(REF_BODY, x_max_km=120.0, n_points=401)
    r = client.post("/profile", json=body)
    assert r.status_code == 200
    data = r.json()
    # 120 km / 30 km/day = 4 day
    assert data["window"]["t_max_day"] == pytest.approx(4.0)
    assert data["window"]["x_max_km"] == pytest.approx(120.0)
    assert len(data["profile"]) == 401
    assert data["critical_within_window"] is True
    # 每行时间与河程满足 x = U t
    for pt in data["profile"]:
        assert pt["x_km"] == pytest.approx(pt["t_day"] * 30.0)
    wm = data["window_max_deficit"]
    assert wm["t_day"] == pytest.approx(data["critical"]["t_c_day"], abs=1e-6)


def test_profile_endpoint_rejects_both_window_units(client):
    body = dict(REF_BODY, t_max_day=4.0, x_max_km=120.0)
    r = client.post("/profile", json=body)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize(
    "override",
    [
        {"k2": -0.6},       # 负复氧系数
        {"u": 0.0},         # 零流速
        {"csat": "abc"},    # 非数值
        {"d0": 11.0},       # D0 超过 Csat
        {"k1": True},       # bool
    ],
)
def test_invalid_inputs_return_400_or_422(client, override):
    body = dict(REF_BODY, **override)
    r = client.post("/critical", json=body)
    assert r.status_code in (400, 422)


def test_nan_coefficient_rejected(client):
    # httpx 客户端本身禁止 NaN/Inf 序列化，需手工构造非标准 JSON 报文；
    # 服务端（pydantic 前置校验 + validation 层）必须拒绝
    raw = json.dumps(dict(REF_BODY, l0=float("nan")))
    r = client.post("/critical", content=raw, headers={"content-type": "application/json"})
    assert r.status_code in (400, 422)


def test_inf_coefficient_rejected(client):
    raw = json.dumps(dict(REF_BODY, k2=float("inf")))
    r = client.post("/critical", content=raw, headers={"content-type": "application/json"})
    assert r.status_code in (400, 422)


def test_sweep_endpoint_k2_decreasing_trend(client):
    body = dict(REF_BODY, parameter="k2", lo=0.45, hi=1.2, n_points=6)
    r = client.post("/sweep", json=body)
    assert r.status_code == 200
    data = r.json()["sweep"]
    assert data["trend"] == "decreasing"
    assert len(data["points"]) == 6


def test_sweep_endpoint_validates_bounds(client):
    body = dict(REF_BODY, parameter="k2", lo=1.2, hi=0.45, n_points=6)
    r = client.post("/sweep", json=body)
    assert r.status_code == 400


def test_unknown_field_rejected(client):
    body = dict(REF_BODY, extra_field=1)
    r = client.post("/critical", json=body)
    assert r.status_code == 422
