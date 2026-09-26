"""HTTP 接口层。只做参数接收/校验与计算模块编排，不含数值实现。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import closed_form as cf
from .critical import find_critical, find_critical_numeric
from .multi_critical import find_critical_multi
from .multi_source import build_river, scan_multi_profile
from .reference import REFERENCE_INPUT, REFERENCE_N_POINTS, REFERENCE_X_MAX_KM
from .scanning import analytic_within_window, refine_window_max, scan_profile
from .sweep import SweepParams, sweep
from .validation import (
    ParameterError,
    validate_model_inputs,
    validate_multi_inputs,
    validate_sweep_bounds,
    validate_window,
)
from .schemas import ModelInput, MultiOutfallRequest, ProfileRequest, SweepRequest

logger = logging.getLogger("sp_do_sag")


def _normalized(model: ModelInput) -> dict:
    # Pydantic 保证结构与基本类型；取值域规则（含 bool/NaN 已在 schema 拦截）
    # 与纯内核模块共用同一套校验。
    return validate_model_inputs(
        model.d0, model.l0, model.k1, model.k2, model.u, model.csat
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即扫参考工况，若内核本身坏了会在日志里立刻暴露
    cp = find_critical(**REFERENCE_INPUT)
    logger.info(
        "参考工况临界点: exists=%s t_c=%s day x_c=%s km D_c=%s mg/L",
        cp.exists,
        cp.t_c_day,
        cp.x_c_km,
        cp.d_c,
    )
    yield


app = FastAPI(
    title="Streeter-Phelps 氧垂分析计算服务",
    version="1.0.0",
    description=(
        "沿排污口下游扫描亏氧曲线、定位溶解氧最低点（临界点），"
        "支持 k1/k2/L0/D0/U 区间批量扫参。时间单位天、河程单位公里、"
        "流速 km/day，t = x/U。"
    ),
    lifespan=lifespan,
)


@app.exception_handler(ParameterError)
async def parameter_error_handler(request: Request, exc: ParameterError):
    return JSONResponse(status_code=400, content={"error": "invalid_parameter", "detail": str(exc)})


def _json_safe(value):
    """把 NaN/Inf、异常对象等不可直接 JSON 序列化的内容净化成字符串，
    保证非法输入（如 NaN 系数）始终以 4xx 返回，而不是错误体自身序列化崩溃。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else "non_finite_float"
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "request_validation_error", "detail": _json_safe(exc.errors())},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "streeter-phelps-do-sag"}


@app.get("/reference")
def reference():
    cp = find_critical(**REFERENCE_INPUT)
    return {
        "description": "排污口下游参考工况：应出现明显氧垂，临界距离为正",
        "input": REFERENCE_INPUT,
        "default_scan": {
            "x_max_km": REFERENCE_X_MAX_KM,
            "n_points": REFERENCE_N_POINTS,
        },
        "critical": cp.as_dict(),
    }


@app.post("/critical")
def critical_endpoint(model: ModelInput):
    p = _normalized(model)
    cp = find_critical(**p)
    numeric_check = find_critical_numeric(**p) if cp.exists else None
    return {"input": p, "critical": cp.as_dict(), "numeric_check": numeric_check}


@app.post("/profile")
def profile_endpoint(model: ProfileRequest):
    p = _normalized(model)
    t_max, x_max = validate_window(model.t_max_day, model.x_max_km)

    # 窗口以河程给出时，务必先折回时间：t = x/U
    if t_max is None:
        if x_max is None:
            t_max = 3.0 / p["k1"]  # 默认扫约三个耗氧时标
        else:
            t_max = cf.time_from_distance(x_max, p["u"])

    points = scan_profile(
        p["d0"], p["l0"], p["k1"], p["k2"], p["u"], p["csat"],
        t_max, model.n_points,
    )
    cp = find_critical(**p)
    window_max = refine_window_max(
        p["d0"], p["l0"], p["k1"], p["k2"], p["u"], p["csat"],
        t_max, model.n_points,
    )

    return {
        "input": p,
        "window": {
            "t_max_day": t_max,
            "x_max_km": cf.distance_from_time(t_max, p["u"]),
            "n_points": model.n_points,
        },
        "critical": cp.as_dict(),
        "critical_within_window": analytic_within_window(cp, t_max),
        "window_max_deficit": window_max,
        "profile": [pt.as_dict() for pt in points],
    }


@app.post("/sweep")
def sweep_endpoint(model: SweepRequest):
    p = _normalized(model)
    lo, hi, n_points = validate_sweep_bounds(
        model.parameter, model.lo, model.hi, model.n_points, csat=p["csat"]
    )
    base = SweepParams(**p)
    result = sweep(base, model.parameter, lo, hi, n_points)
    return {"input": p, "sweep": result}


@app.post("/multi/critical")
def multi_critical_endpoint(model: MultiOutfallRequest):
    """多排污口叠加：合成总亏氧曲线，数值搜索全部局部亏氧极大。"""
    raw_outfalls = [o.model_dump() for o in model.outfalls]
    p = validate_multi_inputs(
        model.d0, model.k1, model.k2, model.u, model.csat, raw_outfalls
    )
    t_max, x_max = validate_window(model.t_max_day, model.x_max_km)
    if t_max is None and x_max is not None:
        # 窗口以河程给出时先折回时间：t = x/U
        t_max = cf.time_from_distance(x_max, p["u"])

    river = build_river(
        p["d0"],
        p["k1"],
        p["k2"],
        p["u"],
        p["csat"],
        p["outfalls"],
        n_submitted=p["n_submitted"],
        n_coincident_merges=p["n_coincident_pairs"],
    )
    result = find_critical_multi(river, t_max)
    points = scan_multi_profile(river, result.t_scan_max_day, model.n_points)

    return {
        "input": {
            "d0": p["d0"],
            "k1": p["k1"],
            "k2": p["k2"],
            "u": p["u"],
            "csat": p["csat"],
            "outfalls": [
                {"x_km": o.x_km, "l0": o.l0} for o in model.outfalls
            ],
        },
        "merged_outfalls": {
            "strategy": "位置完全相同的排口按同位负荷相加合并为一个等效排口",
            "n_submitted": river.n_submitted,
            "n_merged": len(river.outfalls),
            "n_coincident_merges": river.n_coincident_merges,
            "outfalls": [
                {"x_km": o.x_km, "l0": o.l0} for o in river.outfalls
            ],
        },
        "window": {
            "t_max_day": result.t_scan_max_day,
            "x_max_km": result.x_scan_max_km,
            "n_points": model.n_points,
            "auto": t_max is None,
        },
        "critical": result.as_dict(),
        "profile": [pt.as_dict() for pt in points],
    }
