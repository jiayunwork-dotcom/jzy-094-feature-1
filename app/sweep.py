"""区间批量扫参：沿某个系数（k1/k2/L0/D0/U）的区间等间距取样，
逐点定位解析临界点，汇报临界亏氧随该系数的变化趋势。

趋势按相邻样本临界亏氧的符号分类：
- increasing 全程不降（且至少一处严格上升）
- decreasing 全程不增（且至少一处严格下降）
- mixed      有升有降
- flat       全部持平（或全部无临界点）
无临界点的样本以 exists=false、null 字段明确给出，不参与升降比较。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from . import closed_form as cf
from .critical import find_critical
from .numeric import linspace

SweepParameter = Literal["k1", "k2", "l0", "d0", "u"]

SWEEP_PARAMETERS: tuple[str, ...] = ("k1", "k2", "l0", "d0", "u")

# 相邻临界亏氧视为变化的相对阈值，避免末位浮点抖动污染趋势分类
TREND_REL_TOL = 1e-9


@dataclass(frozen=True)
class SweepParams:
    d0: float
    l0: float
    k1: float
    k2: float
    u: float
    csat: float


def sweep(
    base: SweepParams,
    parameter: SweepParameter,
    lo: float,
    hi: float,
    n_points: int,
) -> dict:
    if parameter not in SWEEP_PARAMETERS:
        raise ValueError(f"不支持的扫参系数: {parameter!r}")

    values = linspace(lo, hi, n_points)
    rows: list[dict] = []
    valid_d: list[float] = []

    for value in values:
        params = replace(base, **{parameter: value})
        cp = find_critical(
            params.d0, params.l0, params.k1, params.k2, params.u, params.csat
        )
        row = {
            "value": value,
            "exists": cp.exists,
            "branch": cp.branch,
            "t_c_day": cp.t_c_day,
            "x_c_km": cp.x_c_km,
            "critical_deficit_mg_l": cp.d_c,
            "critical_do_mg_l": cp.do_c,
        }
        rows.append(row)
        if cp.d_c is not None:
            valid_d.append(cp.d_c)

    return {
        "parameter": parameter,
        "lo": lo,
        "hi": hi,
        "n_points": n_points,
        "points": rows,
        "trend": _classify_trend(rows),
        "n_critical_found": len(valid_d),
    }


def _classify_trend(rows: list[dict]) -> str:
    deficits = [
        row["critical_deficit_mg_l"] for row in rows if row["exists"]
    ]
    if len(deficits) < 2:
        return "flat"

    went_up = False
    went_down = False
    for prev, cur in zip(deficits, deficits[1:]):
        scale = max(1.0, abs(prev), abs(cur))
        delta = cur - prev
        if delta > TREND_REL_TOL * scale:
            went_up = True
        elif delta < -TREND_REL_TOL * scale:
            went_down = True

    if went_up and went_down:
        return "mixed"
    if went_up:
        return "increasing"
    if went_down:
        return "decreasing"
    return "flat"
