"""输入校验。任何非法系数（非数值、NaN/Inf、非正、越界等）都抛
ParameterError，由接口层统一转成 HTTP 400。
"""

from __future__ import annotations

import math
from typing import Any


class ParameterError(ValueError):
    """输入参数非法。message 可直接回给调用方。"""


def _check_finite(value: Any, name: str) -> float:
    # bool 是 int 的子类，必须先挡掉，避免 True 被当成 1.0 静默接受
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数值，收到 {value!r}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ParameterError(f"{name} 必须是有限数值，不能为 NaN 或无穷，收到 {value!r}")
    return fv


def require_positive(value: Any, name: str) -> float:
    fv = _check_finite(value, name)
    if fv <= 0.0:
        raise ParameterError(f"{name} 必须严格大于 0，收到 {value!r}")
    return fv


def require_non_negative(value: Any, name: str) -> float:
    fv = _check_finite(value, name)
    if fv < 0.0:
        raise ParameterError(f"{name} 不能为负，收到 {value!r}")
    return fv


def require_int(value: Any, name: str, *, minimum: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}")
    if value < minimum:
        raise ParameterError(f"{name} 必须 >= {minimum}，收到 {value!r}")
    return value


def validate_shared_inputs(
    d0: Any,
    k1: Any,
    k2: Any,
    u: Any,
    csat: Any,
) -> dict:
    """校验单口与多口工况共用的五个输入（除负荷外的模型参数）。

    k1、k2、U、Csat 必须严格为正；D0（多口工况下为来水本底亏氧）必须落在
    [0, Csat] 内。
    """
    k1v = require_positive(k1, "k1（耗氧系数 /day）")
    k2v = require_positive(k2, "k2（复氧系数 /day）")
    uv = require_positive(u, "u（流速 km/day，1 m/s = 86.4 km/day）")
    csatv = require_positive(csat, "csat（饱和溶解氧 mg/L）")
    d0v = require_non_negative(d0, "d0（初始/本底亏氧 mg/L）")
    if d0v > csatv:
        raise ParameterError(
            f"d0（初始/本底亏氧 {d0v}）不能超过 csat（饱和溶解氧 {csatv}）"
        )
    return {
        "d0": d0v,
        "k1": k1v,
        "k2": k2v,
        "u": uv,
        "csat": csatv,
    }


def validate_model_inputs(
    d0: Any,
    l0: Any,
    k1: Any,
    k2: Any,
    u: Any,
    csat: Any,
) -> dict:
    """校验单一工况的六个模型输入，返回规范化后的浮点字典。

    k1、k2、U、Csat 必须严格为正；L0 允许为 0（零负荷，此时没有氧垂）；
    D0 必须落在 [0, Csat] 内（亏氧不可能超过饱和值）。
    """
    shared = validate_shared_inputs(d0, k1, k2, u, csat)
    l0v = require_non_negative(l0, "l0（初始碳质 BOD mg/L）")
    return {
        "d0": shared["d0"],
        "l0": l0v,
        "k1": shared["k1"],
        "k2": shared["k2"],
        "u": shared["u"],
        "csat": shared["csat"],
    }


# 排污口数量的合理上限。真实河段口子以个位/十位数计，超过该上限明显是
# 调用方构造错误，直接在计算前挡回。
MAX_OUTFALLS = 500


def validate_outfalls(outfalls: Any, *, max_outfalls: int = MAX_OUTFALLS) -> list[dict]:
    """校验并规范化排污口列表，返回按河程升序、同位置已合并的规范列表。

    每个口子形如 {"x_km": ..., "l0": ...}：位置必须是非负有限数，负荷必须
    是非负有限数（允许 0，即零负荷口子，不产生贡献）。错误信息携带口子
    序号（按提交顺序从 0 起），便于调用方定位。

    完全相同位置的多个口子按「负荷相加合并为一口」的显式策略处理——叠加
    方程对负荷是线性的，同位置两口的联合贡献与一口承担负荷之和严格相等。
    合并时负荷先按升序再求和，保证提交顺序不影响逐位结果。
    """
    if isinstance(outfalls, bool) or not isinstance(outfalls, (list, tuple)):
        raise ParameterError(f"outfalls（排污口列表）必须是数组，收到 {outfalls!r}")
    if len(outfalls) == 0:
        raise ParameterError("outfalls（排污口列表）不能为空，至少给出一个排污口")
    if len(outfalls) > max_outfalls:
        raise ParameterError(
            f"排污口数量 {len(outfalls)} 超出合理上限 {max_outfalls}，"
            "请先合并邻近口子或分段计算"
        )

    loads_by_position: dict[float, list[float]] = {}
    for i, item in enumerate(outfalls):
        if isinstance(item, dict):
            if "x_km" not in item or "l0" not in item:
                raise ParameterError(
                    f"第 {i} 个排污口必须同时给出 x_km（河程位置 km）与 l0（BOD 负荷 mg/L）"
                )
            x_raw, l0_raw = item["x_km"], item["l0"]
        else:  # 兼容带有同名属性的请求对象
            x_raw = getattr(item, "x_km", None)
            l0_raw = getattr(item, "l0", None)
            if x_raw is None or l0_raw is None:
                raise ParameterError(
                    f"第 {i} 个排污口必须同时给出 x_km（河程位置 km）与 l0（BOD 负荷 mg/L）"
                )
        xv = require_non_negative(x_raw, f"第 {i} 个排污口的 x_km（河程位置 km）")
        lv = require_non_negative(l0_raw, f"第 {i} 个排污口的 l0（初始碳质 BOD mg/L）")
        loads_by_position.setdefault(xv, []).append(lv)

    merged: list[dict] = []
    for x_km in sorted(loads_by_position):
        loads = sorted(loads_by_position[x_km])
        total = 0.0
        for load in loads:
            total += load
        merged.append({"x_km": x_km, "l0": total, "n_merged": len(loads)})
    return merged


def validate_window(
    t_max_day: Any = None,
    x_max_km: Any = None,
    *,
    allow_none: bool = True,
) -> tuple[float | None, float | None]:
    """校验扫描窗口。t 与 x 二选一，均为严格正；都缺省时由上层给默认值。"""
    t_max = None
    x_max = None
    if t_max_day is not None:
        t_max = require_positive(t_max_day, "t_max_day（扫描时长 day）")
    if x_max_km is not None:
        x_max = require_positive(x_max_km, "x_max_km（扫描河程 km）")
    if t_max is not None and x_max is not None:
        raise ParameterError("t_max_day 与 x_max_km 二选一，不要同时给出")
    if not allow_none and t_max is None and x_max is None:
        raise ParameterError("必须给出 t_max_day 或 x_max_km 之一")
    return t_max, x_max


def validate_sweep_bounds(
    parameter: str,
    lo: Any,
    hi: Any,
    n_points: Any,
    *,
    csat: float | None = None,
) -> tuple[float, float, int]:
    """校验扫参区间。系数与流速区间严格为正；L0/D0 允许下界为 0；
    D0 的上界不得超过 Csat。"""
    n = require_int(n_points, "n_points（采样点数）", minimum=2)
    if parameter in ("l0", "d0"):
        lov = require_non_negative(lo, "lo（扫参区间下界）")
    else:
        lov = require_positive(lo, "lo（扫参区间下界）")
    hiv = require_positive(hi, "hi（扫参区间上界）")
    if parameter == "d0" and csat is not None and hiv > csat:
        raise ParameterError(f"d0 的扫参上界不能超过 csat（{csat}），收到 hi={hi!r}")
    if lov >= hiv:
        raise ParameterError(f"扫参区间要求 lo < hi，收到 lo={lo!r}, hi={hi!r}")
    return lov, hiv, n
