"""输入校验。任何非法系数（非数值、NaN/Inf、非正、越界等）都抛
ParameterError，由接口层统一转成 HTTP 400。
"""

from __future__ import annotations

import math
from typing import Any

# 多口工况允许的排污口数量上限。逐段搜索与沿程扫描对排口数是 O(N)/O(N·P)，
# 上万已远超任何真实河段的排口密度，超过即视为明显不合理的请求。
MAX_OUTFALLS = 10000


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
    k1v = require_positive(k1, "k1（耗氧系数 /day）")
    k2v = require_positive(k2, "k2（复氧系数 /day）")
    uv = require_positive(u, "u（流速 km/day，1 m/s = 86.4 km/day）")
    csatv = require_positive(csat, "csat（饱和溶解氧 mg/L）")
    l0v = require_non_negative(l0, "l0（初始碳质 BOD mg/L）")
    d0v = require_non_negative(d0, "d0（初始亏氧 mg/L）")
    if d0v > csatv:
        raise ParameterError(
            f"d0（初始亏氧 {d0v}）不能超过 csat（饱和溶解氧 {csatv}）"
        )
    return {
        "d0": d0v,
        "l0": l0v,
        "k1": k1v,
        "k2": k2v,
        "u": uv,
        "csat": csatv,
    }


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


def validate_multi_inputs(
    d0: Any,
    k1: Any,
    k2: Any,
    u: Any,
    csat: Any,
    outfalls: Any,
) -> dict:
    """校验多排污口工况。

    共用系数 k1/k2/u/csat 与本底亏氧 d0 的规则同 validate_model_inputs
    （本工况没有「全河段统一的 L0」，每个排口自带负荷）。排污口列表：
    - 必须是非空列表；空列表明确报错（不是静默当作零负荷）
    - 数量不得超过 MAX_OUTFALLS
    - 每个元素必须能取到 x_km/l0（dict 或对象属性），且均为有限数值
    - 河程位置 >= 0（负河程无法定位在河段上）、负荷 >= 0
    任何一项错误都必须带上下标，能定位到具体是哪个口子。

    同位排口的明确策略：这里先不做合并，只统计并在返回值里报告需要合并
    的对数；实际的「同位负荷相加」由 multi_source.build_river 执行（线性
    叠加的直接推论），使该策略有唯一实现位置。
    """
    base = validate_model_inputs(d0, 0.0, k1, k2, u, csat)

    if not isinstance(outfalls, list):
        raise ParameterError(
            f"outfalls（排污口列表）必须是数组/列表，收到 {type(outfalls).__name__}"
        )
    if len(outfalls) == 0:
        raise ParameterError("outfalls（排污口列表）不能为空：至少需要一个排污口")
    if len(outfalls) > MAX_OUTFALLS:
        raise ParameterError(
            f"排污口数量 {len(outfalls)} 超过上限 {MAX_OUTFALLS}，"
            "明显不合理，请核对输入"
        )

    pairs: list[tuple[float, float]] = []
    positions: list[float] = []
    for idx, item in enumerate(outfalls):
        tag = f"排污口[{idx}]"
        if isinstance(item, dict):
            x_raw, l_raw = item.get("x_km"), item.get("l0")
        elif isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ParameterError(
                    f"{tag} 若以数组形式给出必须恰为 [x_km, l0] 两个元素"
                )
            x_raw, l_raw = item
        elif hasattr(item, "x_km") and hasattr(item, "l0"):
            x_raw, l_raw = item.x_km, item.l0
        else:
            raise ParameterError(
                f"{tag} 必须同时给出 x_km（河程位置 km）与 l0（初始碳质 BOD mg/L）"
            )
        xv = require_non_negative(x_raw, f"{tag} 的 x_km（河程位置 km）")
        lv = require_non_negative(l_raw, f"{tag} 的 l0（初始碳质 BOD mg/L）")
        pairs.append((xv, lv))
        positions.append(xv)

    # 同位（河程完全相同）排口数量统计——合并本身由 build_river 完成
    sorted_positions = sorted(positions)
    n_coincident = sum(
        1 for a, b in zip(sorted_positions, sorted_positions[1:]) if a == b
    )

    return {
        "d0": base["d0"],
        "k1": base["k1"],
        "k2": base["k2"],
        "u": base["u"],
        "csat": base["csat"],
        "outfalls": pairs,
        "n_submitted": len(pairs),
        "n_coincident_pairs": n_coincident,
    }
