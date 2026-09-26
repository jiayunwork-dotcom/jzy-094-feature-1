"""临界点定位：亏氧最大（溶解氧最低）的位置。

解析公式直接由 dD/dt = 0 得到：

一般式（k1 ≠ k2）：
    t_c = ln[ (k2/k1) · (1 - D0·(k2-k1)/(k1·L0)) ] / (k2 - k1)

特解（k2 = k1 = k）：
    t_c = 1/k - D0/(k·L0)

下游（t_c > 0）出现氧垂的充要条件是初始亏氧正在增大：
    k1·L0 > k2·D0
（起点导数 dD/dt|_{t=0} = k1·L0 - k2·D0。）
不满足时曲线单调复氧（或恒值），不存在临界点，服务必须明确报告，
绝不能返回负河程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import closed_form as cf
from .numeric import bisect_root, golden_section_max


@dataclass(frozen=True)
class CriticalPoint:
    exists: bool
    reason: str
    branch: str  # "general" | "equal_rates"
    t_c_day: float | None
    x_c_km: float | None
    d_c: float | None
    do_c: float | None
    u_km_day: float

    def as_dict(self) -> dict:
        return {
            "exists": self.exists,
            "reason": self.reason,
            "branch": self.branch,
            "t_c_day": self.t_c_day,
            "x_c_km": self.x_c_km,
            "critical_deficit_mg_l": self.d_c,
            "critical_do_mg_l": self.do_c,
            "u_km_day": self.u_km_day,
        }


def _no_point(branch: str, reason: str, u: float) -> CriticalPoint:
    return CriticalPoint(
        exists=False,
        reason=reason,
        branch=branch,
        t_c_day=None,
        x_c_km=None,
        d_c=None,
        do_c=None,
        u_km_day=u,
    )


def _has_downstream_sag(d0: float, l0: float, k1: float, k2: float) -> bool:
    """起点处亏氧必须严格增大（dD/dt|_{t=0} > 0），下游才有氧垂。"""
    return k1 * l0 > k2 * d0


def find_critical(
    d0: float, l0: float, k1: float, k2: float, u: float, csat: float
) -> CriticalPoint:
    """解析定位临界点；非法系数由 validation 层拦截，此处只做数学判定。"""
    branch = "equal_rates" if cf.rates_equal(k1, k2) else "general"

    if not _has_downstream_sag(d0, l0, k1, k2):
        return _no_point(
            branch,
            "k1·L0 <= k2·D0：初始亏氧不再增大，曲线单调复氧，下游无临界点",
            u,
        )

    if branch == "equal_rates":
        # k2 = k1 = k 特解（此时 k1·L0 > k2·D0 已保证 L0 > 0）
        t_c = 1.0 / k1 - d0 / (k1 * l0)
    else:
        # 一般公式，与需求给出的形式逐项对应
        log_arg = (k2 / k1) * (1.0 - d0 * (k2 - k1) / (k1 * l0))
        if log_arg <= 0.0:
            # 数学上有正初始斜率时不该发生，留作防御性判定
            return _no_point(branch, "临界时刻对数项非正，无实临界时刻", u)
        t_c = math.log(log_arg) / (k2 - k1)

    # 解析公式在边界附近可能给出 0 或负值；下游临界点必须严格为正
    if not t_c > 0.0:
        return _no_point(branch, "解析临界时刻非正，下游无临界点", u)

    x_c = cf.distance_from_time(t_c, u)
    if not x_c > 0.0:
        return _no_point(branch, "临界河程非正，下游无临界点", u)

    d_c = cf.deficit(t_c, d0, l0, k1, k2)
    do_c = csat - d_c
    return CriticalPoint(
        exists=True,
        reason="dD/dt=0 给出下游严格正的临界时刻，氧垂存在",
        branch=branch,
        t_c_day=t_c,
        x_c_km=x_c,
        d_c=d_c,
        do_c=do_c,
        u_km_day=u,
    )


# --------------------------------------------------------------------------
# 数值核验（标准库手写）：用 dD/dt 的二分求根与 D(t) 的黄金分割复核解析值。
# 仅供 /critical 输出里的 numeric_check 字段与测试交叉验证，定位以解析解为准。
# --------------------------------------------------------------------------


def _bracket_negative_time(
    d0: float,
    l0: float,
    k1: float,
    k2: float,
    *,
    t_start: float = 1.0,
    max_growth: int = 60,
) -> float:
    """从 t_start 起倍增，直到找到 dD/dt < 0 的时刻，返回该上界。"""
    t = t_start
    for _ in range(max_growth):
        if cf.deficit_rate(t, d0, l0, k1, k2) < 0.0:
            return t
        t *= 2.0
    raise RuntimeError("无法为 dD/dt 找到负号端，函数可能不收敛")


def find_critical_numeric(
    d0: float,
    l0: float,
    k1: float,
    k2: float,
    u: float,
    csat: float,
    *,
    x_tol: float = 1e-10,
) -> dict:
    """纯数值方式求临界点（无根时 exists=false），用于与解析解互验。"""
    rate0 = cf.deficit_rate(0.0, d0, l0, k1, k2)
    if rate0 <= 0.0:
        return {
            "exists": False,
            "t_c_day": None,
            "x_c_km": None,
            "critical_deficit_mg_l": None,
            "critical_do_mg_l": None,
        }

    t_hi = _bracket_negative_time(d0, l0, k1, k2)
    t_root = bisect_root(
        lambda t: cf.deficit_rate(t, d0, l0, k1, k2),
        0.0,
        t_hi,
        x_tol=x_tol,
    )
    # 在根两侧用黄金分割取 D 的最大值，确认根确为峰点
    t_max, d_max = golden_section_max(
        lambda t: cf.deficit(t, d0, l0, k1, k2),
        max(0.0, t_root - 1.0 / max(k1, k2)),
        t_root + 1.0 / max(k1, k2),
        x_tol=x_tol,
    )
    return {
        "exists": True,
        "t_c_day": t_max,
        "x_c_km": cf.distance_from_time(t_max, u),
        "critical_deficit_mg_l": d_max,
        "critical_do_mg_l": csat - d_max,
    }
