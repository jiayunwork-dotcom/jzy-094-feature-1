"""多排污口叠加曲线的临界点数值搜索。

叠加多个口子后，总亏氧是若干段起点不同的指数函数之和，dD/dt = 0 一般
不再有闭式根，必须沿全程数值搜索：

1. 以各口子过流时刻 t_i 把全程切成若干分段（分段内导数连续，跳变只发生
   在 t_i 处，且跳变恒为 +k1·L0_i > 0，因此口子位置本身绝不可能是极大点）；
2. 每个分段内扫描导数符号，凡 + → - 穿越即框住一个局部亏氧峰；
3. 每个括号区间对导数做穷举二分（细分到浮点相邻）定位峰点，峰位置只由
   函数决定、与括号粗细无关，保证下游新增口子不改变上游峰的逐位结果；
4. 汇总所有局部峰，亏氧最深者标为全局临界点，其余作为次级峰一并汇报。

一个上升段都找不到（复氧始终压得住）时如实报告无临界点，绝不凑数。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import closed_form as cf
from .multi_profile import Outfall, search_horizon, total_deficit, total_deficit_rate
from .numeric import bisect_root_to_float_precision, linspace

# 每个分段内导数符号扫描的区间数（分段内导数光滑，16 段足以框住穿越点）
SEGMENT_SCAN_INTERVALS = 16


@dataclass(frozen=True)
class SagPeak:
    """一个局部亏氧峰（氧垂包）。is_global 标记全局最深的临界点。"""

    t_day: float
    x_km: float
    deficit_mg_l: float
    do_mg_l: float
    is_global: bool

    def as_dict(self) -> dict:
        return {
            "t_day": self.t_day,
            "x_km": self.x_km,
            "deficit_mg_l": self.deficit_mg_l,
            "do_mg_l": self.do_mg_l,
            "is_global": self.is_global,
        }


@dataclass(frozen=True)
class MultiCritical:
    """多口叠加曲线的临界点结论：全部局部峰 + 全局最深者。"""

    exists: bool
    reason: str
    n_peaks: int
    peaks: tuple[SagPeak, ...]  # 按河程升序
    global_peak: SagPeak | None
    u_km_day: float
    t_end_day: float  # 实际使用的搜索上界

    def as_dict(self) -> dict:
        g = self.global_peak
        return {
            "exists": self.exists,
            "reason": self.reason,
            "n_peaks": self.n_peaks,
            "global": (
                None
                if g is None
                else {
                    "t_c_day": g.t_day,
                    "x_c_km": g.x_km,
                    "critical_deficit_mg_l": g.deficit_mg_l,
                    "critical_do_mg_l": g.do_mg_l,
                }
            ),
            "peaks": [p.as_dict() for p in self.peaks],
            "u_km_day": self.u_km_day,
            "search_horizon_day": self.t_end_day,
        }


def _segment_peaks(
    d0: float,
    active: tuple[Outfall, ...],
    k1: float,
    k2: float,
    a: float,
    b: float,
) -> list[tuple[float, float]]:
    """在分段 [a, b] 内找局部亏氧峰，返回 (t*, D*) 列表。

    分段内导数连续：均匀扫描导数符号，凡 + → - 相邻符号变化即框住一个峰，
    再对导数做穷举二分（细分到浮点相邻）定位零点，峰位置因此只由函数
    决定、与括号粗细无关；峰值为该点处的总亏氧。
    """
    ts = linspace(a, b, SEGMENT_SCAN_INTERVALS + 1)
    rates = [total_deficit_rate(t, d0, active, k1, k2) for t in ts]
    found: list[tuple[float, float]] = []
    for j in range(SEGMENT_SCAN_INTERVALS):
        r_lo, r_hi = rates[j], rates[j + 1]
        if r_lo > 0.0 and r_hi <= 0.0:
            t_star = bisect_root_to_float_precision(
                lambda t: total_deficit_rate(t, d0, active, k1, k2),
                ts[j],
                ts[j + 1],
            )
            found.append((t_star, total_deficit(t_star, d0, active, k1, k2)))
    return found


def _dedupe(peaks: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """合并细化后几乎重合的峰（保留亏氧更深者），按时刻升序返回。"""
    merged: list[tuple[float, float]] = []
    for t, d in sorted(peaks):
        if merged and t - merged[-1][0] <= tol:
            if d > merged[-1][1]:
                merged[-1] = (t, d)
        else:
            merged.append((t, d))
    return merged


def find_all_peaks(
    d0: float,
    outfalls: tuple[Outfall, ...],
    k1: float,
    k2: float,
    u: float,
    csat: float,
) -> MultiCritical:
    """沿全程数值搜索所有局部亏氧峰，并标出全局临界点。

    outfalls 必须是 validation.validate_outfalls 规范化后的结果（按位置
    升序、同位已合并），否则分段切分与因果性都不再成立。
    """
    t_end = search_horizon(d0, outfalls, k1, k2)

    # 分段边界：起点 0、各口子过流时刻、搜索上界
    bounds = [0.0] + [o.t_day for o in outfalls] + [t_end]

    raw: list[tuple[float, float]] = []
    n_active = 0
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        # 该分段激活的口子：位置不晚于分段左端（口子上游不受其影响）
        while n_active < len(outfalls) and outfalls[n_active].t_day <= a:
            n_active += 1
        active = outfalls[:n_active]
        raw.extend(_segment_peaks(d0, active, k1, k2, a, b))

    peaks = _dedupe(raw, tol=1e-9)

    if not peaks:
        return MultiCritical(
            exists=False,
            reason="全程未找到亏氧上升段：各口子负荷均被复氧压住，无临界点",
            n_peaks=0,
            peaks=(),
            global_peak=None,
            u_km_day=u,
            t_end_day=t_end,
        )

    d_deepest = max(d for _, d in peaks)
    sag_peaks: list[SagPeak] = []
    global_marked = False
    for t, d in peaks:
        # 并列最深时取最上游的一个为全局临界点
        is_global = (d == d_deepest) and not global_marked
        global_marked = global_marked or is_global
        sag_peaks.append(
            SagPeak(
                t_day=t,
                x_km=cf.distance_from_time(t, u),
                deficit_mg_l=d,
                do_mg_l=csat - d,
                is_global=is_global,
            )
        )

    global_peak = next(p for p in sag_peaks if p.is_global)
    return MultiCritical(
        exists=True,
        reason=(
            f"沿程共找到 {len(sag_peaks)} 个局部亏氧峰，"
            f"全局临界点位于 x={global_peak.x_km:.6g} km"
        ),
        n_peaks=len(sag_peaks),
        peaks=tuple(sag_peaks),
        global_peak=global_peak,
        u_km_day=u,
        t_end_day=t_end,
    )
