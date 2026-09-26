"""多排污口叠加后的临界点数值搜索（标准库手写，不引入科学计算依赖）。

单口工况的临界点靠 dD/dt=0 的解析公式直接解；多口叠加后总亏氧是若干段
「起点不同的指数函数之和」，导数为零的方程一般没有闭式根（三口以上尤其
如此），因此这里改用数值搜索，而且不能只找到一个极大值就交卷。

分段结构（这是本算法不漏峰的关键）：
- 各排口按河程升序合并后，时间原点为 t_i = x_i/U。
- 在任意两个相邻排口之间的开区间 (t_i, t_{i+1}) 内，「已经经过」的排口
  集合固定不变，此时总亏氧只是两个指数的线性组合
      D(t) = A·e^{-k1 t} + B·e^{-k2 t}        （k1 ≠ k2）
      D(t) = (A' + B'·t)·e^{-k1 t}            （k1 = k2 特解段）
  其导数在一个段内至多穿越零点一次，所以每段至多有一个局部极大。
- 系数 A、B 由闭式项解析重组得到（只改括号位置、不换公式），并在测试里
  逐点锁定「分段重组式 = multi_source.river_deficit 直和式」。
- 段内用二分找 dD/dt 的 +→- 穿越根，再用黄金分割取该峰的 D 最大值细化；
  每一段都独立检查，保证沿程所有局部极大都被找出来，再标注全局最深的
  一个为 global，其余作为 secondary 一并汇报。
- 排口注入处导数只会向上跳（新负荷立刻贡献 +k1·L_i），不可能在注入点
  本身形成极大，因此段端点不作为峰点，避免把上游旧峰重复计数。

全程没有任何一个上升段时（所有排口负荷都很小、复氧始终压得住），如实
返回空峰列表（exists=false），绝不凑一个正数出来。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import closed_form as cf
from .multi_source import MultiRiver, river_deficit
from .numeric import bisect_root, golden_section_max

# 段内二分/细化的时间精度（天）；x_tol 按段宽自适应时的下限
RATE_T_TOL = 1e-10
# |k2-k1| 小于该相对阈值时，重组系数里的 1/(k2-k1) 会放大浮点误差，
# 段内导数改走 closed_form 闭式项直和（更慢但数值稳定）；cf.rates_equal
# 的 1e-12 特解判据之外留出一段缓冲带。
NEAR_EQUAL_RATES_REL = 1e-6
# 端点导数贴近零的相对容带，用于判定「恰好零斜率」的退化情形
RATE_ZERO_REL_TOL = 1e-10
# 判定一个峰相对于相邻谷底足够「鼓包」的相对阈值，滤除纯数值抖动
PEAK_PROMINENCE_REL_TOL = 1e-9


def _near_equal_rates(k1: float, k2: float) -> bool:
    scale = max(abs(k1), abs(k2))
    return abs(k2 - k1) <= NEAR_EQUAL_RATES_REL * scale


def _direct_segment_rate(river: MultiRiver, through: int, t_day: float) -> float:
    """段内导数的闭式项直和求值（数值稳定路径）。

    与 multi_source.river_rate 的区别仅在于活动集合按段下标 through 截断，
    供「近等速率」缓冲带使用；每一项仍是 cf.deficit_rate 本身。
    """
    rate = cf.deficit_rate(t_day, river.d0, 0.0, river.k1, river.k2)
    for idx in range(through + 1):
        o = river.outfalls[idx]
        t_i = cf.time_from_distance(o.x_km, river.u)
        if t_i <= t_day:
            rate += cf.deficit_rate(t_day - t_i, 0.0, o.l0, river.k1, river.k2)
    return rate


@dataclass(frozen=True)
class MultiCriticalPoint:
    rank: int  # 1 = 全局最深
    is_global: bool
    t_c_day: float
    x_c_km: float
    d_c: float
    do_c: float
    u_km_day: float

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "is_global": self.is_global,
            "kind": "global" if self.is_global else "secondary",
            "t_c_day": self.t_c_day,
            "x_c_km": self.x_c_km,
            "critical_deficit_mg_l": self.d_c,
            "critical_do_mg_l": self.do_c,
            "u_km_day": self.u_km_day,
        }


@dataclass(frozen=True)
class MultiCriticalResult:
    exists: bool
    reason: str
    points: tuple[MultiCriticalPoint, ...]
    global_point: MultiCriticalPoint | None
    t_scan_max_day: float
    x_scan_max_km: float
    still_rising_at_end: bool

    def as_dict(self) -> dict:
        return {
            "exists": self.exists,
            "reason": self.reason,
            "n_local_maxima": len(self.points),
            "global": self.global_point.as_dict() if self.global_point else None,
            "peaks": [p.as_dict() for p in self.points],
            "t_scan_max_day": self.t_scan_max_day,
            "x_scan_max_km": self.x_scan_max_km,
            "still_rising_at_end": self.still_rising_at_end,
        }


# --------------------------------------------------------------------------
# 分段系数：把各闭式项按 e^{-k1 t} / e^{-k2 t} 重组。
# 这只是「换括号位置」——展开后与 cf.deficit 直和逐位相等，不引入近似。
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _SegmentCoeffs:
    # D(t) = A e^{-k1 t} + B e^{-k2 t}（k1 ≠ k2）
    a: float
    b: float


def _segment_coefficients(river: MultiRiver, through: int) -> _SegmentCoeffs:
    """活动排口下标集合为 [0..through]（through=-1 表示只有本底）时的
    段内系数 A、B。

    本底 D0 项：D0·e^{-k2 t}
    排口 i 的负荷项（t >= t_i）：
        k1·L_i/(k2-k1)·(e^{-k1(t-t_i)} - e^{-k2(t-t_i)})
      = [k1·L_i/(k2-k1)·e^{k1 t_i}]·e^{-k1 t}
        + [-k1·L_i/(k2-k1)·e^{k2 t_i}]·e^{-k2 t}
    """
    k1, k2 = river.k1, river.k2
    a = 0.0
    b = river.d0
    factor = k1 / (k2 - k1)
    for idx in range(through + 1):
        o = river.outfalls[idx]
        t_i = cf.time_from_distance(o.x_km, river.u)
        a += factor * o.l0 * math.exp(k1 * t_i)
        b -= factor * o.l0 * math.exp(k2 * t_i)
    return _SegmentCoeffs(a, b)


def segment_deficit(river: MultiRiver, through: int, t_day: float) -> float:
    """按段内重组系数求值；与 river_deficit 展开相同（测试锁定）。

    k1≈k2 的缓冲带里 1/(k2-k1) 会放大舍入误差，改走闭式项直和（同一份
    cf.deficit，只是不换括号）。"""
    if cf.rates_equal(river.k1, river.k2):
        return _segment_deficit_equal_rates(river, through, t_day)
    if _near_equal_rates(river.k1, river.k2):
        return _direct_segment_deficit(river, through, t_day)
    c = _segment_coefficients(river, through)
    return c.a * math.exp(-river.k1 * t_day) + c.b * math.exp(-river.k2 * t_day)


def segment_rate(river: MultiRiver, through: int, t_day: float) -> float:
    """段内 dD/dt = -k1·A·e^{-k1 t} - k2·B·e^{-k2 t}。"""
    if cf.rates_equal(river.k1, river.k2):
        return _segment_rate_equal_rates(river, through, t_day)
    if _near_equal_rates(river.k1, river.k2):
        return _direct_segment_rate(river, through, t_day)
    c = _segment_coefficients(river, through)
    return -river.k1 * c.a * math.exp(-river.k1 * t_day) - river.k2 * c.b * math.exp(
        -river.k2 * t_day
    )


def _direct_segment_deficit(river: MultiRiver, through: int, t_day: float) -> float:
    """段内亏氧的闭式项直和（近等速率缓冲带的稳定路径）。"""
    total = cf.deficit(t_day, river.d0, 0.0, river.k1, river.k2)
    for idx in range(through + 1):
        o = river.outfalls[idx]
        t_i = cf.time_from_distance(o.x_km, river.u)
        if t_i <= t_day:
            total += cf.deficit(t_day - t_i, 0.0, o.l0, river.k1, river.k2)
    return total


# k2 = k1 特解段：对活动排口直接按闭式特解重组
#   本底：D0·e^{-k t}
#   排口 i：k·L_i·(t - t_i)·e^{-k(t-t_i)}
#         = k·L_i·e^{k t_i}·t·e^{-k t} - k·L_i·t_i·e^{k t_i}·e^{-k t}


def _equal_rates_ab(river: MultiRiver, through: int) -> tuple[float, float]:
    """D(t) = (P + Q·t)·e^{-k t} 中的 (P, Q)。"""
    k = river.k1
    p = river.d0
    q = 0.0
    for idx in range(through + 1):
        o = river.outfalls[idx]
        t_i = cf.time_from_distance(o.x_km, river.u)
        w = k * o.l0 * math.exp(k * t_i)
        q += w
        p -= w * t_i
    return p, q


def _segment_deficit_equal_rates(river: MultiRiver, through: int, t_day: float) -> float:
    p, q = _equal_rates_ab(river, through)
    return (p + q * t_day) * math.exp(-river.k1 * t_day)


def _segment_rate_equal_rates(river: MultiRiver, through: int, t_day: float) -> float:
    # d/dt (P+Qt)e^{-kt} = (Q - k(P+Qt)) e^{-kt}
    p, q = _equal_rates_ab(river, through)
    k = river.k1
    return (q - k * (p + q * t_day)) * math.exp(-k * t_day)


# --------------------------------------------------------------------------
# 扫描窗口
# --------------------------------------------------------------------------


def default_scan_end(river: MultiRiver, *, max_growth: int = 60) -> float:
    """选一个把所有下游局部峰都包进去的扫描终点（自最后一个排口算起）。

    最后一个排口之后活动集合不再变化，D(t) 是两指数组合，其峰（若存在）
    距最后一个排口不超过约 1/min(k1,k2) 天的量级；为稳妥先取
    8/min(k1,k2)，再倍增直到终点导数确实 <= 0（下降段），保证峰不会落在
    窗口外。没有任何排口时从河头起算同样处理。
    """
    positions = river.t_positions
    t_last = positions[-1] if positions else 0.0
    through = len(river.outfalls) - 1
    tail = 8.0 / min(river.k1, river.k2)
    t_end = t_last + tail
    for _ in range(max_growth):
        if segment_rate(river, through, t_end) <= 0.0:
            return t_end
        t_end = t_last + (t_end - t_last) * 2.0
    # 理论上有界；保险起见返回当前上界并由 still_rising_at_end 标注
    return t_end


# --------------------------------------------------------------------------
# 多峰搜索
# --------------------------------------------------------------------------


def _zero_slope(value: float, scale: float) -> bool:
    return abs(value) <= RATE_ZERO_REL_TOL * max(1.0, abs(scale))


def _refine_peak(
    river: MultiRiver,
    through: int,
    lo: float,
    hi: float,
    t_guess: float,
) -> tuple[float, float]:
    """在峰根附近用黄金分割细化 (t*, D(t*))；段内单峰。

    括号以二分根为中心，宽度取段宽与耗氧/复氧时标中较小者并夹回段内，
    避免在远离峰的宽区间里因峰区函数值平缓而白白损失位置精度。
    """
    seg_width = hi - lo
    scale = min(seg_width, 4.0 / max(river.k1, river.k2))
    a = max(lo, t_guess - scale / 2.0)
    b = min(hi, t_guess + scale / 2.0)
    if not (a < t_guess < b):  # 根贴段端时退回到整段
        a, b = lo, hi
    tol = max(RATE_T_TOL, (b - a) * 1e-12)
    return golden_section_max(
        lambda t: segment_deficit(river, through, t),
        a,
        b,
        x_tol=tol,
        max_iter=300,
    )


def _scan_segments(river: MultiRiver, t_max: float) -> tuple[list[tuple[float, float, float, int]], bool]:
    """逐段搜索局部极大。

    返回 (峰列表, 终点是否仍在上升)。每个峰为 (t_star, d_star, t_root, through)。
    段内 dD/dt 至多一次 +→- 穿越：检测端点斜率符号，正转负即二分找根。
    """
    positions = list(river.t_positions)
    # 内部边界：落在扫描窗口内（严格小于 t_max）的排口时间
    internal = [t for t in positions if 0.0 < t < t_max]

    bounds = [0.0] + internal + [t_max]
    peaks: list[tuple[float, float, float, int]] = []

    for seg_idx in range(len(bounds) - 1):
        a, b = bounds[seg_idx], bounds[seg_idx + 1]
        # 段 (a,b) 内活动排口 = 时间原点 < b 的全部排口（b 处恰好注入的
        # 新排口不属于这一段，它在下一段才开始贡献）
        through = sum(1 for t_i in positions if t_i < b) - 1

        ra = segment_rate(river, through, a)
        rb = segment_rate(river, through, b)
        scale = max(1.0, abs(river_deficit(river, a)), abs(river_deficit(river, b)))

        rising_left = ra > 0.0 and not _zero_slope(ra, scale)
        falling_right = rb < 0.0 and not _zero_slope(rb, scale)

        t_root = None
        if rising_left and falling_right:
            t_root = bisect_root(
                lambda t: segment_rate(river, through, t),
                a,
                b,
                x_tol=max(RATE_T_TOL, (b - a) * 1e-12),
            )
        elif rising_left and _zero_slope(rb, scale):
            # 零斜率端：微向内探一点，确认是否真为峰（而不是拐点平台）
            probe = b - max(RATE_T_TOL, (b - a) * 1e-9)
            rp = segment_rate(river, through, probe)
            if rp <= 0.0 or _zero_slope(rp, scale):
                t_root = b if rp <= 0.0 else probe
        elif _zero_slope(ra, scale) and falling_right:
            probe = a + max(RATE_T_TOL, (b - a) * 1e-9)
            rp = segment_rate(river, through, probe)
            if rp <= 0.0:
                t_root = a

        if t_root is not None and t_root > 0.0:
            t_star, d_star = _refine_peak(river, through, a, b, t_root)
            # 峰必须落在段内部且确有抬升（滤掉平台/纯数值抖动造成的假峰）
            if a < t_star < b and t_star > 0.0:
                d_left = segment_deficit(river, through, a)
                d_right = segment_deficit(river, through, b)
                if d_star > max(d_left, d_right) + PEAK_PROMINENCE_REL_TOL * max(
                    1.0, abs(d_star)
                ):
                    peaks.append((t_star, d_star, t_root, through))

    # 终点导数只统计「已经经过」（t_i <= t_max）的排口；窗口之外的排口
    # 不参与窗口内曲线，绝不能把它们注入后的斜率算进来
    last_through = -1
    for t_i in positions:
        if t_i <= t_max:
            last_through += 1
        else:
            break
    end_rate = segment_rate(river, last_through, t_max)
    end_scale = max(1.0, abs(river_deficit(river, t_max)))
    still_rising = end_rate > 0.0 and not _zero_slope(end_rate, end_scale)
    return peaks, still_rising


def find_critical_multi(
    river: MultiRiver, t_max_day: float | None = None
) -> MultiCriticalResult:
    """在多口叠加曲线上沿全程搜索所有局部亏氧极大，并标注全局最深峰。"""
    if t_max_day is None:
        t_max = default_scan_end(river)
        window_auto = True
    else:
        t_max = float(t_max_day)
        window_auto = False

    raw_peaks, still_rising = _scan_segments(river, t_max)

    # 同一物理峰可能被相邻段各报到一次（根极贴近注入点时），按时间去重
    deduped = sorted(raw_peaks, key=lambda item: item[0])
    merged_peaks: list[tuple[float, float, float, int]] = []
    for pk in deduped:
        if merged_peaks and abs(pk[0] - merged_peaks[-1][0]) <= 1e-8 * max(1.0, pk[0]):
            if pk[1] > merged_peaks[-1][1]:
                merged_peaks[-1] = pk
        else:
            merged_peaks.append(pk)
    deduped = merged_peaks

    if not deduped:
        reason = (
            "全程未发现任何上升后转降的段：复氧始终压住耗氧，"
            "叠加曲线无局部亏氧极大（无临界点）"
        )
        return MultiCriticalResult(
            exists=False,
            reason=reason,
            points=(),
            global_point=None,
            t_scan_max_day=t_max,
            x_scan_max_km=cf.distance_from_time(t_max, river.u),
            still_rising_at_end=still_rising,
        )

    # 全局最深 = 临界亏氧最大（并列时取最上游的那个，结果确定）
    global_t = min(deduped, key=lambda item: (-item[1], item[0]))[0]

    # global 固定 rank=1，其余按河程顺序从 2 开始
    points: list[MultiCriticalPoint] = []
    secondary_rank = 2
    global_point = None
    for t_star, d_star, _root, _through in deduped:
        is_global = t_star == global_t
        point = MultiCriticalPoint(
            rank=1 if is_global else secondary_rank,
            is_global=is_global,
            t_c_day=t_star,
            x_c_km=cf.distance_from_time(t_star, river.u),
            d_c=d_star,
            do_c=river.csat - d_star,
            u_km_day=river.u,
        )
        points.append(point)
        if is_global:
            global_point = point
        else:
            secondary_rank += 1

    n_peak = len(points)
    if window_auto:
        reason = (
            f"数值分段搜索（二分 dD/dt 穿越根 + 黄金分割细化）共发现 "
            f"{n_peak} 个局部亏氧极大；rank=1 为全局最深临界点，其余为次级峰"
        )
    else:
        reason = (
            f"在给定扫描窗口内数值搜索到 {n_peak} 个局部亏氧极大；"
            "rank=1 为窗口内全局最深"
            + ("；注意窗口终点处亏氧仍在上升，窗口外可能还有峰" if still_rising else "")
        )
    return MultiCriticalResult(
        exists=True,
        reason=reason,
        points=tuple(points),
        global_point=global_point,
        t_scan_max_day=t_max,
        x_scan_max_km=cf.distance_from_time(t_max, river.u),
        still_rising_at_end=still_rising,
    )
