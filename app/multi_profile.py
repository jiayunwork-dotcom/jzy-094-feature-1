"""多排污口叠加的沿程曲线合成。

物理依据：给定 k1、k2 时亏氧方程 dD/dt = k1·L - k2·D 对负荷是线性的，
因此多个排污口的总亏氧可以逐项叠加：

- 来水本底亏氧 d0 自河道最上游（x = 0）起随水流衰减：d0·e^{-k2·t}，
  即单口闭式解中 L0 = 0 的特例，沿程处处存在；
- 位于 x_i（对应时刻 t_i = x_i / U）的排污口，其贡献是把单口闭式解的
  时间原点平移到 t_i：D_i(t) = D_closed(t - t_i; D0=0, L0_i)，
  且仅对 t >= t_i 的下游点生效（口子上游不受它影响）；
- 沿程任意一点的总亏氧 = 本底残余 + 所有位置不晚于该点的口子贡献之和。

所有求值都复用 app/closed_form.py 的闭式解与 t = x/U 换算，不另立公式。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import closed_form as cf
from .critical import find_critical
from .numeric import linspace
from .scanning import ProfilePoint

# 搜索/默认扫描窗口在最后一个可能峰之后再延伸的衰减时标个数，
# 让曲线尾部充分展示复氧衰减段
HORIZON_TAIL_FOLDINGS = 5.0


@dataclass(frozen=True)
class Outfall:
    """规范化后的排污口：位置已同时按河程（km）与过流时刻（day）给出。"""

    x_km: float
    t_day: float
    l0: float


def prepare_outfalls(canonical: list[dict], u: float) -> tuple[Outfall, ...]:
    """把 validation.validate_outfalls 的规范输出（已按位置升序、同位合并）
    转成计算用 Outfall 元组，河程经 t = x/U 折成时刻。"""
    return tuple(
        Outfall(
            x_km=item["x_km"],
            t_day=cf.time_from_distance(item["x_km"], u),
            l0=item["l0"],
        )
        for item in canonical
    )


def total_deficit(
    t_day: float, d0: float, outfalls: tuple[Outfall, ...], k1: float, k2: float
) -> float:
    """t 时刻（自河道最上游起算）的总亏氧：本底残余 + 各已过口子贡献。"""
    d = cf.deficit(t_day, d0, 0.0, k1, k2)  # 本底亏氧沿途衰减后的残余
    for o in outfalls:
        if t_day >= o.t_day:
            d += cf.deficit(t_day - o.t_day, 0.0, o.l0, k1, k2)
    return d


def total_bod(t_day: float, outfalls: tuple[Outfall, ...], k1: float) -> float:
    """t 时刻水中的总碳质 BOD：各已过口子残余 BOD 之和。"""
    total = 0.0
    for o in outfalls:
        if t_day >= o.t_day:
            total += cf.bod(t_day - o.t_day, o.l0, k1)
    return total


def total_deficit_rate(
    t_day: float, d0: float, outfalls: tuple[Outfall, ...], k1: float, k2: float
) -> float:
    """总亏氧对时间的导数 dD/dt = k1·L_total - k2·D_total（逐口子叠加）。

    在口子位置 t_i 处导数有 +k1·L0_i 的向上跳变；在 t = t_i 处求值时计入
    该口贡献，给出的恰是分段右侧极限，供分段扫描的左端点使用。
    """
    rate = cf.deficit_rate(t_day, d0, 0.0, k1, k2)
    for o in outfalls:
        if t_day >= o.t_day:
            rate += cf.deficit_rate(t_day - o.t_day, 0.0, o.l0, k1, k2)
    return rate


def search_horizon(
    d0: float, outfalls: tuple[Outfall, ...], k1: float, k2: float
) -> float:
    """全程数值搜索的时间上界（day）。

    每个口子的贡献在其下游 t_c0（D0=0 的单口临界时刻，与负荷大小无关）
    之后单调衰减，本底亏氧全程单调衰减，因此最后一个可能的局部峰不会晚于
    「最后口子的时刻 + t_c0」。再向后延若干个衰减时标作为曲线尾部。
    """
    t_last = max(o.t_day for o in outfalls)
    # 复用单口解析临界公式：D0=0 时 t_c 与 L0 无关，任取 L0=1、U=1、Csat=1
    cp0 = find_critical(0.0, 1.0, k1, k2, 1.0, 1.0)
    t_c0 = cp0.t_c_day
    assert t_c0 is not None and t_c0 > 0.0  # D0=0、L0>0 时单口氧垂必然存在
    return t_last + t_c0 + HORIZON_TAIL_FOLDINGS / min(k1, k2)


def scan_multi_profile(
    d0: float,
    outfalls: tuple[Outfall, ...],
    k1: float,
    k2: float,
    u: float,
    csat: float,
    t_max_day: float,
    n_points: int,
) -> list[ProfilePoint]:
    """在 [0, t_max] 天上等间距扫描叠加后的亏氧/DO 曲线（x = U·t 同步给出）。"""
    points: list[ProfilePoint] = []
    for t in linspace(0.0, t_max_day, n_points):
        d = total_deficit(t, d0, outfalls, k1, k2)
        points.append(
            ProfilePoint(
                t_day=t,
                x_km=cf.distance_from_time(t, u),
                l_mg_l=total_bod(t, outfalls, k1),
                deficit_mg_l=d,
                do_mg_l=csat - d,
            )
        )
    return points
