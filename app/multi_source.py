"""多排污口叠加的亏氧曲线合成。

叠加的物理依据：在给定 k1、k2 下，Streeter-Phelps 亏氧方程
    dD/dt = k1·L - k2·D
对负荷是线性的。设排污口 i 位于河程 x_i（时间原点 t_i = x_i/U），其初始
碳质 BOD 负荷为 L_i，则该口只对其下游（t >= t_i）有贡献，把单口闭式解
的时间原点挪到 t_i 即可：

    D_total(t) = cf.deficit(t,        D0_bg, 0,   k1, k2)
               + Σ_{t_i <= t} cf.deficit(t - t_i, 0, L_i, k1, k2)

即「本底亏氧 D0_bg 沿程衰减的残余」+「所有已经经过的排污口各自闭式贡献
之和」。每一项都直接调用 closed_form.py 的闭式求值（k2=k1 时自动走同一
份特解），时间/河程换算也共用 cf.time_from_distance /
cf.distance_from_time，本模块不另抄任何近似公式。

BOD 同理按线性叠加：
    L_total(t) = Σ_{t_i <= t} L_i·exp(-k1·(t - t_i))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import closed_form as cf
from .numeric import linspace


@dataclass(frozen=True)
class Outfall:
    """一个排污口：河程位置 x_km（>= 0）与初始碳质 BOD 负荷 L0（>= 0）。"""

    x_km: float
    l0: float


@dataclass(frozen=True)
class MultiRiver:
    """规范化后的多口河段：排口已按河程升序排列，同位排口负荷已合并。"""

    d0: float
    k1: float
    k2: float
    u: float
    csat: float
    outfalls: tuple[Outfall, ...]
    n_submitted: int = 0
    n_coincident_merges: int = 0

    @property
    def t_positions(self) -> tuple[float, ...]:
        """各排口对应的时间原点 t_i = x_i/U（天），与 outfalls 同序。"""
        return tuple(cf.time_from_distance(o.x_km, self.u) for o in self.outfalls)


def build_river(
    d0: float,
    k1: float,
    k2: float,
    u: float,
    csat: float,
    outfalls: list[tuple[float, float]] | list[Outfall],
    *,
    n_submitted: int | None = None,
    n_coincident_merges: int = 0,
) -> MultiRiver:
    """由已校验的 (x_km, l0) 序列构造规范化河段。

    显式的同位处理策略：位置完全相同的排口按「同位负荷相加」合并为一个
    等效排口（线性叠加允许这样做）。合并后按河程升序排列，使后续求和次序
    与提交顺序无关。输入通常来自 validation.validate_multi_inputs。
    """
    normalized = [
        o if isinstance(o, Outfall) else Outfall(float(o[0]), float(o[1]))
        for o in outfalls
    ]
    normalized.sort(key=lambda o: o.x_km)

    merged: list[Outfall] = []
    merges = 0
    for o in normalized:
        if merged and merged[-1].x_km == o.x_km:
            # math.fsum 保证同位多个负荷合并结果与提交顺序无关
            combined = math.fsum((merged[-1].l0, o.l0))
            merged[-1] = Outfall(merged[-1].x_km, combined)
            merges += 1
        else:
            merged.append(o)

    return MultiRiver(
        d0=d0,
        k1=k1,
        k2=k2,
        u=u,
        csat=csat,
        outfalls=tuple(merged),
        n_submitted=n_submitted if n_submitted is not None else len(normalized),
        # 调用方（validation 层）未显式给计数时，采用这里实际合并出的次数
        n_coincident_merges=(
            n_coincident_merges if n_coincident_merges else merges
        ),
    )


def river_bod(river: MultiRiver, t_day: float) -> float:
    """总碳质 BOD：只统计 t_i <= t 的排口，各自 L_i·exp(-k1·(t - t_i))。"""
    total = 0.0
    for o, t_i in zip(river.outfalls, river.t_positions):
        if t_i <= t_day:
            total += cf.bod(t_day - t_i, o.l0, river.k1)
    return total


def river_deficit(river: MultiRiver, t_day: float) -> float:
    """总亏氧：本底亏氧沿程衰减 + 所有已经过排口的闭式贡献之和。

    每个贡献都调用 cf.deficit 本身（d0=0 的纯负荷项），k2=k1 时自动走
    closed_form 里的同一份特解，不在这里重推公式。
    """
    # 本底亏氧：从河道最上游 t=0 起按 D0·e^{-k2 t} 传递（等价于 l0=0 的单口项）
    total = cf.deficit(t_day, river.d0, 0.0, river.k1, river.k2)
    for o, t_i in zip(river.outfalls, river.t_positions):
        if t_i <= t_day:
            total += cf.deficit(t_day - t_i, 0.0, o.l0, river.k1, river.k2)
    return total


def river_rate(
    river: MultiRiver, t_day: float, *, through_index: int | None = None
) -> float:
    """总亏氧对时间的导数 dD/dt，供临界点搜索交叉核验。

    through_index 给定时只累加下标 <= through_index 的排口（分段边界处用，
    显式区分排口注入前后的左/右极限）；缺省时统计所有 t_i <= t 的排口。
    """
    rate = cf.deficit_rate(t_day, river.d0, 0.0, river.k1, river.k2)
    positions = river.t_positions
    for idx, (o, t_i) in enumerate(zip(river.outfalls, positions)):
        if through_index is not None:
            if idx > through_index:
                break
            if t_i > t_day:
                continue
        elif t_i > t_day:
            break
        rate += cf.deficit_rate(t_day - t_i, 0.0, o.l0, river.k1, river.k2)
    return rate


def river_deficit_at_distance(river: MultiRiver, x_km: float) -> float:
    """按河程求值：先以 t = x/U 折成时间，再叠加。绝不把公里数直接代入。"""
    return river_deficit(river, cf.time_from_distance(x_km, river.u))


@dataclass(frozen=True)
class MultiProfilePoint:
    t_day: float
    x_km: float
    l_mg_l: float
    deficit_mg_l: float
    do_mg_l: float

    def as_dict(self) -> dict:
        return {
            "t_day": self.t_day,
            "x_km": self.x_km,
            "bod_mg_l": self.l_mg_l,
            "deficit_mg_l": self.deficit_mg_l,
            "do_mg_l": self.do_mg_l,
        }


def scan_multi_profile(
    river: MultiRiver, t_max_day: float, n_points: int
) -> list[MultiProfilePoint]:
    """在 [0, t_max] 上等间距扫描叠加后的总 BOD/亏氧/DO 曲线。"""
    points: list[MultiProfilePoint] = []
    for t in linspace(0.0, t_max_day, n_points):
        d = river_deficit(river, t)
        points.append(
            MultiProfilePoint(
                t_day=t,
                x_km=cf.distance_from_time(t, river.u),
                l_mg_l=river_bod(river, t),
                deficit_mg_l=d,
                do_mg_l=river.csat - d,
            )
        )
    return points
