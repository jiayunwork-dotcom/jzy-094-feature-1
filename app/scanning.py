"""沿程扫描：沿排污口下游按时间/河程生成亏氧与 DO 曲线，并在网格上
找出窗口内的亏氧最优点，再用黄金分割细化。所有求值都先把河程用
t = x/U 折成时间，绝不把公里数当时间代入。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import closed_form as cf
from .critical import CriticalPoint
from .numeric import golden_section_max, linspace


@dataclass(frozen=True)
class ProfilePoint:
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


def scan_profile(
    d0: float,
    l0: float,
    k1: float,
    k2: float,
    u: float,
    csat: float,
    t_max_day: float,
    n_points: int,
) -> list[ProfilePoint]:
    """在 [0, t_max] 天上等间距扫描，同步给出河程 x = U·t。"""
    points: list[ProfilePoint] = []
    for t in linspace(0.0, t_max_day, n_points):
        x = cf.distance_from_time(t, u)
        d = cf.deficit(t, d0, l0, k1, k2)
        points.append(
            ProfilePoint(
                t_day=t,
                x_km=x,
                l_mg_l=cf.bod(t, l0, k1),
                deficit_mg_l=d,
                do_mg_l=csat - d,
            )
        )
    return points


def grid_argmax(points: list[ProfilePoint]) -> int:
    """网格上亏氧最大点的下标。"""
    return max(range(len(points)), key=lambda i: points[i].deficit_mg_l)


def refine_window_max(
    d0: float,
    l0: float,
    k1: float,
    k2: float,
    u: float,
    csat: float,
    t_max_day: float,
    n_points: int,
    *,
    x_tol: float = 1e-10,
) -> dict:
    """先网格粗扫定位，再在峰点两侧网格区间内黄金分割细化。

    返回的是「扫描窗口内」的亏氧最大点；当氧垂临界点落在窗口之外时，
    它与解析临界点不同（由上层标注 within_window）。
    """
    points = scan_profile(
        d0, l0, k1, k2, u, csat, t_max_day, n_points
    )
    idx = grid_argmax(points)
    n = n_points
    dt = t_max_day / (n - 1)

    lo_t = points[max(0, idx - 1)].t_day
    hi_t = points[min(n - 1, idx + 1)].t_day
    if idx == 0:
        hi_t = dt
    if idx == n - 1:
        lo_t = t_max_day - dt

    t_star, d_star = golden_section_max(
        lambda t: cf.deficit(t, d0, l0, k1, k2),
        lo_t,
        hi_t,
        x_tol=x_tol,
    )
    return {
        "t_day": t_star,
        "x_km": cf.distance_from_time(t_star, u),
        "deficit_mg_l": d_star,
        "do_mg_l": csat - d_star,
        "at_window_boundary": idx == 0 or idx == n - 1,
    }


def analytic_within_window(
    critical: CriticalPoint, t_max_day: float
) -> bool:
    """解析临界点是否落在本次扫描窗口内（含边界）。"""
    return (
        critical.exists
        and critical.t_c_day is not None
        and critical.t_c_day <= t_max_day
    )
