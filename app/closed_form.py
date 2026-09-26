"""Streeter-Phelps 氧垂模型闭式求值。

单位约定（服务内部统一口径）：
- 时间 t：天 day
- 河程 x：公里 km
- 流速 U：公里/天 km/day（1 m/s = 86.4 km/day）
- k1, k2：/day
- D0, L0, D, DO, Csat：mg/L

时间与河程唯一的换算关系：t = x / U。闭式公式只接受时间（天），
绝不允许把公里数直接代进去。
"""

from __future__ import annotations

import math

# k1 与 k2 视为相等的相对阈值。小于该相对偏差时分母 k2-k1 在数值上趋零，
# 必须改走 k2=k1 特解。临界点模块复用同一判定，保证两处口径一致。
RATE_EQUAL_REL_TOL = 1e-12


def rates_equal(k1: float, k2: float, *, rel_tol: float = RATE_EQUAL_REL_TOL) -> bool:
    """复氧系数与耗氧系数是否在数值上相等（走特解的判据）。"""
    return math.isclose(k1, k2, rel_tol=rel_tol, abs_tol=0.0)


def time_from_distance(x_km: float, u: float) -> float:
    """河程 -> 时间：t = x / U（天）。"""
    return x_km / u


def distance_from_time(t_day: float, u: float) -> float:
    """时间 -> 河程：x = U · t（公里）。"""
    return u * t_day


def bod(t_day: float, l0: float, k1: float) -> float:
    """碳质 BOD：L(t) = L0·exp(-k1·t)。"""
    return l0 * math.exp(-k1 * t_day)


def deficit(t_day: float, d0: float, l0: float, k1: float, k2: float) -> float:
    """亏氧 D(t) 的闭式解。

    一般解（k1 ≠ k2）：
        D(t) = k1 L0 / (k2 - k1) · (e^{-k1 t} - e^{-k2 t}) + D0 e^{-k2 t}
    特解（k1 = k2 = k，由极限得到）：
        D(t) = (D0 + k L0 t) · e^{-k t}
    """
    if rates_equal(k1, k2):
        return _deficit_equal_rates(t_day, d0, l0, k1)

    decay1 = math.exp(-k1 * t_day)
    decay2 = math.exp(-k2 * t_day)
    return k1 * l0 / (k2 - k1) * (decay1 - decay2) + d0 * decay2


def _deficit_equal_rates(t_day: float, d0: float, l0: float, k: float) -> float:
    """k2 = k1 = k 特解（分母为零，不能套一般公式）。"""
    return (d0 + k * l0 * t_day) * math.exp(-k * t_day)


def dissolved_oxygen(
    t_day: float, d0: float, l0: float, k1: float, k2: float, csat: float
) -> float:
    """实际溶解氧：DO(t) = Csat - D(t)。"""
    return csat - deficit(t_day, d0, l0, k1, k2)


def deficit_rate(t_day: float, d0: float, l0: float, k1: float, k2: float) -> float:
    """亏氧对时间的导数 dD/dt = k1·L(t) - k2·D(t)，临界点处为零。

    用 ODE 右端计算而不是对闭式解另做符号推导，便于与模型定义直接核对。
    """
    return k1 * bod(t_day, l0, k1) - k2 * deficit(t_day, d0, l0, k1, k2)


def deficit_at_distance(
    x_km: float, d0: float, l0: float, k1: float, k2: float, u: float
) -> float:
    """按河程求值：先以 t = x/U 折成时间，再代入闭式解。"""
    return deficit(time_from_distance(x_km, u), d0, l0, k1, k2)
