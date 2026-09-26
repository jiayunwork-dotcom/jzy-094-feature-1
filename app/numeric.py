"""仅依赖标准库的数值小工具：线性网格、二分求根、黄金分割寻优。

刻意不引入 NumPy/SciPy 等重型数值库——本服务只需要一维求根与单峰寻优，
教科书迭代即可，且更便于审计。
"""

from __future__ import annotations

from collections.abc import Callable
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def linspace(start: float, stop: float, n_points: int) -> list[float]:
    """生成等间距网格（含两端）。n_points 必须 >= 2。"""
    if n_points < 2:
        raise ValueError("linspace 需要至少 2 个点")
    step = (stop - start) / (n_points - 1)
    return [start + step * i for i in range(n_points)]


def bisect_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    x_tol: float = 1e-10,
    f_tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """在 [a, b] 上对连续函数 f 做二分求根，要求 f(a)·f(b) <= 0。

    端点本身为根时直接返回端点。
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("二分求根需要两端函数值异号")

    lo, hi = a, b
    flo = fa
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) <= f_tol or (hi - lo) <= x_tol:
            return mid
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2.0


def golden_section_max(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    x_tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    """在 [a, b] 上用黄金分割搜索单峰函数的最大值，返回 (x*, f(x*))。"""
    h = (b - a) / PHI
    c = b - h
    d = a + h
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) <= x_tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            h = (b - a) / PHI
            c = b - h
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            h = (b - a) / PHI
            d = a + h
            fd = f(d)
    x_star = (a + b) / 2.0
    return x_star, f(x_star)
