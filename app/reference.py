"""预置参考工况：排污口下游应出现明显氧垂，临界距离为正。

D0=2 mg/L、L0=20 mg/L、k1=0.3 /day、k2=0.6 /day、U=30 km/day（≈0.347 m/s）、
Csat=10 mg/L。解析结果 t_c≈1.959 day、x_c≈58.8 km、D_c≈5.56 mg/L。
"""

from __future__ import annotations

REFERENCE_INPUT = {
    "d0": 2.0,
    "l0": 20.0,
    "k1": 0.3,
    "k2": 0.6,
    "u": 30.0,
    "csat": 10.0,
}

# 沿程扫描默认窗口与点数（覆盖临界点约一个数量级的下游衰减距离）
REFERENCE_X_MAX_KM = 120.0
REFERENCE_N_POINTS = 401
