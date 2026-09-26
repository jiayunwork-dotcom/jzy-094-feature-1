"""FastAPI 请求模型。Pydantic 负责结构与基本类型，系数取值合法性
（正数、有限、D0<=Csat 等）统一在 validation.py 里做，错误回 HTTP 400。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .sweep import SWEEP_PARAMETERS


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d0: float = Field(..., description="初始亏氧 mg/L")
    l0: float = Field(..., description="初始碳质 BOD mg/L")
    k1: float = Field(..., description="耗氧系数 /day（取值域由 validation 层校验：严格为正）")
    k2: float = Field(..., description="复氧系数 /day（取值域由 validation 层校验：严格为正）")
    u: float = Field(..., description="流速 km/day（1 m/s = 86.4 km/day，严格为正）")
    csat: float = Field(..., description="饱和溶解氧 mg/L（严格为正）")

    @field_validator("d0", "l0", "k1", "k2", "u", "csat", mode="before")
    @classmethod
    def _reject_bool_and_non_finite(cls, v):
        # Pydantic 默认会把 bool 吞成 0.0/1.0，这里显式拒绝；
        # NaN 会通过 float()，这里一并挡掉，Inf 等留给 validation 层统一报错
        if isinstance(v, bool):
            raise ValueError("不接受布尔值，必须是数值")
        if isinstance(v, (int, float)) and v != v:  # NaN 自比较
            raise ValueError("不接受 NaN")
        return v


class ProfileRequest(ModelInput):
    t_max_day: float | None = Field(
        default=None, description="扫描时长（天），与 x_max_km 二选一，严格为正"
    )
    x_max_km: float | None = Field(
        default=None, description="扫描河程（公里），与 t_max_day 二选一，严格为正"
    )
    n_points: int = Field(default=401, ge=2, le=10001, description="扫描采样点数")

    @field_validator("t_max_day", "x_max_km", mode="before")
    @classmethod
    def _reject_bool_window(cls, v):
        if isinstance(v, bool):
            raise ValueError("不接受布尔值，必须是数值")
        if isinstance(v, (int, float)) and v != v:
            raise ValueError("不接受 NaN")
        return v

    @field_validator("n_points", mode="before")
    @classmethod
    def _reject_bool_n_profile(cls, v):
        if isinstance(v, bool):
            raise ValueError("不接受布尔值，必须是整数")
        return v


class SweepRequest(ModelInput):
    parameter: Literal["k1", "k2", "l0", "d0", "u"] = Field(
        ..., description=f"批量扫参的系数，可选：{', '.join(SWEEP_PARAMETERS)}"
    )
    lo: float = Field(..., description="区间下界（k1/k2/u 必须为正；l0/d0 可取 0）")
    hi: float = Field(..., description="区间上界（严格为正，取值域由 validation 层校验）")
    n_points: int = Field(default=21, ge=2, le=10001, description="扫参采样点数")

    @field_validator("lo", "hi", mode="before")
    @classmethod
    def _reject_bool_bounds(cls, v):
        if isinstance(v, bool):
            raise ValueError("不接受布尔值，必须是数值")
        if isinstance(v, (int, float)) and v != v:
            raise ValueError("不接受 NaN")
        return v

    @field_validator("n_points", mode="before")
    @classmethod
    def _reject_bool_n(cls, v):
        if isinstance(v, bool):
            raise ValueError("不接受布尔值，必须是整数")
        return v
