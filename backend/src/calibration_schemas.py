from datetime import datetime
import math
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class BatchInput(BaseModel):
    building: str = Field(min_length=1, max_length=40)
    position: str = Field(min_length=2, max_length=60)
    captured_at: datetime
    role: Literal['train', 'holdout']
    note: str = Field(default='', max_length=300)
    samples: list['SampleInput'] = Field(min_length=1, max_length=200)

    @field_validator('building', 'position')
    @classmethod
    def stripped(cls, v):
        if not v.strip():
            raise ValueError('不可全为空格')
        return v.strip()

    @field_validator('captured_at')
    @classmethod
    def naive(cls, v):
        if v.tzinfo:
            raise ValueError('采集时刻使用本地无时区时间')
        return v


class SampleInput(BaseModel):
    source: Literal['equipment', 'impact', 'speech']
    energies: list[float] = Field(min_length=8, max_length=8)

    @field_validator('energies')
    @classmethod
    def finite(cls, v):
        if not all(math.isfinite(x) and 0 <= x <= 1000 for x in v) or sum(v) <= 0:
            raise ValueError('8个频带须非负、有限且总能量大于0')
        return v


class CorrectionInput(BaseModel):
    source: Literal['equipment', 'impact', 'speech']
    energies: list[float] = Field(min_length=8, max_length=8)
    reason: str = Field(min_length=6, max_length=300)

    @field_validator('energies')
    @classmethod
    def finite(cls, v):
        if not all(math.isfinite(x) and 0 <= x <= 1000 for x in v) or sum(v) <= 0:
            raise ValueError('8个频带须非负、有限且总能量大于0')
        return v

    @field_validator('reason')
    @classmethod
    def stripped(cls, v):
        if not v.strip():
            raise ValueError('更正原因不可为空')
        return v.strip()


class EvaluateInput(BaseModel):
    building: str = Field(min_length=1, max_length=40)
    position: str = Field(min_length=2, max_length=60)
    train_batch_ids: list[int] = Field(min_length=1, max_length=20)
    holdout_batch_ids: list[int] = Field(min_length=1, max_length=20)

    @field_validator('train_batch_ids', 'holdout_batch_ids')
    @classmethod
    def unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('同一批次不能重复选择')
        if any(x <= 0 for x in v):
            raise ValueError('批次编号须为正整数')
        return v

    @model_validator(mode='after')
    def disjoint(self):
        if set(self.train_batch_ids) & set(self.holdout_batch_ids):
            raise ValueError('同一次采集整批只能用作训练或留出，不能同时选择')
        self.building = self.building.strip()
        self.position = self.position.strip()
        return self
