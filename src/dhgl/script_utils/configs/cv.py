from __future__ import annotations

from pydantic import field_validator

from dhgl.script_utils import BaseConfig


class CVConfig(BaseConfig):

    seed: int
    ith_fold: int
    num_folds: int

    @field_validator('seed', mode='before')
    @classmethod
    def parse_int(cls, v):
        if isinstance(v, str) and v.startswith('0x'):
            return int(v, 16)
        return v
