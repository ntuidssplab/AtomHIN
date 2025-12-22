from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from typing import Any, Callable, Literal, no_type_check

from optuna.samplers import BaseSampler
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from ray.tune.search.sample import Domain


class ParametersInvalidError(ValueError):
    pass


class CVConfig(BaseModel):
    seed: int
    num_folds: int


class BaseTuneConfig(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric: str | list[str]
    mode: Literal['min', 'max'] | list[Literal['min', 'max']]
    space: dict[str, Any]
    repeat: int

    param_to_config: Callable[[dict], dict]
    trainer_fn: Callable[..., dict] | None = Field(
        None,
        deprecated='Now trainer config required to have member `config.run()`.'
    )
    points_to_evaluate: list[dict] | None
    sampler: BaseSampler | None
    reduction_factor: int | None = None
    cache_dir: str | None = None

    oom_report_value: float | list[float] | None = None
    """value to report when oom. This could avoid sampler from keeping exploring large model"""

    cv_config: CVConfig | None = None

    # @property
    # def hash_str(self) -> str:
    #     sha1 = hashlib.sha1()
    #     sha1.update(f'{str(self.sampler) = }'.encode())
    #     sha1.update(f'{self.metric = }'.encode())
    #     sha1.update(f'{self.mode = }'.encode())
    #     sha1.update(f'{self.repeat = }'.encode())
    #     for name, space in self.space.items():
    #         sha1.update(f'{name}: {space.domain_str}'.encode())
    #     return sha1.hexdigest()

    @property
    def multi_objective(self) -> bool:
        return not isinstance(self.metric, str)

    @property
    def main_mean_metric(self) -> bool:
        if self.multi_objective:
            return self.mean_metric[0]
        return self.mean_metric

    @property
    def main_mode(self) -> bool:
        if self.multi_objective:
            return self.mode[0]
        return self.mode

    @property
    def mean_metric(self) -> str | list[str]:
        if isinstance(self.metric, str):
            return f'{self.metric}/mean'
        return [f'{m}/mean' for m in self.metric]

    @property
    def hash_str(self) -> str:
        sha1 = hashlib.sha1(
            self.model_dump_json(
                exclude_none=True, exclude_defaults=True,
                exclude='param_to_config'
            ).encode()
        )
        return sha1.hexdigest()

    @field_validator('cache_dir', mode='after')
    @classmethod
    def check_is_abspath(cls, field):
        if field is None:
            return field
        assert os.path.abspath(field), f'Require an abspath, but got: {field}'
        return field

    @field_serializer('space', when_used='json')
    def serialize_space(self, space: dict[str, Domain | dict], _info):
        return {
            name: (s.domain_str if isinstance(s, Domain) else s)
            for name, s in space.items()
        }

    @field_serializer('trainer_fn', when_used='json-unless-none')
    def serialize_trainer_fn(self, trainer_fn: Callable, _info):
        return trainer_fn.__module__

    @field_serializer('sampler', when_used='json-unless-none')
    def serialize_sampler(self, sampler: BaseSampler, _info):
        return str(sampler)


@no_type_check
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
