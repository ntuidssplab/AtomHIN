import os
from pydantic import BaseModel, model_validator, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    'filter_env_private_fields', 'seed_all', 'correct_module_path',
    'BaseConfig'
]


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        extra='forbid',
        frozen=True,
        use_attribute_docstrings=True,
    )


def filter_env_private_fields(config_cls: type[BaseModel]):
    """Filter input fields starts with '_'. This is useful for the case using .env
    as input since it can use something like `a_field=${_var}`.

    Example:
    ```
    @filter_env_private_fields
    class MyConfig(BaseSettings):
        ...
    ```

    Note that this will keep the fields starts with `_env_`
    """

    @model_validator(mode='before')
    def filter_env_private(data: dict[str, str], _info):

        def _filter(data: dict[str, str]):
            for k, v in data.items():
                if k.startswith('_') and not k.startswith('_env_'):
                    continue
                if isinstance(v, dict):
                    yield k, dict(_filter(v))
                else:
                    yield k, v

        return dict(_filter(data))

    return create_model(
        config_cls.__name__,
        __base__=config_cls,
        __module__=config_cls.__module__,
        __validators__={'__fiter_env_private_fields': filter_env_private},
    )


def correct_module_path(__file: str):

    def get_module_path():
        if os.path.basename(__file) == '__main__.py':
            return os.path.dirname(__file)
        path, _ = os.path.splitext(__file)
        return path

    def traverse(module_path: str):
        if os.path.samefile(module_path, os.getcwd()):
            return
        remain, module = os.path.split(module_path)
        yield module
        yield from traverse(remain)

    module_path = '.'.join(reversed(list(traverse(get_module_path()))))
    return module_path or '.'


def seed_all(seed):
    # pylint: disable=import-outside-toplevel
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
