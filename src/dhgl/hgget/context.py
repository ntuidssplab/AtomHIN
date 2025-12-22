from typing import ClassVar
from contextlib import ContextDecorator
from functools import lru_cache, update_wrapper, _lru_cache_wrapper
from ..data.base import BaseHeteroGraphLike


class use_cache(ContextDecorator):  # pylint: disable=invalid-name

    enable_states: ClassVar[list[bool]] = [False]
    function_caches: ClassVar[list[set[_lru_cache_wrapper]]] = []

    def __init__(self, enable: bool = True):
        self.state = enable
        return

    def __enter__(self):
        use_cache.enable_states.append(self.state)
        if self.state:
            use_cache.function_caches.append(set())
        return self

    def __exit__(self, *ext):
        if use_cache.enable_states[-1]:
            cached_funcs = use_cache.function_caches.pop()
            if len(use_cache.function_caches) == 0:
                # print(
                #     f'cleaning following fns {[f.__name__ for f in cached_funcs]}'
                # )
                for func in cached_funcs:
                    func.cache_clear()
            else:
                use_cache.function_caches[-1] |= cached_funcs
        use_cache.enable_states.pop()
        return False


def my_cache(max_size: int = 1):

    def __my_cache(func):

        def new_func(hg: BaseHeteroGraphLike, *args):
            for key in [
                'feat', 'label', 'test_mask', 'val_mask', 'train_mask'
            ]:
                assert key in hg.ndata, (
                    'The input graph is not a valid heterogeneous graph. '
                    'Consider use dgl builtin api instead.'
                )
            return func(hg, *args)

        cached_func = lru_cache(maxsize=max_size)(new_func)
        update_wrapper(cached_func, func)

        def selector(*args, **kwargs):
            if use_cache.enable_states[-1]:
                assert use_cache.function_caches
                use_cache.function_caches[-1].add(cached_func)
                return cached_func(*args, **kwargs)
            return func(*args, **kwargs)

        return selector

    return __my_cache
