from __future__ import annotations
from typing import (ClassVar, Dict, List, Type, Protocol, Literal)


class MetricsLike(Protocol):
    """static: describes information of this criterion"""

    def better_than(self, rhs: BaseMetrics) -> bool:
        ...

    def __str__(self) -> str:
        ...

    names: ClassVar[List[str]]

    @property
    def value(self) -> float:
        ...


class BaseMetrics:

    def __init__(self, value) -> None:
        self._value = value

    def better_than(self, rhs: BaseMetrics) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    names: ClassVar[List[str]]

    @property
    def value(self):
        return self._value


class Loss(BaseMetrics):
    """Default Loss Metrics, real number, lower is better.
    """

    names: ClassVar[List[str]] = ["loss"]

    def better_than(self, rhs: Loss):
        return self.value < rhs.value

    def __str__(self) -> str:
        if self.value < 1e-4 or self.value > 1e6:
            return f"{self.value:.6e}"
        return f"{self.value:.6f}"


class Objective(BaseMetrics):
    """Default Objective Metrics, real number, higher is better.
    """

    names: ClassVar[List[str]] = ["objective"]

    def better_than(self, rhs: Loss):
        return self.value > rhs.value

    def __str__(self) -> str:
        if self.value < 1e-4 or self.value > 1e6:
            return f"{self.value:.6e}"
        return f"{self.value:.6f}"


class Ratio(BaseMetrics):
    """Default Postive Ratio Metrics, value in [0, 1], higher is better.
    """

    names: ClassVar[List[str]] = [
        "ratio", "pos_ratio", "accuracy", "precision", "recall"
    ]

    def better_than(self, rhs: Ratio):
        return self.value > rhs.value

    def __str__(self):
        return f"{self.value * 100:.4f} %"


class NegRatio(BaseMetrics):
    """Default Negative Ratio Metrics, value in [0, 1], lower is better.
    """

    names: ClassVar[List[str]] = ["neg_ratio"]

    def better_than(self, rhs: NegRatio):
        return self.value < rhs.value

    def __str__(self):
        return f"{self.value * 100:.4f} %"


BUILTIN_TYPES = Literal["loss", "objective", "ratio", "neg_ratio"]
BUILTIN_METRICS: Dict[str, Type[BaseMetrics]] =\
    {
        name: met
        for met in [Loss, Objective, Ratio, NegRatio]
        for name in met.names
    }
