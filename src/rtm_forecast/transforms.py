"""Variance-stabilising transform for prices."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AsinhTransform:
    """arcsinh(x / scale). Close to a logarithm when x is much larger than the scale, close to a
    straight line near zero, and defined for zero and negative values."""

    scale: float

    def forward(self, values: np.ndarray) -> np.ndarray:
        return np.arcsinh(values / self.scale)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.sinh(values) * self.scale


def choose_scale(training_values: np.ndarray, candidates: Sequence[float]) -> float:
    """The candidate scale whose transformed training values are closest to symmetric.

    Only training values may be passed in, so the choice does not look at the test period.
    """
    values = pd.Series(training_values).dropna()
    skews = {
        scale: abs(cast(float, pd.Series(np.arcsinh(values / scale)).skew()))
        for scale in candidates
    }
    return min(skews, key=lambda scale: skews[scale])
