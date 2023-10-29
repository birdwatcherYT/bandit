from abc import ABCMeta
from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit import BanditBase


class ContextualCascadingBanditBase(BanditBase, metaclass=ABCMeta):
    def __init__(
        self,
        arm_ids: list[str],
        K: int,
        item_vectors: dict[str, np.ndarray],
        intercept: bool = True,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.item_vectors = (
            {k: np.concatenate([v, [1]]) for k, v in item_vectors.items()}
            if intercept
            else item_vectors
        )
        self.K = K
        super().__init__(arm_ids, initial_parameter)
