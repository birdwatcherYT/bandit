from abc import ABCMeta
from typing import Any, Optional

import numpy as np

from .bandit import BanditBase


class ContextualCascadingBanditBase(BanditBase, metaclass=ABCMeta):
    def __init__(
        self,
        arm_ids: list[str],
        item_vectors: dict[str, np.ndarray],
        intercept: bool = True,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.item_vectors = (
            {k: np.concatenate([v, [1]]) for k, v in item_vectors.items()}
            if intercept
            else item_vectors
        )
        super().__init__(arm_ids, initial_parameter)
