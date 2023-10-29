from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.contextual_cascading_bandit import ContextualCascadingBanditBase


class CascadingLinUCB(ContextualCascadingBanditBase):
    def __init__(
        self,
        arm_ids: list[str],
        K: int,
        item_vectors: dict[str, np.ndarray],
        alpha: int = 1,
        intercept: bool = True,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.alpha = alpha
        super().__init__(arm_ids, K, item_vectors, intercept, initial_parameter)

    def common_parameter(self) -> dict[str, Any]:
        dim = len(list(self.item_vectors.values())[0])
        # NOTE: 分散既知設定なのでsigma=1としている
        std = 1
        A = np.eye(dim)
        b = np.zeros(dim)
        Ainv = np.linalg.inv(A)
        return {
            "A": A,
            "b": b,
            "std": std,
            "mu": Ainv @ b / (std**2),
            "Sigma": Ainv,
        }

    def arm_parameter(self) -> dict[str, Any]:
        return {}

    def train(self, reward_df: pd.DataFrame) -> None:
        params = self.parameter["common"]
        for i, row in reward_df.iterrows():
            for observed in row["order"]:
                x = self.item_vectors[observed]
                params["A"] += np.outer(x, x) / (params["std"] ** 2)
                if observed == row["clicked"]:
                    params["b"] += x
                    break

        Ainv = np.linalg.inv(params["A"])
        params["Sigma"] = Ainv
        params["mu"] = Ainv @ params["b"] / (params["std"] ** 2)

    def select_arm(self, x: Optional[np.ndarray] = None) -> list[str]:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): contexts. Defaults to None.

        Returns:
            list[str]: 腕IDのリスト
        """
        params = self.parameter["common"]
        index = np.argsort(
            [
                min(
                    self.item_vectors[a] @ params["mu"]
                    + self.alpha
                    * np.sqrt(
                        self.item_vectors[a] @ params["Sigma"] @ self.item_vectors[a]
                    ),
                    1,
                )
                for a in self.arm_ids
            ]
        )[::-1]
        return [self.arm_ids[i] for i in index[: self.K]]
