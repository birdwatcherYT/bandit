from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.contextual_cascading_bandit import ContextualCascadingBanditBase


class CascadeLinTS(ContextualCascadingBanditBase):
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
                np.dot(
                    self.item_vectors[arm_id],
                    np.random.multivariate_normal(params["mu"], params["Sigma"]),
                )
                for arm_id in self.arm_ids
            ]
        )[::-1]
        return [self.arm_ids[i] for i in index[: self.K]]
