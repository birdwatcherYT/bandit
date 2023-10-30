from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.contextual_bandit import ContextualBanditBase


class LinTS(ContextualBanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        dim = len(self.context_features) + int(self.intercept)
        # NOTE: 分散既知設定なのでsigma=1としている
        std = 1
        A = np.eye(dim)
        b = np.zeros(dim)
        Ainv = np.linalg.inv(A)
        return {
            "A": A,
            "b": b,
            "std": std,
            "mu": Ainv @ b,
            "Sigma": std * std * Ainv,
        }

    def train(self, reward_df: pd.DataFrame) -> None:
        params = self.parameter["arms"]
        for arm_id, arm_df in reward_df.groupby("arm_id"):
            contexts = self.context_transform(
                arm_df[self.context_features].astype(float).to_numpy()
            )
            if self.intercept:
                contexts = np.concatenate(
                    [contexts, np.ones(contexts.shape[0]).reshape((-1, 1))], axis=1
                )
            rewards = arm_df["reward"].astype(float).to_numpy()
            #
            A = params[arm_id]["A"]
            b = params[arm_id]["b"]
            std = params[arm_id]["std"]
            for x in contexts:
                A += np.outer(x, x)
            b += rewards @ contexts
            Ainv = np.linalg.inv(A)
            #
            params[arm_id]["A"] = A
            params[arm_id]["b"] = b
            params[arm_id]["mu"] = Ainv @ b
            params[arm_id]["Sigma"] = std * std * Ainv

    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): contexts. Defaults to None.

        Returns:
            str: 腕ID
        """
        x_transform = self.context_transform(x)
        if self.intercept:
            x_transform = np.concatenate([x_transform, [1]])
        params = self.parameter["arms"]
        index = np.argmax(
            [
                np.dot(
                    x_transform,
                    np.random.multivariate_normal(
                        params[arm_id]["mu"], params[arm_id]["Sigma"]
                    ),
                )
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
