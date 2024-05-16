from typing import Any, Optional

import numpy as np
import pandas as pd

from polyagamma import random_polyagamma

from .bandit_base.contextual_bandit import ContextualBanditBase


class LogisticPGTS(ContextualBanditBase):
    def __init__(
        self,
        arm_ids: list[str],
        context_features: list[str],
        intercept: bool = True,
        M: int = 1,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.M = M
        super().__init__(arm_ids, context_features, intercept, initial_parameter)

    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        dim = len(self.context_features) + int(self.intercept)
        B = np.eye(dim)
        b = np.zeros(dim)
        # theta = np.random.multivariate_normal(b, B)
        Binv = np.linalg.inv(B)
        return {
            # "theta": theta,
            "B": B,
            "b": b,
            "Binv": Binv,
        }

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列、context_featuresが必要。
        """
        params = self.parameter["arms"]
        for arm_id, arm_df in reward_df.groupby("arm_id"):
            contexts = self.context_transform(
                arm_df[self.context_features].astype(float).to_numpy()
            )
            if self.intercept:
                contexts = np.concatenate(
                    [contexts, np.ones(contexts.shape[0]).reshape((-1, 1))], axis=1
                )
            rewards = arm_df["reward"].astype(int).to_numpy()

            B = params[arm_id]["B"]
            b = params[arm_id]["b"]
            Binv = params[arm_id]["Binv"]
            theta = np.random.multivariate_normal(b, B)
            kappa = rewards - 0.5
            # theta = params[arm_id]["theta"]
            for _ in range(self.M):
                Omega = np.diag([random_polyagamma(1, x @ theta) for x in contexts])
                Vinv = (contexts.T @ Omega) @ contexts + Binv
                V = np.linalg.inv(Vinv)
                m = V @ (contexts.T @ kappa + Binv @ b)
                theta = np.random.multivariate_normal(m, V)
            # params[arm_id]["theta"] = theta
            params[arm_id]["B"] = V
            params[arm_id]["Binv"] = Vinv
            params[arm_id]["b"] = m

    def __get_score__(self, x: Optional[np.ndarray] = None) -> list[float]:
        x_transform = self.context_transform(x)
        if self.intercept:
            x_transform = np.concatenate([x_transform, [1]])
        params = self.parameter["arms"]
        return [
            x_transform @
            # params[arm_id]["theta"]
            np.random.multivariate_normal(params[arm_id]["b"], params[arm_id]["B"])
            for arm_id in self.arm_ids
        ]
