from typing import Any, Optional

import numpy as np
import pandas as pd

from polyagamma import random_polyagamma

from .bandit_base.contextual_bandit import ContextualBanditBase


class LogisticPGTS(ContextualBanditBase):
    def prior_parameter(self) -> dict[str, Any]:
        """多次元正規分布の事前分布のパラメーター 平均ベクトルmu, 分散共分散行列Sigma

        Returns:
            dict[str, Any]: 事前分布のパラメータ
        """
        dim = len(self.context_features) + int(self.intercept)
        B = np.eye(dim)
        b = np.zeros(dim)
        theta = np.random.multivariate_normal(b, B)
        Binv = np.linalg.inv(B)
        return {
            "theta": theta,
            "B": B,
            "b": b,
            "Binv": Binv,
        }

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列、context_featuresが必要。学習に関係のあるbandit_idだけに絞っている必要がある。
        """
        # M = 1
        M = 10
        # M = 100
        params = self.parameter["arms"]
        for arm_id in reward_df["arm_id"].unique():
            selector = reward_df["arm_id"] == arm_id
            contexts = self.context_transform(
                reward_df.loc[selector, self.context_features].astype(float).to_numpy()
            )
            if self.intercept:
                contexts = np.concatenate(
                    [contexts, np.ones(contexts.shape[0]).reshape((-1, 1))], axis=1
                )
            rewards = reward_df.loc[selector, "reward"].astype(int).to_numpy()

            B = params[arm_id]["B"]
            b = params[arm_id]["b"]
            Binv = params[arm_id]["Binv"]
            theta = np.random.multivariate_normal(b, B)
            kappa = rewards - 0.5
            # theta = params[arm_id]["theta"]
            for _ in range(M):
                Omega = np.diag([random_polyagamma(1, x @ theta) for x in contexts])
                Vinv = (contexts.T @ Omega) @ contexts + Binv
                V = np.linalg.inv(Vinv)
                m = V @ (contexts.T @ kappa + Binv @ b)
                theta = np.random.multivariate_normal(m, V)
            # params[arm_id]["theta"] = theta
            params[arm_id]["B"] = V
            params[arm_id]["Binv"] = Vinv
            params[arm_id]["b"] = m

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
                x_transform @
                # params[arm_id]["theta"]
                np.random.multivariate_normal(params[arm_id]["b"], params[arm_id]["B"])
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
