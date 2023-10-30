from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.bandit import BanditBase


class NormalTS(BanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        return {"a": 1, "b": 1, "m": 0, "beta": 1}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。
        """
        params = self.parameter["arms"]
        for arm_id, arm_df in reward_df.groupby("arm_id"):
            rewards = arm_df["reward"].astype(float).to_numpy()
            sum_rewards = rewards.sum()
            sum_rewards2 = rewards @ rewards
            n = rewards.shape[0]

            m = params[arm_id]["m"]
            beta = params[arm_id]["beta"]
            a = params[arm_id]["a"]
            b = params[arm_id]["b"]

            params[arm_id]["m"] = (m * beta + sum_rewards) / (beta + n)
            params[arm_id]["beta"] = beta + n
            params[arm_id]["a"] = a + n / 2
            params[arm_id]["b"] = (
                b
                + (
                    sum_rewards2
                    + beta * (m**2)
                    - (beta * m + sum_rewards) ** 2 / (beta + n)
                )
                / 2
            )

    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 使わない. Defaults to None.

        Returns:
            str: 腕ID
        """
        params = self.parameter["arms"]
        index = np.argmax(
            [
                np.random.normal(
                    params[arm_id]["m"],
                    1
                    / (
                        params[arm_id]["beta"]
                        * np.random.gamma(params[arm_id]["a"], 1 / params[arm_id]["b"])
                    ),
                )
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
