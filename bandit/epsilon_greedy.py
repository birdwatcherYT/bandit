from typing import Any, Optional

import numpy as np
import pandas as pd


from .bandit_base.bandit import BanditBase


class EpsilonGreedyBandit(BanditBase):
    def __init__(
        self,
        arm_ids: list[str],
        epsilon: float,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.epsilon = epsilon
        super().__init__(arm_ids, initial_parameter)

    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        return {"sum": 0, "count": 0}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。
        """
        params = self.parameter["arms"]
        diff = reward_df.groupby("arm_id")["reward"].agg(["sum", "size"])
        for arm_id, row in diff.iterrows():
            params[arm_id]["sum"] += row["sum"]
            params[arm_id]["count"] += row["size"]

    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 使わない. Defaults to None.

        Returns:
            str: 腕ID
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arm_ids)
        params = self.parameter["arms"]
        index = np.argmax(
            [
                params[arm_id]["sum"] / params[arm_id]["count"]
                if params[arm_id]["count"] != 0
                else float("inf")
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
