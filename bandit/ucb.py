from typing import Any, Optional

import numpy as np
import pandas as pd


from .bandit_base.bandit import BanditBase


class UCB(BanditBase):
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
        params = self.parameter["arms"]
        total_count = np.sum([params[arm_id]["count"] for arm_id in self.arm_ids])
        index = np.argmax(
            [
                params[arm_id]["sum"] / params[arm_id]["count"]
                + np.sqrt(2 * np.log(total_count) / params[arm_id]["count"])
                if params[arm_id]["count"] != 0
                else float("inf")
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
