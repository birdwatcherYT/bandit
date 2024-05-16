from typing import Any, Optional

import numpy as np
import pandas as pd


from .bandit_base.bandit import BanditBase


class UCB(BanditBase):
    def __init__(
        self,
        arm_ids: list[str],
        alpha: float = 1,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.alpha = alpha
        super().__init__(arm_ids, initial_parameter)

    def common_parameter(self) -> dict[str, Any]:
        return {"total_count": 0}

    def arm_parameter(self) -> dict[str, Any]:
        return {"sum": 0, "count": 0}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。
        """
        params = self.parameter["arms"]
        self.parameter["common"]["total_count"] += reward_df.shape[0]
        diff = reward_df.groupby("arm_id")["reward"].agg(["sum", "size"])
        for arm_id, row in diff.iterrows():
            params[arm_id]["sum"] += row["sum"]
            params[arm_id]["count"] += row["size"]

    def __get_score__(self, x: Optional[np.ndarray] = None) -> list[float]:
        params = self.parameter["arms"]
        total_count = self.parameter["common"]["total_count"]
        return [
            (
                params[arm_id]["sum"] / params[arm_id]["count"]
                + np.sqrt(self.alpha * np.log(total_count) / params[arm_id]["count"])
                if params[arm_id]["count"] != 0
                else float("inf")
            )
            for arm_id in self.arm_ids
        ]
