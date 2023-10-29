from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.bandit import BanditBase


class BernoulliTS(BanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        """beta分布のパラメータ Beta(theta|alpha, beta)=theta^{alpha-1}(1-theta)^{beta-1}/B(alpha, beta)

        Returns:
            dict[str, Any]: 事前分布のパラメータ
        """
        return {"alpha": 1, "beta": 1}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。
        """
        params = self.parameter["arms"]
        diff = reward_df.groupby("arm_id")["reward"].agg(["sum", "size"])
        for arm_id, row in diff.iterrows():
            # 報酬1の個数
            params[arm_id]["alpha"] += row["sum"]
            # 報酬0の個数
            params[arm_id]["beta"] += row["size"] - row["sum"]

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
                np.random.beta(params[arm_id]["alpha"], params[arm_id]["beta"])
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
