from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.cascading_bandit import CascadingBanditBase


class CascadeUCB(CascadingBanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {"total_count": 0}

    def arm_parameter(self) -> dict[str, Any]:
        return {"sum": 0, "count": 0}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。どのアイテムがクリックされたかが記載された"clicked"列、そのときの順序が記載された"order"列が必要
        """
        params = self.parameter["arms"]
        for i, row in reward_df.iterrows():
            self.parameter["common"]["total_count"] += 1
            for observed in row["order"]:
                params[observed]["count"] += 1
                if observed == row["clicked"]:
                    params[observed]["sum"] += 1
                    break

    def select_arm(self, x: Optional[np.ndarray] = None) -> list[str]:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 使わない. Defaults to None.

        Returns:
            list[str]: 腕IDのリスト
        """
        params = self.parameter["arms"]
        total_count = self.parameter["common"]["total_count"]
        index = np.argsort(
            [
                params[arm_id]["sum"] / params[arm_id]["count"]
                + np.sqrt(1.5 * np.log(total_count) / params[arm_id]["count"])
                if params[arm_id]["count"] != 0
                else float("inf")
                for arm_id in self.arm_ids
            ]
        )[::-1]
        return [self.arm_ids[i] for i in index[: self.K]]
