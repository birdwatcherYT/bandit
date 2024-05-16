from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
from typing import Optional

from .bandit_base.bandit import BanditBase
from .tools import newton_method


class KLUCB(BanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {"total_count": 0}

    def arm_parameter(self) -> dict[str, Any]:
        return {"sum": 0, "count": 0, "klucb": 0}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。どのアイテムがクリックされたかが記載された"clicked"列、そのときの順序が記載された"order"列が必要
        """
        params = self.parameter["arms"]
        self.parameter["common"]["total_count"] += reward_df.shape[0]
        diff = reward_df.groupby("arm_id")["reward"].agg(["sum", "size"])
        for arm_id, row in diff.iterrows():
            params[arm_id]["sum"] += row["sum"]
            params[arm_id]["count"] += row["size"]

        self.optimize()

    def optimize(self):
        params = self.parameter["arms"]
        total_count = self.parameter["common"]["total_count"]
        for arm_id in self.arm_ids:
            if params[arm_id]["count"] == 0 or params[arm_id]["sum"] == 0:
                params[arm_id]["klucb"] = 1
                continue
            p = params[arm_id]["sum"] / params[arm_id]["count"]
            params[arm_id]["klucb"] = newton_method(
                obj=partial(
                    KLUCB.objective,
                    p=p,
                    count_a=params[arm_id]["count"],
                    total_count=total_count,
                ),
                grad=partial(
                    KLUCB.gradient,
                    p=p,
                    count_a=params[arm_id]["count"],
                    total_count=total_count,
                ),
                x0=1 - 1e-6,
                x_lower=p,
                x_upper=1,
            )

    def __get_score__(self, x: Optional[np.ndarray] = None) -> list[float]:
        params = self.parameter["arms"]
        return [params[arm_id]["klucb"] for arm_id in self.arm_ids]

    @classmethod
    def objective(
        cls,
        q: float,
        p: float,
        count_a: float,
        total_count: float,
    ) -> float:
        return count_a * (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) - (
            np.log(total_count) + 3 * np.log(np.log(total_count))
        )

    @classmethod
    def gradient(
        cls,
        q: float,
        p: float,
        count_a: float,
        total_count: float,
    ) -> float:
        return count_a * (-p * 1 / q + (1 - p) / (1 - q))
