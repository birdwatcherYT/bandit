from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import warnings
from typing import Callable, Optional

from .bandit_base.cascading_bandit import CascadingBanditBase


def newton_method(
    obj: Callable[[float], float],
    grad: Callable[[float], float],
    x0: float,
    x_lower: float,
    x_upper: float,
    max_iter: int = 10000,
    eps: float = 1e-12,
) -> float:
    """ニュートン法(obj(x)=0を求める)

    Args:
        obj (Callable[[float], float]): 目的関数
        grad (Callable[[float], float]): 勾配
        x0 (float): 初期解
        x_lower (float): この値を下回ったら終了
        x_upper (float): この値を上回ったら終了
        max_iter (int, optional): 最大ループ数. Defaults to 10000.
        eps (float, optional): 精度. Defaults to 1e-12.

    Returns:
        float: _description_
    """
    x = np.copy(x0)
    for i in range(max_iter):
        if x <= x_lower:
            return x_lower
        if x >= x_upper:
            return x_upper
        d = -obj(x) / grad(x)
        if np.mean(np.abs(d)) <= eps:
            return x
        x += d
    warnings.warn("not convergence")
    return x


class CascadingKLUCB(CascadingBanditBase):
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
        for i, row in reward_df.iterrows():
            self.parameter["common"]["total_count"] += 1
            for observed in row["order"]:
                params[observed]["count"] += 1
                if observed == row["clicked"]:
                    params[observed]["sum"] += 1
                    break

        total_count = self.parameter["common"]["total_count"]
        for arm_id in self.arm_ids:
            if params[arm_id]["count"] == 0 or params[arm_id]["sum"] == 0:
                params[arm_id]["klucb"] = 1
                continue
            p = params[arm_id]["sum"] / params[arm_id]["count"]
            params[arm_id]["klucb"] = newton_method(
                obj=partial(
                    CascadingKLUCB.objective,
                    p=p,
                    count_a=params[arm_id]["count"],
                    total_count=total_count,
                ),
                grad=partial(
                    CascadingKLUCB.gradient,
                    p=p,
                    count_a=params[arm_id]["count"],
                    total_count=total_count,
                ),
                x0=1 - 1e-6,
                x_lower=p,
                x_upper=1,
            )

    def select_arm(self, x: Optional[np.ndarray] = None) -> list[str]:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 使わない. Defaults to None.

        Returns:
            list[str]: 腕IDのリスト
        """
        params = self.parameter["arms"]
        index = np.argsort([params[arm_id]["klucb"] for arm_id in self.arm_ids])[::-1]
        return [self.arm_ids[i] for i in index[: self.K]]

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
