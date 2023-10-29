from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BanditBase(metaclass=ABCMeta):
    def __init__(
        self, arm_ids: list[str], initial_parameter: Optional[dict[str, Any]] = None
    ) -> None:
        self.arm_ids = arm_ids
        # 腕ごとに事前分布のパラメータを設定
        self.parameter = {
            "arms": {arm_id: self.prior_parameter() for arm_id in arm_ids}
        }
        if initial_parameter is not None:
            self.parameter.update(initial_parameter)

    @abstractmethod
    def prior_parameter(self) -> dict[str, Any]:
        """事前分布のパラメータ

        Returns:
            dict[str, Any]: 事前分布のパラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。学習に関係のあるbandit_idだけに絞っている必要がある。
        """
        raise NotImplementedError()

    @abstractmethod
    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 特徴ベクトル. Defaults to None.

        Returns:
            str: 腕ID
        """
        raise NotImplementedError()
