from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class BanditBase(metaclass=ABCMeta):
    def __init__(
        self, arm_ids: list[str], initial_parameter: Optional[dict[str, Any]] = None
    ) -> None:
        self.arm_ids = arm_ids
        self.parameter = {
            "common": self.common_parameter(),
            "arms": {arm_id: self.arm_parameter() for arm_id in arm_ids},
        }
        if initial_parameter is not None:
            self.parameter.update(initial_parameter)

    @abstractmethod
    def common_parameter(self) -> dict[str, Any]:
        """共通パラメータ

        Returns:
            dict[str, Any]: 共通パラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def arm_parameter(self) -> dict[str, Any]:
        """腕パラメータ

        Returns:
            dict[str, Any]: 腕パラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列が必要。
        """
        raise NotImplementedError()

    @abstractmethod
    def __get_score__(self, x: Optional[np.ndarray] = None) -> list[float]:
        """全腕に対応するスコアを取得

        Args:
            x (Optional[np.ndarray], optional): 特徴ベクトル. Defaults to None.

        Returns:
            list[float]: 腕のスコア
        """
        raise NotImplementedError()

    def select_arm(
        self, x: Optional[np.ndarray] = None, top_k: Optional[int] = None
    ) -> Union[str, list[str]]:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): 特徴ベクトル. Defaults to None.
            top_k (Optional[int], optional): . Defaults to None.

        Returns:
            Union[str, list[str]]: 腕ID / 評価値が高い順に腕IDのリストを返す
        """
        score = self.__get_score__(x)
        if top_k is None:
            # 上位1件のみを返す
            return self.arm_ids[np.argmax(score)]
        # top_k個を返す
        # NOTE: 降順にするためマイナスをつける
        return [self.arm_ids[i] for i in np.argsort(-np.array(score))[:top_k]]
