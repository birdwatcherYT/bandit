from abc import ABCMeta
from typing import Any, Optional

import numpy as np

from .bandit import BanditBase


class ContextualBanditBase(BanditBase, metaclass=ABCMeta):
    def __init__(
        self,
        arm_ids: list[str],
        context_features: list[str],
        intercept: bool = True,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.context_features = context_features
        self.intercept = intercept
        super().__init__(arm_ids, initial_parameter)

    def set_context_mean_std(self, mean: np.ndarray, std: np.ndarray):
        """contextの標準化に使うパラメータをセット

        Args:
            mean (np.ndarray): 平均
            std (np.ndarray): 標準偏差
        """
        self.parameter["scaler"] = {"mean": mean, "std": std}

    def get_context_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """contextの標準化に使うパラメータを返す

        Returns:
            tuple[np.ndarray, np.ndarray]: (平均, 標準偏差)
        """
        if "scaler" in self.parameter:
            return (self.parameter["scaler"]["mean"], self.parameter["scaler"]["std"])
        return np.zeros(len(self.context_features)), np.ones(len(self.context_features))

    def context_transform(self, x: np.ndarray) -> np.ndarray:
        """標準化、欠損値埋め

        Args:
            x (np.ndarray): コンテキストベクトル(またはユーザー数xコンテキストベクトル長の行列)

        Returns:
            np.ndarray: 標準化されたベクトル
        """
        context_mean, context_std = self.get_context_mean_std()
        # NOTE: 欠損値を平均で埋めてから標準化することを考えると、結局ゼロで埋めることに相当する
        return np.nan_to_num((x - context_mean) / context_std, nan=0)
