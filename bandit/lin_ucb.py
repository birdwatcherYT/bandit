from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.contextual_bandit import ContextualBanditBase


class LinUCB(ContextualBanditBase):
    def prior_parameter(self) -> dict[str, Any]:
        """初期パラメーター

        Returns:
            dict[str, Any]: 初期パラメータ
        """
        dim = len(self.context_features) + int(self.intercept)
        A = np.eye(dim)
        b = np.zeros(dim)
        Ainv = np.linalg.inv(A)
        return {
            "A": A,
            "b": b,
            "theta": Ainv @ b,
            "Ainv": Ainv,
        }

    def train(self, reward_df: pd.DataFrame) -> None:
        params = self.parameter["arms"]
        for arm_id in reward_df["arm_id"].unique():
            selector = reward_df["arm_id"] == arm_id
            contexts = self.context_transform(
                reward_df.loc[selector, self.context_features].astype(float).to_numpy()
            )
            if self.intercept:
                contexts = np.concatenate(
                    [contexts, np.ones(contexts.shape[0]).reshape((-1, 1))], axis=1
                )
            rewards = reward_df.loc[selector, "reward"].astype(float).to_numpy()
            #
            A = np.array(params[arm_id]["A"])
            b = np.array(params[arm_id]["b"])
            for x in contexts:
                A += np.outer(x, x)
            b += rewards @ contexts
            Ainv = np.linalg.inv(A)
            #
            params[arm_id]["A"] = A
            params[arm_id]["b"] = b
            params[arm_id]["theta"] = Ainv @ b
            params[arm_id]["Ainv"] = Ainv

    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): contexts. Defaults to None.

        Returns:
            str: 腕ID
        """
        alpha = 1
        x_transform = self.context_transform(x)
        if self.intercept:
            x_transform = np.concatenate([x_transform, [1]])
        params = self.parameter["arms"]
        index = np.argmax(
            [
                (x_transform @ params[arm_id]["theta"])
                + alpha * np.sqrt(x_transform @ (params[arm_id]["Ainv"] @ x_transform))
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]
