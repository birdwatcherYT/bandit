from typing import Any, Optional

import numpy as np
import pandas as pd

from .bandit_base.contextual_bandit import ContextualBanditBase


class LinUCBHybrid(ContextualBanditBase):
    def __init__(
        self,
        arm_ids: list[str],
        context_features: list[str],
        intercept: bool = True,
        alpha: float = 1,
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.alpha = alpha
        super().__init__(arm_ids, context_features, intercept, initial_parameter)

    def common_parameter(self) -> dict[str, Any]:
        dim = len(self.context_features) + int(self.intercept)
        A0 = np.eye(dim)
        b0 = np.zeros(dim)
        A0inv = np.linalg.inv(A0)
        beta = A0inv @ b0
        return {
            "A0": A0,
            "b0": b0,
            "A0inv": A0inv,
            "beta": beta,
        }

    def arm_parameter(self) -> dict[str, Any]:
        dim = len(self.context_features) + int(self.intercept)
        A = np.eye(dim)
        b = np.zeros(dim)
        B = np.zeros((dim, dim))
        Ainv = np.linalg.inv(A)
        beta = np.zeros(dim)
        theta = Ainv @ (b - B @ beta)
        return {
            "A": A,
            "b": b,
            "B": B,
            "Ainv": Ainv,
            "theta": theta,
        }

    def train(self, reward_df: pd.DataFrame) -> None:
        params = self.parameter["arms"]
        A0 = self.parameter["common"]["A0"]
        b0 = self.parameter["common"]["b0"]
        for arm_id, arm_df in reward_df.groupby("arm_id"):
            contexts = self.context_transform(
                arm_df[self.context_features].astype(float).to_numpy()
            )
            rewards = arm_df["reward"].astype(float).to_numpy()

            A = params[arm_id]["A"]
            b = params[arm_id]["b"]
            B = params[arm_id]["B"]
            Ainv = params[arm_id]["Ainv"]
            for x, r in zip(contexts, rewards):
                #
                A0 += B.T @ Ainv @ B
                b0 += B.T @ Ainv @ b
                #
                A += np.outer(x, x)
                B += np.outer(x, x)
                b += r * x
                Ainv = np.linalg.inv(A)
                #
                A0 += np.outer(x, x) - B.T @ Ainv @ B
                b0 += r * x - B.T @ Ainv @ b

            params[arm_id]["A"] = A
            params[arm_id]["b"] = b
            params[arm_id]["B"] = B
            params[arm_id]["Ainv"] = Ainv
        self.parameter["common"]["A0"] = A0
        self.parameter["common"]["b0"] = b0
        A0inv = np.linalg.inv(A0)
        beta = A0inv @ b0
        self.parameter["common"]["A0inv"] = A0inv
        self.parameter["common"]["beta"] = beta
        for arm_id in self.arm_ids:
            b = params[arm_id]["b"]
            B = params[arm_id]["B"]
            Ainv = params[arm_id]["Ainv"]
            params[arm_id]["theta"] = Ainv @ (b - B @ beta)

    def __get_score__(self, x: Optional[np.ndarray] = None) -> list[float]:
        x_transform = self.context_transform(x)
        if self.intercept:
            x_transform = np.concatenate([x_transform, [1]])
        params = self.parameter["arms"]
        beta = self.parameter["common"]["beta"]
        A0inv = self.parameter["common"]["A0inv"]
        return [
            (x_transform @ beta)
            + (x_transform @ params[arm_id]["theta"])
            + self.alpha
            * np.sqrt(
                x_transform @ A0inv @ x_transform
                - 2
                * (
                    x_transform
                    @ A0inv
                    @ params[arm_id]["B"].T
                    @ params[arm_id]["Ainv"]
                    @ x_transform
                )
                + x_transform @ params[arm_id]["Ainv"] @ x_transform
                + (
                    x_transform
                    @ params[arm_id]["Ainv"]
                    @ params[arm_id]["B"]
                    @ A0inv
                    @ params[arm_id]["B"].T
                    @ params[arm_id]["Ainv"]
                    @ x_transform
                )
            )
            for arm_id in self.arm_ids
        ]
