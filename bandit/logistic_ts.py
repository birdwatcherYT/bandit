from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import expit
import warnings
from typing import Callable, Optional

from .bandit_base.contextual_bandit import ContextualBanditBase


def gradient_descent(
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 10000,
    eps: float = 1e-12,
    decay: float = 0.5,
    hess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """勾配法（最急降下法、ニュートン法）

    Args:
        obj (Callable[[np.ndarray], float]): 目的関数
        grad (Callable[[np.ndarray], np.ndarray]): 勾配
        x0 (np.ndarray): 初期解
        max_iter (int, optional): 最大ループ数. Defaults to 10000.
        eps (float, optional): 精度. Defaults to 1e-12.
        decay (float, optional): ラインサーチ時の減少係数. Defaults to 0.1.
        hess (Optional[Callable[[np.ndarray], np.ndarray]], optional): ヘシアン. ニュートン法になる. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    x = np.copy(x0)
    minimum = obj(x0)
    for i in range(max_iter):
        d = -grad(x)
        if hess is not None:
            d = np.dot(np.linalg.inv(hess(x)), d)
        # ラインサーチ
        alpha = 1
        o = obj(x + alpha * d)
        while alpha > eps and o >= minimum:
            alpha *= decay
            o = obj(x + alpha * d)
        minimum = o
        if alpha * np.mean(np.abs(d)) <= eps:
            return x
        x += alpha * d
    warnings.warn("not convergence")
    return x


class LogisticTS(ContextualBanditBase):
    def common_parameter(self) -> dict[str, Any]:
        return {}

    def arm_parameter(self) -> dict[str, Any]:
        dim = len(self.context_features) + int(self.intercept)
        # NOTE: 行列は先頭大文字
        return {"mu": np.zeros(dim), "Sigma": np.eye(dim)}

    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。"arm_id"と"reward"列、context_featuresが必要。
        """
        params = self.parameter["arms"]
        for arm_id, arm_df in reward_df.groupby("arm_id"):
            contexts = self.context_transform(
                arm_df[self.context_features].astype(float).to_numpy()
            )
            if self.intercept:
                contexts = np.concatenate(
                    [contexts, np.ones(contexts.shape[0]).reshape((-1, 1))], axis=1
                )
            rewards = arm_df["reward"].astype(int).to_numpy()
            mu0 = params[arm_id]["mu"]
            Sigma0 = params[arm_id]["Sigma"]
            # MAP推定値
            theta_map = gradient_descent(
                obj=partial(
                    LogisticTS.objective,
                    contexts=contexts,
                    rewards=rewards,
                    mu0=mu0,
                    Sigma0=Sigma0,
                ),
                grad=partial(
                    LogisticTS.gradient,
                    contexts=contexts,
                    rewards=rewards,
                    mu0=mu0,
                    Sigma0=Sigma0,
                ),
                x0=mu0,
                hess=partial(
                    LogisticTS.hessian,
                    contexts=contexts,
                    rewards=rewards,
                    mu0=mu0,
                    Sigma0=Sigma0,
                ),
            )
            hess_map = LogisticTS.hessian(
                theta_map,
                contexts=contexts,
                rewards=rewards,
                mu0=mu0,
                Sigma0=Sigma0,
            )
            params[arm_id]["mu"] = theta_map
            params[arm_id]["Sigma"] = np.linalg.inv(hess_map)

    def select_arm(self, x: Optional[np.ndarray] = None) -> str:
        """腕の選択

        Args:
            x (Optional[np.ndarray], optional): contexts. Defaults to None.

        Returns:
            str: 腕ID
        """
        x_transform = self.context_transform(x)
        if self.intercept:
            x_transform = np.concatenate([x_transform, [1]])
        params = self.parameter["arms"]
        index = np.argmax(
            [
                np.dot(
                    x_transform,
                    np.random.multivariate_normal(
                        params[arm_id]["mu"], params[arm_id]["Sigma"]
                    ),
                )
                for arm_id in self.arm_ids
            ]
        )
        return self.arm_ids[index]

    @classmethod
    def objective(
        cls,
        theta: np.ndarray,
        contexts: np.ndarray,
        rewards: np.ndarray,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
    ) -> float:
        Sigma0_inv = np.linalg.inv(Sigma0)
        diff = theta - mu0
        contexts_theta = contexts @ theta
        y = 2 * rewards - 1

        return (
            0.5 * (diff @ Sigma0_inv @ diff) - np.log(expit(y * contexts_theta)).sum()
        )

    @classmethod
    def gradient(
        cls,
        theta: np.ndarray,
        contexts: np.ndarray,
        rewards: np.ndarray,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
    ) -> np.ndarray:
        Sigma0_inv = np.linalg.inv(Sigma0)
        diff = theta - mu0
        contexts_theta = contexts @ theta
        y = 2 * rewards - 1

        return Sigma0_inv @ diff - (y * expit(-y * contexts_theta)) @ contexts

    @classmethod
    def hessian(
        cls,
        theta: np.ndarray,
        contexts: np.ndarray,
        rewards: np.ndarray,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
    ) -> np.ndarray:
        Sigma0_inv = np.linalg.inv(Sigma0)
        contexts_theta = contexts @ theta
        y = 2 * rewards - 1
        coeff = expit(y * contexts_theta) * expit(-y * contexts_theta)

        hess = Sigma0_inv
        for x, ci in zip(contexts, coeff):
            hess += np.outer(x, x) * ci
        return hess
