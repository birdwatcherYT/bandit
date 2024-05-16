from typing import Optional

import numpy as np
import warnings
from typing import Callable, Optional


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
