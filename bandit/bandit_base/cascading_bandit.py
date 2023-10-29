from abc import ABCMeta
from typing import Any, Optional

from .bandit import BanditBase


class CascadingBanditBase(BanditBase, metaclass=ABCMeta):
    def __init__(
        self,
        arm_ids: list[str],  # アイテムID
        K: int,  # 何個提示するか
        initial_parameter: Optional[dict[str, Any]] = None,
    ) -> None:
        self.K = K
        super().__init__(arm_ids, initial_parameter)
