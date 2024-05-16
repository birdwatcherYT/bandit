import pandas as pd

from .klucb import KLUCB


class CascadeKLUCB(KLUCB):
    def train(self, reward_df: pd.DataFrame) -> None:
        """パラメータの更新

        Args:
            reward_df (pd.DataFrame): 報酬のログ。どのアイテムがクリックされたかが記載された"clicked"列、そのときの順序が記載された"order"列が必要
        """
        params = self.parameter["arms"]
        for i, row in reward_df.iterrows():
            # 提示順で最初にクリックされたアイテム
            clicked = row["clicked"][0] if len(row["clicked"]) != 0 else None
            self.parameter["common"]["total_count"] += 1
            for observed in row["order"]:
                params[observed]["count"] += 1
                if observed == clicked:
                    params[observed]["sum"] += 1
                    break
        self.optimize()
