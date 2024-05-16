import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm
from typing import Optional

from bandit.bernoulli_ts import BernoulliTS
from bandit.logistic_ts import LogisticTS
from bandit.logistic_pgts import LogisticPGTS
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.bandit_base.bandit import BanditBase

K = 10


def get_batch(
    bandit: BanditBase,
    true_theta: np.ndarray,
    features: list[str],
    batch_size: int,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        x = np.random.rand(len(features))
        order = bandit.select_arm(x, top_k=K)
        true_prob = {a: expit(theta @ x) for a, theta in true_theta.items()}
        sorted_true_prob = np.array(sorted(true_prob.values())[::-1])
        maxobj = -np.log(1 - sorted_true_prob[: len(order)]).sum()
        obj = -np.log([1 - true_prob[a] for a in order]).sum()
        # NOTE: 複数クリックを許容する
        clicked = [a for a in order if np.random.binomial(1, true_prob[a])]
        log.append(
            {
                "order": order,
                "clicked": clicked,
                "regret": maxobj - obj,
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


def expand_cascade_data(
    reward_df: pd.DataFrame,
    only_first_click: bool,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """cascading bandit から contextual banditで扱うデータ形式に変換する

    Args:
        reward_df (pd.DataFrame): cascading banditで扱っているデータ形式。"order", "clicked"を含む
        only_first_click (bool): True: 最初のクリックまでを見る。False: 最後のクリックまでを見る
        features (Optional[list[str]]): 特徴名のリスト

    Returns:
        pd.DataFrame: 変換後のデータ形式。"reward", "arm_id"が含まれる。
    """
    records = []
    for i, row in reward_df.iterrows():
        assert isinstance(row["clicked"], list)
        clicked = set(row["clicked"])
        clicked_num = len(clicked)
        for observed in row["order"]:
            reward = int(observed in clicked)
            feature_info = row[features].to_dict() if features is not None else {}
            records.append({"reward": reward, "arm_id": observed} | feature_info)
            if reward:
                clicked_num -= 1
                if only_first_click or clicked_num == 0:
                    # 最初のクリックのみを見る場合 or 最後のクリックなら
                    break
    return pd.DataFrame(records)


batch_size = 10
arm_num = 20
feature_num = 5
intercept = False
arm_ids = [f"arm{i}" for i in range(arm_num)]
features = [f"feat{i}" for i in range(feature_num)]
true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
print(true_theta)
# only_first_click = True
only_first_click = False

report = {}

for bandit in [
    LogisticTS(arm_ids, features, intercept),
    LogisticPGTS(arm_ids, features, intercept, M=10),
    LinTS(arm_ids, features, intercept),
    LinUCB(arm_ids, features, intercept, alpha=1),
    BernoulliTS(arm_ids),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(1000)):
        reward_df = get_batch(bandit, true_theta, features, batch_size)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(expand_cascade_data(reward_df, only_first_click, features))
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.xlabel("Batch Iteration")
plt.ylabel("Cumulative Regret")
plt.title(
    f"Personalized Cascading Bandit: batch_size={batch_size}, item_num={arm_num}, \nonly_first_click={only_first_click}"
)
plt.show()
