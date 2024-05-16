import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm

from bandit.ucb import UCB
from bandit.klucb import KLUCB
from bandit.cascade_lin_ts import CascadeLinTS
from bandit.cascade_lin_ucb import CascadeLinUCB
from bandit.bandit_base.bandit import BanditBase
from bandit.bandit_base.contextual_cascading_bandit import ContextualCascadingBanditBase
from bandit.tools import expand_cascade_data

K = 10


def get_batch(
    bandit: BanditBase,
    true_theta: np.ndarray,
    item_vectors: dict[str, float],
    batch_size: int,
) -> pd.DataFrame:
    # 学習データ
    log = []
    sorted_true_prob = np.array(
        sorted([expit(x @ true_theta) for x in item_vectors.values()])[::-1]
    )
    for _ in range(batch_size):
        order = bandit.select_arm(top_k=K)
        # maxobj = 1 - np.prod(1 - sorted_true_prob[: len(order)])
        # obj = 1 - np.prod([1 - expit(true_theta @ item_vectors[a]) for a in order])
        maxobj = -np.log(1 - sorted_true_prob[: len(order)]).sum()
        obj = -np.log([1 - expit(true_theta @ item_vectors[a]) for a in order]).sum()

        clicked = []
        for a in order:
            prob = expit(true_theta @ item_vectors[a])
            if np.random.binomial(1, prob):
                # NOTE: 論文では初回クリックしたらそれ以降は見ないという仮定だった
                clicked = [a]
                break
        log.append(
            {
                "order": order,
                "clicked": clicked,
                "regret": maxobj - obj,
            }
        )
    return pd.DataFrame(log)


batch_size = 1
item_num = 20
feature_num = 5
item_ids = [f"arm{i}" for i in range(item_num)]
true_theta = np.random.normal(size=feature_num)
print(true_theta)
item_vectors = {f"arm{i}": np.random.normal(size=feature_num) for i in range(item_num)}

report = {}
for bandit in [
    CascadeLinTS(item_ids, item_vectors),
    CascadeLinUCB(item_ids, item_vectors, alpha=1),
    UCB(item_ids),
    KLUCB(item_ids),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(10000)):
        reward_df = get_batch(bandit, true_theta, item_vectors, batch_size)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        if isinstance(bandit, ContextualCascadingBanditBase):
            bandit.train(reward_df)
        else:
            bandit.train(expand_cascade_data(reward_df, True))
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.xlabel("Batch Iteration")
plt.ylabel("Cumulative Regret")
plt.title(f"Contextual Cascading Bandit: batch_size={batch_size}, item_num={item_num}")
plt.show()
