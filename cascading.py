import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit.cascade_ucb import CascadeUCB
from bandit.cascade_klucb import CascadeKLUCB
from bandit.bandit_base.cascading_bandit import CascadingBanditBase


def get_batch(
    bandit: CascadingBanditBase, true_prob: dict[str, float], batch_size: int
) -> pd.DataFrame:
    # 学習データ
    log = []
    sorted_true_prob = np.array(sorted(true_prob.values())[::-1])
    for _ in range(batch_size):
        order = bandit.select_arm()
        clicked = None
        maxobj = 1 - np.prod(1 - sorted_true_prob[: len(order)])
        obj = 1 - np.prod([1 - true_prob[a] for a in order])
        for a in order:
            if clicked is None and np.random.binomial(1, true_prob[a]):
                clicked = a
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
K = 10
item_ids = [f"arm{i}" for i in range(item_num)]
true_prob = {a: np.random.rand() for a in item_ids}
print(true_prob)

report = {}
for bandit in [
    CascadeUCB(item_ids, K),
    CascadeKLUCB(item_ids, K),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(10000)):
        reward_df = get_batch(bandit, true_prob, batch_size)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.xlabel("Batch Iteration")
plt.ylabel("Cumulative Regret")
plt.title(f"Cascading Bandit: batch_size={batch_size}, item_num={item_num}")
plt.show()
