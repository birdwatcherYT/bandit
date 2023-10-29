import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit.ts_binary import TSBinaryBandit
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.logistic_ts import LogisticTS
from bandit.logistic_pgts import LogisticPGTS
from bandit.bandit_base.bandit import BanditBase


def get_batch(
    bandit: BanditBase,
    true_prob: list[dict[str, float]],
    centroid: np.ndarray,
    features: list[str],
    batch_size,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        x = np.random.rand(len(features))
        arm_id = bandit.select_arm(x)
        cluster = np.argmin(np.linalg.norm(centroid - x, axis=1))
        argmax = np.argmax(list(true_prob[cluster].values()))
        maxprob = list(true_prob[cluster].values())[argmax]
        log.append(
            {
                "best_arm_id": bandit.arm_ids[argmax],
                "arm_id": arm_id,
                "reward": np.random.binomial(1, true_prob[cluster][arm_id]),
                "regret": maxprob - true_prob[cluster][arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


batch_size = 100
arm_num = 5
feature_num = 10
intercept = True
# intercept = False

onehot_maxprob = 0.5
onehot_minprob = 0.2
onehot = True
# onehot = False

cluster_num = arm_num if onehot else 4
arm_ids = [f"arm{i}" for i in range(arm_num)]
features = [f"feat{i}" for i in range(feature_num)]
centroid = np.array([np.random.rand(feature_num) for i in range(cluster_num)])
true_prob = [
    {
        a: (onehot_maxprob if (k == i) else onehot_minprob)
        if onehot
        else np.random.rand()
        for k, a in enumerate(arm_ids)
    }
    for i in range(cluster_num)
]
print(true_prob)

report = {}
for bandit in [
    LogisticTS(arm_ids, features, intercept),
    LogisticPGTS(arm_ids, features, intercept, M=10),
    LinTS(arm_ids, features, intercept),
    LinUCB(arm_ids, features, intercept, alpha=1),
    TSBinaryBandit(arm_ids),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(100)):
        reward_df = get_batch(bandit, true_prob, centroid, features, batch_size)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log

    reward_df = get_batch(bandit, true_prob, centroid, features, 1000)
    print(reward_df.groupby(["best_arm_id","arm_id"]).size())
pd.DataFrame(report).plot()
plt.xlabel("Batch Iteration")
plt.ylabel("Cumulative Regret")
plt.title(f"Contextual Binary Reward Bandit for Clustering data: batch_size={batch_size}, arm_num={arm_num}")
plt.show()
