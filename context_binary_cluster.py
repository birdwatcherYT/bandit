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
    truth_prob: list[dict[str, float]],
    centroid: np.ndarray,
    features: list[str],
    batch_size: int = 100,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        x = np.random.rand(len(features))
        arm_id = bandit.select_arm(x)
        cluster = np.argmin(np.linalg.norm(centroid - x, axis=1))
        maxprob = max(truth_prob[cluster].values())
        log.append(
            {
                "arm_id": arm_id,
                "reward": np.random.binomial(1, truth_prob[cluster][arm_id]),
                "regret": maxprob - truth_prob[cluster][arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


arm_num = 5
feature_num = 10
intercept = True
# intercept = False

onehot_maxprob = 0.7
onehot_minprob = 0.3
# onehot = True
onehot=False

cluster_num = arm_num if onehot else 4
arm_ids = [f"arm{i}" for i in range(arm_num)]
features = [f"feat{i}" for i in range(feature_num)]
centroid = np.array([np.random.rand(feature_num) for i in range(cluster_num)])
truth_prob = [
    {
        a: (onehot_maxprob if (k == i) else onehot_minprob)
        if onehot
        else np.random.rand()
        for k, a in enumerate(arm_ids)
    }
    for i in range(cluster_num)
]
print(truth_prob)

report = {}
for bandit in [
    LogisticTS(arm_ids, features, intercept),
    LogisticPGTS(arm_ids, features, intercept),
    LinTS(arm_ids, features, intercept),
    LinUCB(arm_ids, features, intercept),
    TSBinaryBandit(arm_ids),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(100)):
        reward_df = get_batch(bandit, truth_prob, centroid, features)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.show()