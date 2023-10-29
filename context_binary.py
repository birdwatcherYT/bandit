import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import expit

from bandit.ts_binary import TSBinaryBandit
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.logistic_ts import LogisticTS
from bandit.logistic_pgts import LogisticPGTS
from bandit.bandit_base.bandit import BanditBase


def get_batch(
    bandit: BanditBase,
    true_theta: dict[str, np.ndarray],
    features: list[str],
    batch_size: int = 100,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        x = np.random.rand(len(features))
        arm_id = bandit.select_arm(x)
        true_prob = {a: expit(theta @ x) for a, theta in true_theta.items()}
        maxprob = max(true_prob.values())
        log.append(
            {
                "arm_id": arm_id,
                "reward": np.random.binomial(1, true_prob[arm_id]),
                "regret": maxprob - true_prob[arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


arm_num = 3
feature_num = 5
intercept = False
arm_ids = [f"arm{i}" for i in range(arm_num)]
features = [f"feat{i}" for i in range(feature_num)]
true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
print(true_theta)

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
        reward_df = get_batch(bandit, true_theta, features)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.show()
