import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bandit.epsilon_greedy import EpsilonGreedyBandit
from bandit.ts_binary import TSBinaryBandit
from bandit.ucb import UCBBandit
from bandit.bandit_base.bandit import BanditBase


def get_batch(
    bandit: BanditBase, truth_prob: dict[str, float], batch_size: int = 1
) -> pd.DataFrame:
    # 学習データ
    log = []
    maxprob = max(truth_prob.values())
    for _ in range(batch_size):
        arm_id = bandit.select_arm()

        log.append(
            {
                "arm_id": arm_id,
                "reward": np.random.binomial(1, truth_prob[arm_id]),
                "regret": maxprob - truth_prob[arm_id],
            }
        )
    return pd.DataFrame(log)


arm_num = 3
arm_ids = [f"arm{i}" for i in range(arm_num)]
truth_prob = {a: np.random.rand() for a in arm_ids}
print(truth_prob)

report = {}
for bandit in [
    TSBinaryBandit(arm_ids),
    UCBBandit(arm_ids),
    EpsilonGreedyBandit(arm_ids),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in range(1000):
        reward_df = get_batch(bandit, truth_prob)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.show()
