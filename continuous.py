import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit.epsilon_greedy import EpsilonGreedyBandit
from bandit.ts_continuous import TSContinuousBandit
from bandit.ucb import UCBBandit
from bandit.bandit_base.bandit import BanditBase


def get_batch(
    bandit: BanditBase, true_param: dict[str, dict[str, float]], batch_size: int
) -> pd.DataFrame:
    # 学習データ
    log = []
    maxprob = max([v["mu"] for v in true_param.values()])
    for _ in range(batch_size):
        arm_id = bandit.select_arm()

        log.append(
            {
                "arm_id": arm_id,
                "reward": np.random.normal(
                    true_param[arm_id]["mu"], true_param[arm_id]["sigma"]
                ),
                "regret": maxprob - true_param[arm_id]["mu"],
            }
        )
    return pd.DataFrame(log)


batch_size = 1
arm_num = 5
arm_ids = [f"arm{i}" for i in range(arm_num)]
true_param = {a: {"mu": np.random.rand(), "sigma": np.random.rand()} for a in arm_ids}
print(true_param)

report = {}
for bandit in [
    TSContinuousBandit(arm_ids),
    UCBBandit(arm_ids),
    EpsilonGreedyBandit(arm_ids, epsilon=0.1),
]:
    name = bandit.__class__.__name__
    print(name)
    regret_log = []
    cumsum_regret = 0
    for i in tqdm(range(10000)):
        reward_df = get_batch(bandit, true_param, batch_size)
        cumsum_regret += reward_df["regret"].sum()
        regret_log.append(cumsum_regret)
        bandit.train(reward_df)
    report[name] = regret_log
pd.DataFrame(report).plot()
plt.xlabel("Batch Iteration")
plt.ylabel("Cumulative Regret")
plt.title(f"Continuous Reward Bandit: batch_size={batch_size}, arm_num={arm_num}")
plt.show()
