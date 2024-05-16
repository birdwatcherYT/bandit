import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit.normal_ts import NormalTS
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.lin_ucb_hybrid import LinUCBHybrid
from bandit.bandit_base.bandit import BanditBase


def get_batch(
    bandit: BanditBase,
    true_theta: dict[str, np.ndarray],
    features: list[str],
    batch_size: int,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        x = np.random.rand(len(features))
        arm_id = bandit.select_arm(x)
        true_mean = {a: (theta @ x) for a, theta in true_theta.items()}
        maxmean = max(true_mean.values())
        log.append(
            {
                "arm_id": arm_id,
                "reward": np.random.normal(true_mean[arm_id], 1),
                "regret": maxmean - true_mean[arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


if __name__ == "__main__":

    batch_size = 100
    arm_num = 3
    feature_num = 5
    intercept = False
    arm_ids = [f"arm{i}" for i in range(arm_num)]
    features = [f"feat{i}" for i in range(feature_num)]
    true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
    print(true_theta)

    report = {}
    for bandit in [
        LinTS(arm_ids, features, intercept),
        LinUCB(arm_ids, features, intercept, alpha=1),
        LinUCBHybrid(arm_ids, features, intercept, alpha=1),
        NormalTS(arm_ids),
    ]:
        name = bandit.__class__.__name__
        print(name)
        regret_log = []
        cumsum_regret = 0
        for i in tqdm(range(100)):
            reward_df = get_batch(bandit, true_theta, features, batch_size)
            cumsum_regret += reward_df["regret"].sum()
            regret_log.append(cumsum_regret)
            bandit.train(reward_df)
        report[name] = regret_log
    pd.DataFrame(report).plot()
    plt.xlabel("Batch Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title(
        f"Contextual Continuous Reward Bandit: batch_size={batch_size}, arm_num={arm_num}"
    )
    plt.show()
