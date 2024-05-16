import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm

from bandit.ucb import UCB
from bandit.logistic_ts import LogisticTS
from bandit.logistic_pgts import LogisticPGTS
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.bandit_base.bandit import BanditBase
from bandit.tools import expand_cascade_data

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
    UCB(arm_ids),
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
plt.title(f"Personalized Cascading Bandit: batch_size={batch_size}, item_num={arm_num}")
plt.show()
