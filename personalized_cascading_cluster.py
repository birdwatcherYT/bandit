import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit.bernoulli_ts import BernoulliTS
from bandit.logistic_ts import LogisticTS
from bandit.tools import ClusterBandit
from personalized_cascading import get_batch, expand_cascade_data


if __name__ == "__main__":

    batch_size = 10
    arm_num = 20  # アイテム数
    top_k = 10
    init_k = 5

    feature_num = 5
    intercept = False
    arm_ids = [f"arm{i}" for i in range(arm_num)]
    features = [f"feat{i}" for i in range(feature_num)]
    true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
    print(true_theta)
    # only_first_click = True  # 最初のクリックまでを見る場合
    only_first_click = False  # 最後のクリックまで見る場合

    # クラスタリングのためのデータ用意
    num_users = 10000
    user_contexts = np.random.rand(num_users, len(features))
    cluster_num = [5, 10, 20]

    report = {}

    for bandit, suffix in [
        (
            ClusterBandit(
                [BernoulliTS(arm_ids) for _ in range(c)], user_contexts, features
            ),
            f": BernoulliTS x {c}",
        )
        for c in cluster_num
    ] + [
        (LogisticTS(arm_ids, features, intercept), ""),
        (BernoulliTS(arm_ids), ""),
    ]:
        name = bandit.__class__.__name__ + suffix
        print(name)
        regret_log = []
        cumsum_regret = 0
        for i in tqdm(range(2000)):
            reward_df = get_batch(top_k, bandit, true_theta, features, batch_size)
            cumsum_regret += reward_df["regret"].sum()
            regret_log.append(cumsum_regret)
            bandit.train(
                expand_cascade_data(init_k, reward_df, only_first_click, features)
            )
        report[name] = regret_log
    pd.DataFrame(report).plot()
    plt.xlabel("Batch Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title(
        f"Personalized Cascading Bandit: batch_size={batch_size}, item_num={arm_num}, \nonly_first_click={only_first_click}"
    )
    plt.show()
