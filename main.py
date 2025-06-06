import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from algorithms import (
    RandomAlgorithm,
    # EpsilonGreedyAlgorithm,
    # ThompsonSamplingAlgorithm,
    # BayesUCBAlgorithm,
    # VarianceIDSAlgorithm,
)

# from gaussian_algorithms import (
#     EpsilonGreedyAlgorithm,
#     ThompsonSamplingAlgorithm,
#     BayesUCBAlgorithm,
#     VarianceIDSAlgorithm,
# )

from poisson_algorithms import (
    EpsilonGreedyAlgorithm,
    ThompsonSamplingAlgorithm,
    BayesUCBAlgorithm,
    VarianceIDSAlgorithm,
)

# from linear_algorithms import (
#     EpsilonGreedyAlgorithm,
#     ThompsonSamplingAlgorithm,
#     BayesUCBAlgorithm,
#     VarianceIDSAlgorithm,
# )

from bandits import (
    BernoulliBanditEnv,
    GaussianBanditEnv,
    PoissonBanditEnv,
    LinearBanditEnv,
)

# TODO: add command line config

num_trials = 1000
num_arms = 10
T = 1000


# TODO: move this function somewhere else
def cumulative_regret(bandit_env, rewards):
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


# TODO: refactor this
methods = [
    # "random",
    # "greedy",
    # "e-greedy 0.2",
    "e-greedy decay",
    # "explore-commit 200",
    "Bayes UCB",
    "TS",
    # "V-IDS",
    # "V-IDS argmin",
]


def trial(_):
    regret_sums = np.zeros((len(methods), T))
    regret_sq_sums = np.zeros((len(methods), T))

    # bandit_env = BernoulliBanditEnv(num_arms)
    # bandit_env = GaussianBanditEnv(num_arms)
    bandit_env = PoissonBanditEnv(num_arms)
    # bandit_env = LinearBanditEnv(num_arms, d=5)

    # print("theta:")
    # print(bandit_env.theta)
    # print("means:")
    # print(np.array([arm.mean for arm in bandit_env.arms]))

    algorithms = [
        # RandomAlgorithm(bandit_env),
        # EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.0),
        # EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.2),
        EpsilonGreedyAlgorithm(bandit_env, lambda t: np.power(t + 1, -1 / 3)),
        # EpsilonGreedyAlgorithm(bandit_env, lambda t: 1.0 if t < 200 else 0.0),
        BayesUCBAlgorithm(bandit_env, 0),
        ThompsonSamplingAlgorithm(bandit_env),
        # VarianceIDSAlgorithm(bandit_env, 10000),
        # VarianceIDSAlgorithm(bandit_env, 10000, use_argmin=True),
    ]
    # TODO: refactor this
    assert len(algorithms) == len(methods)

    for i, alg in enumerate(algorithms):
        _, rewards = alg.run(T)
        regrets = cumulative_regret(bandit_env, rewards)
        regret_sums[i] += regrets
        regret_sq_sums[i] += np.square(regrets)

    return regret_sums, regret_sq_sums


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    regret_sums = np.zeros((len(methods), T))
    regret_sq_sums = np.zeros((len(methods), T))

    with Pool(processes=None) as pool:
        it = pool.imap(trial, range(num_trials))
        for r, r_s in tqdm(it, total=num_trials, smoothing=0.1):
            regret_sums += r
            regret_sq_sums += r_s

    """
    Section below is for plotting
    """
    title = "Gamma-Poisson Bandit"
    output = "poisson"

    for i in range(len(methods)):
        plt.plot(regret_sums[i] / num_trials, label=methods[i])
    # plt.xlim(left=0, right=T)
    plt.title(title)
    plt.xlabel("timestep t")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.savefig(f"{output}.png")
    plt.show()

    for i in range(len(methods)):
        plt.plot(regret_sums[i] / num_trials, label=methods[i])
    # lines for comparison
    x = np.arange(1, T)
    y = 8 * np.sqrt(x)
    plt.plot(x, y, "k--")
    # plt.xlim(left=0, right=T)
    # plt.ylim(bottom=0, top=120)
    plt.title(title)
    plt.xlabel("iteration (log)")
    plt.ylabel("cumulative regret (log)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"{output}_log.png")
    plt.show()
