import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import RandomAlgorithm, EpsilonGreedyAlgorithm, ThompsonSamplingAlgorithm, BayesUCBAlgorithm, VarianceIDSAlgorithm
from bandits import BernoulliBanditEnv

# TODO: add command line config

np.set_printoptions(precision=3)

num_trials = 10
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
    # "e-greedy 0.1",
    # "e-greedy 0.2",
    # "e-greedy decay",
    # "explore-commit 200",
    # "Bayes UCB",
    "TS",
    "V-IDS"
]

regret_sums = np.zeros((len(methods), T))
regret_sq_sums = np.zeros((len(methods), T))

for _ in tqdm(range(num_trials)):
    bandit_env = BernoulliBanditEnv(num_arms)
    # print("arm means")
    # print("---------------")
    # for i, arm in enumerate(bandit_env.arms):
    #     print(f"arm {i}: {arm.mean}")
    # print("---------------")

    algorithms = [
        # RandomAlgorithm(bandit_env),
        # EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.0),
        # EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.1),
        # EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.2),
        # EpsilonGreedyAlgorithm(bandit_env, lambda t: np.power(t+1, -1 / 3)),
        # EpsilonGreedyAlgorithm(bandit_env, lambda t: 1.0 if t < 200 else 0.0),
        # BayesUCBAlgorithm(bandit_env, 0),
        ThompsonSamplingAlgorithm(bandit_env),
        VarianceIDSAlgorithm(bandit_env, 10000),
    ]
    # TODO: refactor this
    assert(len(algorithms) == len(methods))

    for i, alg in enumerate(algorithms):
        _, rewards = alg.run(T)
        regrets = cumulative_regret(bandit_env, rewards)
        regret_sums[i] += regrets
        regret_sq_sums[i] += np.square(regrets)

for i in range(len(methods)):
    plt.plot(regret_sums[i] / num_trials, label=methods[i])
# plt.xlim(left=0, right=T)
# plt.ylim(bottom=0, top=120)
plt.title("beta-Bernoulli Bandit")
plt.xlabel("iteration")
plt.ylabel("cumulative regret")
# plt.yscale("log")
# plt.xscale("log")
plt.legend()
plt.show()