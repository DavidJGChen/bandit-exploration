import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import RandomAlgorithm, EpsilonGreedyAlgorithm, ThompsonSamplingAlgorithm
from bandits import BernoulliBanditEnv

# TODO: add command line config

num_trials = 1000
num_arms = 10
T = 1000
epsilon = 0.1

def cumulative_regret(bandit_env, rewards):
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward

# TODO: refactor this shit
methods = ["random", "greedy", "e-greedy 0.1", "e-greedy 0.2", "explore-commit 200", "e-greedy decay", "TS"]

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
        RandomAlgorithm(bandit_env),
        EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.0),
        EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.1),
        EpsilonGreedyAlgorithm(bandit_env, lambda _: 0.2),
        EpsilonGreedyAlgorithm(bandit_env, lambda t: 1.0 if t < 200 else 0.0),
        EpsilonGreedyAlgorithm(bandit_env, lambda t: min(1, 0.2 * (1 - t / T))),
        ThompsonSamplingAlgorithm(bandit_env)
    ]
    assert(len(algorithms) == len(methods))

    for i, alg in enumerate(algorithms):
        _, rewards = alg.run(T)
        regrets = cumulative_regret(bandit_env, rewards)
        regret_sums[i] += regrets
        regret_sq_sums[i] += np.square(regrets)

for i in range(len(methods)):
    plt.plot(regret_sums[i] / num_trials, label=methods[i])
plt.xlim(left=0)
plt.ylim(bottom=0, top=120)
plt.title("beta-Bernoulli Bandit")
plt.xlabel("iteration")
plt.ylabel("cumulative regret")
plt.legend()
plt.show()