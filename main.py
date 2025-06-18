import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from bayesian_state import (
    BaseBayesianState,
    BetaBernoulliState,
    GaussianGaussianState,
    GammaPoissonState,
    LinearGaussianState,
)

from base_algorithms import (
    BaseAlgorithm,
    RandomAlgorithm,
    EpsilonGreedyAlgorithm,
    ThompsonSamplingAlgorithm,
    BayesUCBAlgorithm,
    VarianceIDSAlgorithm,
)

from bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    PoissonBanditEnv,
    LinearBanditEnv,
)

# TODO: add command line config

num_trials = 100
num_processes = 1
T = 500
V_IDS_samples = 10000

num_arms = 10
bandit_env_configs: dict[
    str, tuple[type[BaseBanditEnv], type[BaseBayesianState], dict, str]
] = {
    "Beta-Bernoulli Bandit": (BernoulliBanditEnv, BetaBernoulliState, {}, "bernoulli"),
    "Gaussian-Gaussian Bandit": (
        GaussianBanditEnv,
        GaussianGaussianState,
        {},
        "gaussian",
    ),
    "Gamma-Poisson Bandit": (PoissonBanditEnv, GammaPoissonState, {}, "poisson"),
    "Linear-Gaussian Bandit": (
        LinearBanditEnv,
        LinearGaussianState,
        {"d": 5},
        "linear",
    ),
}
bandit_env_name = "Beta-Bernoulli Bandit"
bandit_env_config = bandit_env_configs[bandit_env_name]

algorithms: list[tuple[str, type[BaseAlgorithm], dict]] = [
    # ("random", RandomAlgorithm, {}),
    # ("greedy", EpsilonGreedyAlgorithm, {"epsilon_func": lambda _: 0.0}),
    # ("e-greedy 0.2", EpsilonGreedyAlgorithm, {"epsilon_func": lambda _: 0.2}),
    # (
    #     "e-greedy decay",
    #     EpsilonGreedyAlgorithm,
    #     {"epsilon_func": lambda t: np.power(t + 1, -1 / 3)},
    # ),
    # (
    #     "explore-commit 200",
    #     EpsilonGreedyAlgorithm,
    #     {"epsilon_func": lambda t: 1.0 if t < 200 else 0.0},
    # ),
    ("Bayes UCB", BayesUCBAlgorithm, {"c": 0}),
    ("TS", ThompsonSamplingAlgorithm, {}),
    ("V-IDS", VarianceIDSAlgorithm, {"M": V_IDS_samples}),
    ("V-IDS argmin", VarianceIDSAlgorithm, {"M": V_IDS_samples, "use_argmin": True}),
]
num_algs = len(algorithms)


# TODO: move this function somewhere else
def cumulative_regret(bandit_env, rewards):
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


def trial(_):
    regret_sums = np.zeros((num_algs, T))
    regret_sq_sums = np.zeros((num_algs, T))

    kwargs = bandit_env_config[2]
    bandit_env = bandit_env_config[0](
        num_arms,
        **kwargs,
    )
    bayesian_state = bandit_env_config[1](bandit_env)

    # print("theta:")
    # print(bandit_env.theta)
    # print("means:")
    # print(np.array([arm.mean for arm in bandit_env.arms]))

    for i, (_, alg, kwargs) in enumerate(algorithms):
        alg_instance = alg(bandit_env, bayesian_state, **kwargs)
        rewards, _ = alg_instance.run(T)
        regrets = cumulative_regret(bandit_env, rewards)
        regret_sums[i] += regrets
        regret_sq_sums[i] += np.square(regrets)

    return regret_sums, regret_sq_sums


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    regret_sums = np.zeros((num_algs, T))
    regret_sq_sums = np.zeros((num_algs, T))

    with Pool(processes=num_processes) as pool:
        it = pool.imap(trial, range(num_trials))
        for r, r_s in tqdm(it, total=num_trials, smoothing=0.1):
            regret_sums += r
            regret_sq_sums += r_s

    """
    Section below is for plotting
    """
    title = bandit_env_name
    output = bandit_env_config[3]

    for i in range(num_algs):
        plt.plot(regret_sums[i] / num_trials, label=algorithms[i][0])
    # plt.xlim(left=0, right=T)
    plt.title(title)
    plt.xlabel("timestep t")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.savefig(f"{output}.png")
    plt.show()

    for i in range(num_algs):
        plt.plot(regret_sums[i] / num_trials, label=algorithms[i][0])
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
