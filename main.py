from ray.util.multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from cyclopts import App
from functools import partial
# from icecream import ic

from setting import init_setting, get_settings, Settings

from bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
)
from base_algorithms import (
    BaseAlgorithm,
    BayesUCBAlgorithm,
    ThompsonSamplingAlgorithm,
    VarianceIDSAlgorithm,
)
from bayesian_state import (
    BaseBayesianState,
    BetaBernoulliState,
    GammaPoissonState,
    GaussianGaussianState,
    LinearGaussianState,
)


app = App()

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


def get_algorithms(settings: Settings) -> list[tuple[str, type[BaseAlgorithm], dict]]:
    V_IDS_samples = settings.V_IDS_samples

    return [
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
        (
            "V-IDS argmin",
            VarianceIDSAlgorithm,
            {"M": V_IDS_samples, "use_argmin": True},
        ),
    ]


# TODO: move this function somewhere else
def cumulative_regret(bandit_env, rewards):
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


def trial(_, settings: Settings):
    algorithms = get_algorithms(settings)
    num_algs = len(algorithms)

    num_arms = settings.num_arms
    T = settings.T

    regret_sums = np.zeros((num_algs, T))
    regret_sq_sums = np.zeros((num_algs, T))

    kwargs = bandit_env_config[2]
    bandit_env = bandit_env_config[0](
        num_arms,
        **kwargs,
    )
    bayesian_state = bandit_env_config[1](bandit_env)

    # ic("theta:")
    # ic(bandit_env.theta)
    # ic("means:")
    # ic(np.array([arm.mean for arm in bandit_env.arms]))

    for i, (_, alg, kwargs) in enumerate(algorithms):
        alg_instance = alg(bandit_env, bayesian_state, **kwargs)
        rewards, _ = alg_instance.run(T)
        regrets = cumulative_regret(bandit_env, rewards)
        regret_sums[i] += regrets
        regret_sq_sums[i] += np.square(regrets)

    return regret_sums, regret_sq_sums


@app.default()
def main(
    num_trials: int = 100,
    num_processes: int = 10,
    T: int = 500,
    V_IDS_samples: int = 10000,
    num_arms: int = 10,
) -> None:
    """Bandit simulation.

    Parameters
    ----------
    num_trials: int
        The number of trials.
    num_processes: int
        The number of parallel simulation processes
    T: int
        The help string for T
    V_IDS_samples: int
        The number of samples for IDS algorithm
    num_arms: int
        The number of bandits arms
    """

    init_setting(num_trials, num_processes, T, V_IDS_samples, num_arms)
    setting = get_settings()

    algorithms = get_algorithms(setting)
    num_algs = len(algorithms)

    np.set_printoptions(precision=3)

    regret_sums = np.zeros((num_algs, T))
    regret_sq_sums = np.zeros((num_algs, T))

    with Pool(processes=num_processes) as pool:
        it = pool.imap(partial(trial, settings=setting), range(num_trials))
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


if __name__ == "__main__":
    app()
