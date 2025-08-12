from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
from cyclopts import App
from datetime import datetime
from numpy import float64
from numpy.typing import NDArray
from ray import ray
from ray.util.multiprocessing import Pool
from ray.experimental import tqdm_ray

from bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
    BernoulliAlignmentBanditEnv,
)
from base_algorithms import (
    BaseAlgorithm,
    # BayesUCBAlgorithm,
    # ThompsonSamplingAlgorithm,
    # VarianceIDSAlgorithm,
)
from bayesian_state import (
    BaseBayesianState,
    BetaBernoulliState,
    GammaPoissonState,
    GaussianGaussianState,
    LinearGaussianState,
    BetaBernoulliAlignmentState,
)
from ai_alignment_algorithms import IDSAlignmentAlgorithm
from common import Action, Reward

from icecream import ic
from setting import Settings, get_settings, init_setting

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
    "Beta-Bernoulli Alignment Bandit": (
        BernoulliAlignmentBanditEnv,
        BetaBernoulliAlignmentState,
        {},
        "alignment",
    ),
}
bandit_env_name = "Beta-Bernoulli Alignment Bandit"
bandit_env_config = bandit_env_configs[bandit_env_name]


@dataclass
class AlgorithmConfig:
    label: str
    algorithm_type: type[BaseAlgorithm]
    extra_params: dict


def get_algorithms(settings: Settings) -> list[AlgorithmConfig]:
    V_IDS_samples = settings.V_IDS_samples

    return [
        # AlgorithmConfig("random", RandomAlgorithm, {}),
        # AlgorithmConfig(
        #     "greedy", EpsilonGreedyAlgorithm, {"epsilon_func": lambda _: 0.0}
        # ),
        # AlgorithmConfig(
        #     "e-greedy 0.2", EpsilonGreedyAlgorithm, {"epsilon_func": lambda _: 0.2}
        # ),
        # AlgorithmConfig(
        #     "e-greedy decay",
        #     EpsilonGreedyAlgorithm,
        #     {"epsilon_func": lambda t: np.power(t + 1, -1 / 3)},
        # ),
        # AlgorithmConfig(
        #     "explore-commit 200",
        #     EpsilonGreedyAlgorithm,
        #     {"epsilon_func": lambda t: 1.0 if t < 200 else 0.0},
        # ),
        # AlgorithmConfig("Bayes UCB", BayesUCBAlgorithm, {"c": 0}),
        # AlgorithmConfig("TS", ThompsonSamplingAlgorithm, {}),
        # AlgorithmConfig("V-IDS", VarianceIDSAlgorithm, {"M": V_IDS_samples}),
        # AlgorithmConfig(
        #     "V-IDS argmin",
        #     VarianceIDSAlgorithm,
        #     {"M": V_IDS_samples, "use_argmin": True},
        # ),
        AlgorithmConfig("IDS", IDSAlignmentAlgorithm, {"M": V_IDS_samples}),
    ]


# TODO: move this function somewhere else
def cumulative_regret(
    bandit_env: BaseBanditEnv, rewards: NDArray[Reward]
) -> NDArray[Reward]:
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


def trial(
    trial_num: int, settings: Settings
) -> tuple[int, NDArray[Reward], NDArray[float64]]:
    seed = settings.base_seed + trial_num
    np.random.seed(seed)

    algorithms = get_algorithms(settings)
    num_algs = len(algorithms)

    num_arms = settings.num_arms
    T = settings.T

    all_regrets = np.zeros((num_algs, T))
    all_actions = np.zeros((num_algs, T))
    all_extras = []

    kwargs = bandit_env_config[2]
    bandit_env = bandit_env_config[0](
        num_arms,
        **kwargs,
    )
    bayesian_state = bandit_env_config[1](bandit_env)

    ic("means:", np.array([arm.mean for arm in bandit_env.arms]))
    ic("best mean:", bandit_env.optimal_mean)

    for i, alg_config in enumerate(algorithms):
        alg_class = alg_config.algorithm_type
        kwargs = alg_config.extra_params
        alg_instance = alg_class(bandit_env, bayesian_state, **kwargs)
        rewards, actions, extras = alg_instance.run(T, trial_num)
        regrets = cumulative_regret(bandit_env, rewards)

        all_regrets[i] = regrets
        all_actions[i] = actions
        all_extras.append(extras)

    extra_df = pl.from_dicts(all_extras[0])
    ic(extra_df)

    ic("est means:", bayesian_state.get_means())

    return trial_num, all_regrets, all_actions


@app.default()
def main(
    num_trials: int = 100,
    num_processes: int = 10,
    T: int = 500,
    V_IDS_samples: int = 10000,
    num_arms: int = 10,
    base_seed: int = 0,
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
    today = datetime.now()
    output_dir = f"output/{today.strftime('%Y%m%d-%H-%M')}-{bandit_env_config[3]}"

    init_setting(
        num_trials, num_processes, T, V_IDS_samples, num_arms, base_seed, output_dir
    )
    setting = get_settings()

    os.makedirs(output_dir)

    algorithms = get_algorithms(setting)
    num_algs = len(algorithms)

    np.set_printoptions(precision=3)

    # ------------------------------------------------------------------

    # with Pool(processes=num_processes) as pool:
    #     it = pool.imap(partial(trial, settings=setting), range(num_trials))
    #     prog_bar = tqdm_ray.tqdm(total=num_trials, position=0, desc="trials")
    #     for trial_num, r, actions in it:
    #         seed = base_seed + trial_num
    #         for alg_config in algorithms:
    #             filename = f"{trial_num}_{alg_config.label}_seed{seed}.npy"
    #             with open(os.path.join(output_dir, f"regrets_{filename}"), 'wb') as f:
    #                 np.save(f, r)
    #             with open(os.path.join(output_dir, f"actions_{filename}"), 'wb') as f:
    #                 np.save(f, actions)

    #         prog_bar.update(1)

    # ray.shutdown()

    # ------------------------------------------------------------------

    trial_num = 6
    _, r, actions = trial(trial_num, settings=setting)
    seed = base_seed + trial_num
    for alg_config in algorithms:
        filename = f"{trial_num}_{alg_config.label}_seed{seed}.npy"
        with open(os.path.join(output_dir, f"regrets_{filename}"), "wb") as f:
            np.save(f, r)
        with open(os.path.join(output_dir, f"actions_{filename}"), "wb") as f:
            np.save(f, actions)

    # ------------------------------------------------------------------

    regrets = np.zeros((num_trials, num_algs, T), dtype=Reward)
    chosen_actions = np.zeros((num_trials, num_algs, T), dtype=Action)

    for trial_num in range(num_trials):
        seed = base_seed + trial_num
        for alg_config in algorithms:
            filename = f"{trial_num}_{alg_config.label}_seed{seed}.npy"
            with open(os.path.join(output_dir, f"regrets_{filename}"), "rb") as f:
                regrets[trial_num] = np.load(f)
            with open(os.path.join(output_dir, f"actions_{filename}"), "rb") as f:
                chosen_actions[trial_num] = np.load(f)

    regret_means = np.mean(regrets, axis=0)
    regret_stds = np.std(regrets, axis=0)

    """
    Section below is for plotting
    """
    title = bandit_env_name
    output = bandit_env_config[3]

    plt.figure(figsize=(5, 5))
    for i in range(num_trials):
        for alg in range(num_algs):
            plt.plot(regrets[i][alg], color="blue", alpha=0.3)

    for alg in range(num_algs):
        plt.plot(regret_means[alg], label=algorithms[alg].label, color="blue")
    # plt.xlim(left=0, right=T)
    plt.title(title)
    plt.xlabel("timestep t")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.savefig(f"images/{output}.png")
    plt.show()

    # log log plot
    plt.figure(figsize=(5, 5))
    for alg in range(num_algs):
        plt.plot(
            np.arange(10, T),
            regret_means[alg][10:],
            label=algorithms[alg].label,
        )
        plt.fill_between(
            np.arange(10, T),
            regret_means[alg][10:] + regret_stds[alg][10:],
            regret_means[alg][10:] - regret_stds[alg][10:],
            alpha=0.3,
        )
    # lines for comparison
    x = np.arange(10, T)
    sqrt_x = 30 * np.sqrt(x)

    plt.plot(x, x, "k--")
    plt.plot(x, sqrt_x, "k--")
    # plt.xlim(left=0, right=T)
    # plt.ylim(bottom=0, top=120)
    plt.title(title)
    plt.xlabel("iteration (log)")
    plt.ylabel("cumulative regret (log)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"images/{output}_log.png")
    plt.show()


if __name__ == "__main__":
    app()
