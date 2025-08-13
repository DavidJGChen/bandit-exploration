from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
from collections.abc import Iterable
from cyclopts import App, Parameter
from typing import Annotated
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
from ai_alignment_algorithms import (
    EpsilonThompsonSamplingAlignmentAlgorithm,
    IDSAlignmentAlgorithm,
)
from common import Action, Reward

from icecream import ic
from setting import Settings, get_settings, init_setting

app = App()


@dataclass
class BanditEnvConfig:
    bandit_env: type[BaseBanditEnv]
    bayesian_state: type[BaseBayesianState]
    extra_args: dict
    label: str


bandit_env_configs: dict[str, BanditEnvConfig] = {
    "Beta-Bernoulli Bandit": BanditEnvConfig(
        BernoulliBanditEnv, BetaBernoulliState, {}, "bernoulli"
    ),
    "Gaussian-Gaussian Bandit": BanditEnvConfig(
        GaussianBanditEnv,
        GaussianGaussianState,
        {},
        "gaussian",
    ),
    "Gamma-Poisson Bandit": BanditEnvConfig(
        PoissonBanditEnv, GammaPoissonState, {}, "poisson"
    ),
    "Linear-Gaussian Bandit": BanditEnvConfig(
        LinearBanditEnv,
        LinearGaussianState,
        {"d": 5},
        "linear",
    ),
    "Beta-Bernoulli Alignment Bandit": BanditEnvConfig(
        BernoulliAlignmentBanditEnv,
        BetaBernoulliAlignmentState,
        {},
        "alignment",
    ),
}
bandit_env_name = "Beta-Bernoulli Alignment Bandit"
bandit_env_config: BanditEnvConfig = bandit_env_configs[bandit_env_name]


@dataclass
class AlgorithmConfig:
    label: str
    algorithm_type: type[BaseAlgorithm]
    extra_params: dict


def get_algorithms(settings: Settings) -> list[AlgorithmConfig]:
    # mcmc_particles = settings.mcmc_particles
    T = settings.T
    K = settings.num_arms
    epsilon_for_TS = np.sqrt(K) / np.sqrt(T)

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
        # AlgorithmConfig("V-IDS", VarianceIDSAlgorithm, {"M": mcmc_particles}),
        # AlgorithmConfig(
        #     "V-IDS argmin",
        #     VarianceIDSAlgorithm,
        #     {"M": mcmc_particles, "use_argmin": True},
        # ),
        # AlgorithmConfig("IDS", IDSAlignmentAlgorithm, {"M": mcmc_particles}),
        AlgorithmConfig(
            "TS-ep1",
            EpsilonThompsonSamplingAlignmentAlgorithm,
            {"epsilon_func": lambda t: min(1.0, np.sqrt(K / (t + 1)))},
        ),
        AlgorithmConfig(
            "TS-ep2",
            EpsilonThompsonSamplingAlignmentAlgorithm,
            {"epsilon_func": lambda t: epsilon_for_TS},
        ),
    ]


# TODO: move this function somewhere else
def cumulative_regret(
    bandit_env: BaseBanditEnv, rewards: NDArray[Reward]
) -> NDArray[Reward]:
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


# TODO: move this function somewhere else
def generate_base_filename(base_seed: int, trial_id: int, alg_label) -> str:
    return f"{alg_label}_seed{base_seed}_id{trial_id}.npy"


def trial(
    trial_id: int, settings: Settings
) -> tuple[int, NDArray[Reward], NDArray[float64]]:
    rng = np.random.default_rng([trial_id, settings.base_seed])

    algorithms = get_algorithms(settings)
    num_algs = len(algorithms)

    num_arms = settings.num_arms
    T = settings.T

    all_regrets = np.zeros((num_algs, T))
    all_actions = np.zeros((num_algs, T))
    all_extras = []

    kwargs = bandit_env_config.extra_args
    bandit_env = bandit_env_config.bandit_env(
        num_arms,
        rng,
        **kwargs,
    )
    bayesian_state = bandit_env_config.bayesian_state(bandit_env, rng)

    ic("means:", np.array([arm.mean for arm in bandit_env.arms]))
    ic("best mean:", bandit_env.optimal_mean)

    for i, alg_config in enumerate(algorithms):
        alg_class = alg_config.algorithm_type
        kwargs = alg_config.extra_params
        alg_instance = alg_class(bandit_env, bayesian_state, rng, **kwargs)
        rewards, actions, extras = alg_instance.run(T, trial_id, alg_config.label)
        regrets = cumulative_regret(bandit_env, rewards)

        all_regrets[i] = regrets
        all_actions[i] = actions
        all_extras.append(extras)

    extra_df = pl.from_dicts(all_extras[0])
    ic(extra_df)
    ic("est means:", bayesian_state.get_means())

    return trial_id, all_regrets, all_actions


@app.default()
def main(
    num_trials: Annotated[int, Parameter(alias="-n")] = 100,
    num_processes: int = 10,
    T: int = 500,
    mcmc_particles: int = 10000,
    num_arms: Annotated[int, Parameter(alias="-K")] = 10,
    base_seed: int = 0,
    multiprocessing: bool = True,
    trial_id_overrides: list[int] | None = None,
) -> None:
    """Bandit simulation.

    Parameters
    ----------
    num_trials: int
        The number of trials.
    num_processes: int
        The number of parallel simulation processes.
    T: int
        The horizon for each trial.
    mcmc_particles: int
        The number of particles to use in MCMC for IDS.
    num_arms: int
        The number of bandits arms
    base_seed: int
        The base seed for random number generation.
    multiprocessing: bool
        Whether to enable multiprocessing or not.
    trial_id_overrides: list[int] | None
        Run a specific set of trial IDs. Overrides num_trials.
        This in combination with base_seed determines the random behavior of all trials.
    """
    today = datetime.now()
    output_dir = f"output/{today.strftime('%Y%m%d-%H%M')}-{bandit_env_config.label}"

    init_setting(
        num_trials,
        num_processes,
        T,
        mcmc_particles,
        num_arms,
        base_seed,
        multiprocessing,
        trial_id_overrides,
        output_dir,
    )
    setting = get_settings()

    os.makedirs(output_dir)

    algorithms = get_algorithms(setting)
    num_algs = len(algorithms)

    trial_ids: Iterable[int]
    if trial_id_overrides is not None and len(trial_id_overrides) > 0:
        trial_ids = trial_id_overrides
        num_trials = len(trial_ids)
    else:
        trial_ids = range(num_trials)

    np.set_printoptions(precision=3)

    # ------------------------------------------------------------------

    if multiprocessing:
        with Pool(processes=num_processes) as pool:
            it = pool.imap(partial(trial, settings=setting), trial_ids)
            prog_bar = tqdm_ray.tqdm(total=num_trials, position=0, desc="trials")
            for trial_id, regrets, actions in it:
                for alg_config in algorithms:
                    filename = generate_base_filename(
                        base_seed, trial_id, alg_config.label
                    )
                    with open(
                        os.path.join(output_dir, f"regrets_{filename}"), "wb"
                    ) as f:
                        np.save(f, regrets)
                    with open(
                        os.path.join(output_dir, f"actions_{filename}"), "wb"
                    ) as f:
                        np.save(f, actions)
                prog_bar.update(1)
        ray.shutdown()

    else:
        for trial_id in trial_ids:
            _, regrets, actions = trial(trial_id, settings=setting)
            for alg_config in algorithms:
                filename = generate_base_filename(base_seed, trial_id, alg_config.label)
                with open(os.path.join(output_dir, f"regrets_{filename}"), "wb") as f:
                    np.save(f, regrets)
                with open(os.path.join(output_dir, f"actions_{filename}"), "wb") as f:
                    np.save(f, actions)

    # ------------------------------------------------------------------

    regrets = np.zeros((num_trials, num_algs, T), dtype=Reward)
    chosen_actions = np.zeros((num_trials, num_algs, T), dtype=Action)

    for i, trial_id in enumerate(trial_ids):
        for alg_config in algorithms:
            filename = generate_base_filename(base_seed, trial_id, alg_config.label)
            with open(os.path.join(output_dir, f"regrets_{filename}"), "rb") as f:
                regrets[i] = np.load(f)
            with open(os.path.join(output_dir, f"actions_{filename}"), "rb") as f:
                chosen_actions[i] = np.load(f)

    regret_means = np.mean(regrets, axis=0)
    regret_stds = np.std(regrets, axis=0)

    """
    Section below is for plotting
    """
    title = bandit_env_name
    output = bandit_env_config.label

    # plt.figure(figsize=(5, 5))
    # for i in range(num_trials):
    #     for alg in range(num_algs):
    #         plt.plot(regrets[i][alg], color="blue", alpha=0.3)

    # for alg in range(num_algs):
    #     plt.plot(regret_means[alg], label=algorithms[alg].label, color="blue")
    # # plt.xlim(left=0, right=T)
    # plt.title(title)
    # plt.xlabel("timestep t")
    # plt.ylabel("cumulative regret")
    # plt.legend()
    # plt.savefig(f"images/{output}.png")
    # plt.show()

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
    sqrt_x_log_x = 20 * np.sqrt(x) * np.log(x)
    x_3_4 = 10 * x ** (3 / 4)

    # plt.plot(x, x, "k--")
    plt.plot(x, sqrt_x, "k--")
    plt.plot(x, sqrt_x_log_x, "k--")
    plt.plot(x, x_3_4, "k--")
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
