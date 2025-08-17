import os
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from functools import partial
from typing import Annotated

import numpy as np
import polars as pl
import yaml  # type: ignore
from cyclopts import App, Parameter
from cyclopts.types import NonNegativeInt, PositiveInt, UInt8
from icecream import ic
from numpy import float64
from numpy.typing import NDArray
from ray import ray
from ray.experimental import tqdm_ray
from ray.util.multiprocessing import Pool

from .bandits import BaseBanditEnv
from .common import Reward
from .configs import bandit_env_config, bandit_env_name, get_algorithms
from .setting import Settings, get_settings, init_setting

app = App("bandit-sim")


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
def entry(
    num_trials: Annotated[PositiveInt, Parameter(alias="-n")] = 100,
    num_processes: UInt8 = 10,
    T: Annotated[PositiveInt, Parameter(alias="-T")] = 500,
    mcmc_particles: PositiveInt = 10000,
    num_arms: Annotated[UInt8, Parameter(alias="-K")] = 10,
    base_seed: int = 0,
    multiprocessing: bool = True,
    trial_id_overrides: list[NonNegativeInt] | None = None,
) -> None:
    """Bandit simulation.

    Parameters
    ----------
    num_trials: PositiveInt
        The number of trials.
    num_processes: UInt8
        The number of parallel simulation processes.
    T: PositiveInt
        The horizon for each trial.
    mcmc_particles: PositiveInt
        The number of particles to use in MCMC for IDS.
    num_arms: UInt8
        The number of bandits arms
    base_seed: int
        The base seed for random number generation.
    multiprocessing: bool
        Whether to enable multiprocessing or not.
    trial_id_overrides: list[NonNegativeInt] | None
        Run a specific set of trial IDs. Overrides num_trials.
        This in combination with base_seed determines the random behavior of all trials.
    """
    ic.disable()

    today = datetime.now()
    output_dir = f"output/{today.strftime('%Y%m%d-%H%M')}-{bandit_env_config.label}"

    trial_ids: Iterable[int]
    if trial_id_overrides is not None and len(trial_id_overrides) > 0:
        trial_ids = trial_id_overrides
        num_trials = len(trial_ids)
    else:
        trial_ids = range(num_trials)

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

    algorithms = get_algorithms(setting)
    num_algs = len(algorithms)

    memory_footprint_gb = (num_trials * num_algs * T * 128) >> 30
    GIGABYTE_LIMIT = 12
    if memory_footprint_gb > GIGABYTE_LIMIT:
        raise ValueError(
            f"Total memory footprint of at least {memory_footprint_gb} GB "
            f"potentially exceeds {GIGABYTE_LIMIT} GB. Please lower either the "
            "number of trials, algorithms, or the horizon."
        )

    np.set_printoptions(precision=3)

    # ------------------------------------------------------------------
    # Output config.

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "simulation_config.yaml"), "w") as config_file:
        saved_config = {
            "bandit_env_name": bandit_env_name,
            "bandit_env_config": bandit_env_config.model_dump(),
            "algorithm_configs": [alg_config.model_dump() for alg_config in algorithms],
            "settings": asdict(setting),
        }
        yaml.dump(saved_config, config_file)

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


def main():
    app()


if __name__ == "__main__":
    main()
