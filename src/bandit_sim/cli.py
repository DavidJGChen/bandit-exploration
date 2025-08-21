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
from numpy.typing import NDArray
from ray import ray
from ray.experimental import tqdm_ray
from ray.util.multiprocessing import Pool

from .bandits import BaseBanditEnv
from .common import Reward
from .configs import bandit_env_config, bandit_env_name, get_algorithms
from .setting import Settings, generate_base_filename, get_settings, init_setting

app = App("bandit-sim")


# TODO: move this function somewhere else
def cumulative_regret(
    bandit_env: BaseBanditEnv, rewards: NDArray[Reward]
) -> NDArray[Reward]:
    T = len(rewards)
    optimal_reward = bandit_env.optimal_mean
    cumulative_reward = np.cumulative_sum(rewards)
    return optimal_reward * np.arange(1, T + 1) - cumulative_reward


def trial(
    x: tuple[int, int], settings: Settings
) -> tuple[int, int, pl.DataFrame, NDArray]:
    trial_num, trial_id = x
    rng = np.random.default_rng([trial_id, settings.base_seed])
    initial_rng_state = rng.bit_generator.state

    algorithms = get_algorithms(settings)

    num_arms = settings.num_arms
    T = settings.T

    kwargs = bandit_env_config.extra_args
    bandit_env = bandit_env_config.bandit_env(
        num_arms,
        rng,
        **kwargs,
    )
    bayesian_state = bandit_env_config.bayesian_state(bandit_env, rng)

    result_df = pl.DataFrame()  # TODO: Add schema

    for i, alg_config in enumerate(algorithms):
        # reset rng state
        rng.bit_generator.state = initial_rng_state

        alg_class = alg_config.algorithm_type
        kwargs = alg_config.extra_params
        alg_instance = alg_class(bandit_env, bayesian_state, rng, **kwargs)
        df: pl.DataFrame = alg_instance.run(T, trial_id, alg_config.label).with_columns(
            algorithm=pl.lit(alg_config.label, pl.Categorical),
            time_step=pl.arange(0, T, dtype=pl.UInt32),
        )

        # TODO: Potentially inefficient, can refactor.
        regrets = cumulative_regret(bandit_env, df["reward"])
        df = df.with_columns(regret=pl.Series(regrets))
        ic(result_df.schema)
        ic(df.schema)

        result_df = pl.concat([result_df, df], how="diagonal")

    return (
        trial_num,
        trial_id,
        result_df.with_columns(trial=pl.lit(trial_id, dtype=pl.UInt16)),
        bandit_env.export_params(),
    )


@app.default()
def entry(
    num_trials: Annotated[PositiveInt, Parameter(alias="-n")] = 100,
    num_processes: UInt8 = 10,
    T: Annotated[PositiveInt, Parameter(alias="-T")] = 500,
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

    trial_ids: Iterable[tuple[int, int]]
    if trial_id_overrides is not None and len(trial_id_overrides) > 0:
        trial_ids = list(enumerate(trial_id_overrides))
        num_trials = len(trial_ids)
    else:
        trial_ids = list(enumerate(range(num_trials)))

    init_setting(
        num_trials,
        num_processes,
        T,
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

    bandit_env_param_list: list[NDArray | None] = [None for _ in range(num_trials)]

    ic(trial_ids)

    if multiprocessing:
        with Pool(processes=num_processes) as pool:
            it = pool.imap(partial(trial, settings=setting), trial_ids)
            prog_bar = tqdm_ray.tqdm(total=num_trials, position=0, desc="trials")
            for i, trial_id, df, bandit_env_params in it:
                ic(i, trial_id)
                bandit_env_param_list[i] = bandit_env_params
                filename = generate_base_filename(base_seed, trial_id)
                with open(os.path.join(output_dir, f"data_{filename}"), "wb") as f:
                    df.write_parquet(f)
                prog_bar.update(1)
        ray.shutdown()

    else:
        for i, trial_id in trial_ids:
            _, _, df, bandit_env_params = trial((i, trial_id), settings=setting)
            bandit_env_param_list[i] = bandit_env_params
            filename = generate_base_filename(base_seed, trial_id)
            with open(os.path.join(output_dir, f"data_{filename}"), "wb") as f:
                df.write_parquet(f)

    ic(bandit_env_param_list)
    bandit_env_params_array = np.array(bandit_env_param_list)
    filename = generate_base_filename(base_seed)
    with open(os.path.join(output_dir, f"bandit_env_params_{filename}"), "wb") as f:
        np.save(f, bandit_env_params_array)


def main():
    app()


if __name__ == "__main__":
    main()
