from typing import Annotated, Any, TypeVar

import numpy as np
from pydantic import BaseModel, PlainSerializer

from .ai_alignment_algorithms import (
    EpsilonThompsonSamplingAlignmentAlgorithm,
    IDSAlignmentAlgorithm,
)
from .bandits import (
    BaseBanditEnv,
    BernoulliAlignmentBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
)
from .base_algorithms import (
    BaseAlgorithm,
    # BayesUCBAlgorithm,
    # ThompsonSamplingAlgorithm,
    # VarianceIDSAlgorithm,
)
from .bayesian_state import (
    BaseBayesianState,
    BetaBernoulliAlignmentState,
    BetaBernoulliState,
    GammaPoissonState,
    GaussianGaussianState,
    LinearGaussianState,
)
from .epsilon_functions import SqrtDecay
from .setting import Settings

T = TypeVar("T")


def serializer(class_type: type[Any]) -> str:
    return class_type.__name__


SerializableType = Annotated[type[T], PlainSerializer(serializer)]


class BanditEnvConfig(BaseModel):
    bandit_env: SerializableType[BaseBanditEnv]
    bayesian_state: SerializableType[BaseBayesianState]
    extra_args: dict
    label: str


_bandit_env_configs: dict[str, BanditEnvConfig] = {
    "Beta-Bernoulli Bandit": BanditEnvConfig(
        bandit_env=BernoulliBanditEnv,
        bayesian_state=BetaBernoulliState,
        extra_args={},
        label="bernoulli",
    ),
    "Gaussian-Gaussian Bandit": BanditEnvConfig(
        bandit_env=GaussianBanditEnv,
        bayesian_state=GaussianGaussianState,
        extra_args={},
        label="gaussian",
    ),
    "Gamma-Poisson Bandit": BanditEnvConfig(
        bandit_env=PoissonBanditEnv,
        bayesian_state=GammaPoissonState,
        extra_args={},
        label="poisson",
    ),
    "Linear-Gaussian Bandit": BanditEnvConfig(
        bandit_env=LinearBanditEnv,
        bayesian_state=LinearGaussianState,
        extra_args={"d": 5},
        label="linear",
    ),
    "Beta-Bernoulli Alignment Bandit": BanditEnvConfig(
        bandit_env=BernoulliAlignmentBanditEnv,
        bayesian_state=BetaBernoulliAlignmentState,
        extra_args={},
        label="alignment",
    ),
}
bandit_env_name = "Beta-Bernoulli Alignment Bandit"
bandit_env_config: BanditEnvConfig = _bandit_env_configs[bandit_env_name]


class AlgorithmConfig(BaseModel):
    label: str
    algorithm_type: SerializableType[BaseAlgorithm]
    extra_params: dict[str, SerializableType | Any]


def get_algorithms(settings: Settings) -> list[AlgorithmConfig]:
    mcmc_particles = settings.mcmc_particles

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
        AlgorithmConfig(
            label="IDS",
            algorithm_type=IDSAlignmentAlgorithm,
            extra_params={"M": mcmc_particles, "use_argmin": True},
        ),
        AlgorithmConfig(
            label="TS-ep1",
            algorithm_type=EpsilonThompsonSamplingAlignmentAlgorithm,
            extra_params={"epsilon_factory": SqrtDecay},
        ),
        # AlgorithmConfig(
        #     "TS-ep2",
        #     EpsilonThompsonSamplingAlignmentAlgorithm,
        #     {"epsilon_func": lambda t: epsilon_for_TS},
        # ),
    ]
