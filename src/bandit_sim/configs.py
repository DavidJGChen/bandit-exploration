"""
This file is heavily WIP. I intend to rework it to the point where config files can
be saved with each run, and those same config files can be loaded into the bandit
simulator to reproduce the run.Action

I'm using Pydantic for serialization and validation.Action

As of right now, the configuration is hard-coded in this file, and also requires
some tricks to make sure classes are loaded properly. However, outputting the config
to a YAML seems to work fine.

TODO:
- Create intermediate Enums for each class, and wrap them to be serializable and
    deserializable.
- Somehow create validator for epsilon functions
- Write code to load in config and validate it to config models.
"""

from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, BeforeValidator, PlainSerializer

from .ai_alignment_algorithms import *  # noqa: F403
from .bandits import BaseBanditEnv
from .base_algorithms import BaseAlgorithm
from .bayesian_state import BaseBayesianState
from .epsilon_functions import EpsilonFactory
from .setting import Settings

# This is a workaround to make sure all the subclasses are loaded properly.
class_dict = (
    {alg.__name__: alg for alg in BaseAlgorithm.__subclasses__()}
    | {env.__name__: env for env in BaseBanditEnv.__subclasses__()}
    | {state.__name__: state for state in BaseBayesianState.__subclasses__()}
    | {ep_func.__name__: ep_func for ep_func in EpsilonFactory.__subclasses__()}
)

T = TypeVar("T")


def serializer(class_type: type[Any]) -> str:
    return class_type.__name__


def validator[T](string: str) -> type[Any]:
    if not isinstance(string, str):
        raise ValueError("Not a str")
    target_class = class_dict[string]
    if not isinstance(target_class, type):
        raise ValueError("Not a class")
    return target_class


SerializableType = Annotated[
    type[T], PlainSerializer(serializer), BeforeValidator(validator)
]


class BanditEnvConfig(BaseModel):
    bandit_env: SerializableType[BaseBanditEnv]
    bayesian_state: SerializableType[BaseBayesianState]
    extra_args: dict
    label: str


_bandit_env_configs: dict[str, BanditEnvConfig] = {
    "Beta-Bernoulli Bandit": BanditEnvConfig(
        bandit_env="BernoulliBanditEnv",
        bayesian_state="BetaBernoulliState",
        extra_args={},
        label="bernoulli",
    ),
    "Gaussian-Gaussian Bandit": BanditEnvConfig(
        bandit_env="GaussianBanditEnv",
        bayesian_state="GaussianGaussianState",
        extra_args={},
        label="gaussian",
    ),
    "Gamma-Poisson Bandit": BanditEnvConfig(
        bandit_env="PoissonBanditEnv",
        bayesian_state="GammaPoissonState",
        extra_args={},
        label="poisson",
    ),
    "Linear-Gaussian Bandit": BanditEnvConfig(
        bandit_env="LinearBanditEnv",
        bayesian_state="LinearGaussianState",
        extra_args={"d": 5},
        label="linear",
    ),
    "Beta-Bernoulli Alignment Bandit": BanditEnvConfig(
        bandit_env="BernoulliAlignmentBanditEnv",
        bayesian_state="BetaBernoulliAlignmentState",
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
        # AlgorithmConfig("V-IDS", VarianceIDSAlgorithm, {"M": 10000}),
        # AlgorithmConfig(
        #     "V-IDS argmin",
        #     VarianceIDSAlgorithm,
        #     {"M": 10000, "use_argmin": True},
        # ),
        AlgorithmConfig(
            label="IDS",
            algorithm_type="IDSAlignmentAlgorithm",
            extra_params={"M": 10000, "use_argmin": False},
        ),
        AlgorithmConfig(
            label="TS-ep1",
            algorithm_type="EpsilonThompsonSamplingAlignmentAlgorithm",
            extra_params={"epsilon_factory": "SqrtDecay"},
        ),
        # AlgorithmConfig(
        #     "TS-ep2",
        #     EpsilonThompsonSamplingAlignmentAlgorithm,
        #     {"epsilon_func": lambda t: epsilon_for_TS},
        # ),
    ]
