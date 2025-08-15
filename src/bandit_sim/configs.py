import numpy as np
from dataclasses import dataclass

from .bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
    BernoulliAlignmentBanditEnv,
)
from .base_algorithms import (
    BaseAlgorithm,
    # BayesUCBAlgorithm,
    # ThompsonSamplingAlgorithm,
    # VarianceIDSAlgorithm,
)
from .bayesian_state import (
    BaseBayesianState,
    BetaBernoulliState,
    GammaPoissonState,
    GaussianGaussianState,
    LinearGaussianState,
    BetaBernoulliAlignmentState,
)
from .ai_alignment_algorithms import (
    EpsilonThompsonSamplingAlignmentAlgorithm,
    IDSAlignmentAlgorithm,
)
from .setting import Settings

@dataclass
class BanditEnvConfig:
    bandit_env: type[BaseBanditEnv]
    bayesian_state: type[BaseBayesianState]
    extra_args: dict
    label: str


_bandit_env_configs: dict[str, BanditEnvConfig] = {
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
bandit_env_config: BanditEnvConfig = _bandit_env_configs[bandit_env_name]


@dataclass
class AlgorithmConfig:
    label: str
    algorithm_type: type[BaseAlgorithm]
    extra_params: dict


def get_algorithms(settings: Settings) -> list[AlgorithmConfig]:
    mcmc_particles = settings.mcmc_particles
    # T = settings.T
    K = settings.num_arms
    # epsilon_for_TS = np.sqrt(K) / np.sqrt(T)

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
            "IDS", IDSAlignmentAlgorithm, {"M": mcmc_particles, "use_argmin": True}
        ),
        AlgorithmConfig(
            "TS-ep1",
            EpsilonThompsonSamplingAlignmentAlgorithm,
            {"epsilon_func": lambda t: min(1.0, np.sqrt(K / (t + 1)))},
        ),
        # AlgorithmConfig(
        #     "TS-ep2",
        #     EpsilonThompsonSamplingAlignmentAlgorithm,
        #     {"epsilon_func": lambda t: epsilon_for_TS},
        # ),
    ]