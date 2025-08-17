from typing import Callable, Protocol, TypeAlias

import numpy as np

from .bandits import BaseBanditEnv
from .bayesian_state import BaseBayesianState

EpsilonFunction: TypeAlias = Callable[[int], float]


class EpsilonFactory(Protocol):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction: ...


class Greedy(EpsilonFactory):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction:
        return lambda _: 0.0


class Constant(EpsilonFactory):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction:
        return lambda _: 0.2


class ExpDecay(EpsilonFactory):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction:
        return lambda t: np.power(t + 1, -1 / 3)


class SqrtDecay(EpsilonFactory):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction:
        K = bandit_env.K
        return lambda t: min(1.0, np.sqrt(K / (t + 1)))


class HorizonSqrtDecay(EpsilonFactory):
    @staticmethod
    def func_creator(
        T: int, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> EpsilonFunction:
        K = bandit_env.K
        return lambda t: min(1.0, np.sqrt(K / T))
