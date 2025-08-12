"""
This module contains everything needed to create a bandit environment.

Environments rely on specific implementation of arms. For example, the
BernoulliBanditEnv instantiates Bernoulli arms.
"""

from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy import float64, int_
from numpy.typing import NDArray

from common import Action, Reward, SampleOutput

T = TypeVar("T", covariant=True)


class _BanditsArm(Protocol[T]):
    mean: float64

    def sample(self) -> T | SampleOutput: ...


P = TypeVar("P", bound=_BanditsArm)


class _BernoulliArm:
    def __init__(self, theta: float64):
        self.theta = theta
        self.mean = theta

    def sample(self) -> float64:
        return np.float64(1.0 if np.random.rand() < self.theta else 0.0)


class _GaussianArm:
    def __init__(self, mu: float64, eta: float64):
        self.mu = mu
        self.eta = eta
        self.mean = mu

    def sample(self) -> float64:
        return np.float64(np.random.normal(self.mu, self.eta))


class _PoissonArm:
    def __init__(self, rate: float64):
        self.rate = rate
        self.mean = rate

    def sample(self) -> int_:
        return int_(np.random.poisson(self.rate))


class _LinearArm:
    def __init__(self, feature: NDArray[float64], theta: NDArray[float64]):
        self.feature = feature
        self.mean = np.dot(feature, theta)

    def sample(self) -> float64:
        return self.mean + np.random.standard_normal()


class _BernoulliAlignmentArmPair:
    def __init__(self, phi: float64, theta: float64):
        self.phi = phi
        self.theta = theta
        self.mean = phi * theta + (1 - phi) * (1 - theta)

    def sample(self) -> SampleOutput:
        """
        returns a tuple of:
            (outcome, reward, is_reward_observed)
        """
        return self.__sample_env()

    def __reward(self, outcome: int) -> Reward:
        return self.theta if outcome == 1 else 1 - self.theta

    def __sample_env(self) -> SampleOutput:
        outcome = 1 if np.random.rand() < self.phi else 0
        return SampleOutput(
            outcome=np.float64(outcome), reward=self.__reward(outcome), reward_obs=False
        )

    def sample_human(self) -> SampleOutput:
        outcome = 1 if np.random.rand() < self.theta else 0
        return SampleOutput(
            outcome=np.float64(outcome), reward=np.float64(-1.0), reward_obs=True
        )


# ------------------------------------------------


class BaseBanditEnv(Generic[P]):
    def __init__(self, K: int):
        self.K = K
        self.arms: list[P] = self.initialize_arms()
        self.optimal_action = np.argmax([arm.mean for arm in self.arms])
        self.optimal_mean = self.arms[self.optimal_action].mean

    def initialize_arms(self) -> list[P]:
        raise NotImplementedError

    def sample(self, action: Action) -> float64 | SampleOutput:
        try:
            return self.arms[action].sample()
        except KeyError as e:
            print("Action invalid:", e)
            raise


class BernoulliBanditEnv(BaseBanditEnv[_BernoulliArm]):
    def initialize_arms(self) -> list[_BernoulliArm]:
        thetas: NDArray[float64] = np.random.uniform(0, 1, size=(self.K,))
        return [_BernoulliArm(theta) for theta in thetas]


class GaussianBanditEnv(BaseBanditEnv[_GaussianArm]):
    def initialize_arms(self) -> list[_GaussianArm]:
        mus: NDArray[float64] = np.random.normal(0, 1, size=(self.K,))
        etas: NDArray[float64] = np.ones(self.K)
        return [_GaussianArm(mu, eta) for mu, eta in zip(mus, etas)]


class PoissonBanditEnv(BaseBanditEnv[_PoissonArm]):
    def initialize_arms(self) -> list[_PoissonArm]:
        rates: NDArray[float64] = np.random.exponential(1.0, size=(self.K,))
        return [_PoissonArm(rate) for rate in rates]


class LinearBanditEnv(BaseBanditEnv[_LinearArm]):
    def __init__(self, K: int, d: int):
        """
        k: number of arms
        d: dimension of feature vector and hidden theta vector.
        """
        self.d = d
        self.theta: NDArray[float64]
        self.phi: NDArray[float64]
        super().__init__(K)

    def initialize_arms(self) -> list[_LinearArm]:
        mean_vec: NDArray[float64] = np.zeros(self.d)
        covariance: NDArray[float64] = 10 * np.eye(self.d)
        feature_radius = 1 / np.sqrt(5)
        self.theta = np.random.multivariate_normal(mean_vec, covariance)
        features: NDArray[float64] = np.random.uniform(
            -feature_radius, feature_radius, size=(self.K, self.d)
        )
        self.phi = features
        return [_LinearArm(feature, self.theta) for feature in features]


class BernoulliAlignmentBanditEnv(BaseBanditEnv[_BernoulliAlignmentArmPair]):
    def __init__(self, K: int):
        if K % 2 != 0:
            raise ValueError("K must be a positive even integer.")
        self.K_env = K // 2
        self.K_human = self.K_env  # defined for convenience.
        super().__init__(K)

    def initialize_arms(self) -> list[_BernoulliAlignmentArmPair]:
        params: NDArray[float64] = np.random.uniform(0, 1, size=(self.K_env, 2))
        return [_BernoulliAlignmentArmPair(phi, theta) for phi, theta in params]

    def is_env(self, action: Action) -> bool:
        return int(action) < self.K_env

    def sample(self, action: Action) -> SampleOutput:
        try:
            if self.is_env(action):  # environment action
                return self.arms[action].sample()
            else:  # human-query action
                return self.arms[action - self.K_env].sample_human()
        except KeyError as e:
            print("Action invalid:", e)
            raise
