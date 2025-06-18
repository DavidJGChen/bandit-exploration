"""
This module contains everything needed to create a bandit environment.

Environments rely on specific implementation of arms. For example, the
BernoulliBanditEnv instantiates Bernoulli arms.
"""

from abc import abstractmethod
from typing import Protocol
from numpy import float64, int_
from numpy.typing import NDArray
import numpy as np


class BaseArm[OutcomeT](Protocol):
    mean: float64

    @abstractmethod
    def sample(self) -> OutcomeT:
        raise NotImplementedError


class BernoulliArm(BaseArm[float64]):
    def __init__(self, theta: float64):
        self.theta = theta
        self.mean = theta

    def sample(self) -> float64:
        return 1.0 if np.random.rand() < self.theta else 0.0


class GaussianArm(BaseArm[float64]):
    def __init__(self, mu: float64, eta: float64):
        self.mu = mu
        self.eta = eta
        self.mean = mu

    def sample(self) -> float64:
        return np.random.normal(self.mu, self.eta)


class PoissonArm(BaseArm[int_]):
    def __init__(self, rate: float64):
        self.rate = rate
        self.mean = rate

    def sample(self) -> int_:
        return np.random.poisson(self.rate)


class LinearArm(BaseArm[float64]):
    def __init__(self, feature: NDArray[float64], theta: NDArray[float64]):
        self.feature = feature
        self.mean = np.dot(feature, theta)

    def sample(self) -> float64:
        return self.mean + np.random.standard_normal()


class BaseBanditEnv[A: BaseArm]:
    def __init__(self, K: int):
        self.K = K
        self.arms: list[A] = self.initialize_arms()
        self.optimal_action = np.argmax([arm.mean for arm in self.arms])
        self.optimal_mean = self.arms[self.optimal_action].mean

    def initialize_arms(self) -> list[A]:
        raise NotImplementedError

    def sample(self, action) -> float64:
        try:
            return self.arms[action].sample()
        except KeyError as e:
            print("Action invalid:", e)
            raise


class BernoulliBanditEnv(BaseBanditEnv[BernoulliArm]):

    def initialize_arms(self):
        thetas: NDArray[float64] = np.random.uniform(0, 1, size=(self.K,))
        return [BernoulliArm(theta) for theta in thetas]


class GaussianBanditEnv(BaseBanditEnv[GaussianArm]):

    def initialize_arms(self):
        mus: NDArray[float64] = np.random.normal(0, 1, size=(self.K,))
        etas: NDArray[float64] = np.ones(self.K)
        return [GaussianArm(mu, eta) for mu, eta in zip(mus, etas)]


class PoissonBanditEnv(BaseBanditEnv[PoissonArm]):

    def initialize_arms(self):
        rates: NDArray[float64] = np.random.exponential(1.0, size=(self.K,))
        return [PoissonArm(rate) for rate in rates]


class LinearBanditEnv(BaseBanditEnv[LinearArm]):
    def __init__(self, K: int, d: int):
        """
        K: number of arms
        d: dimension of feature vector and hidden theta vector.
        """
        self.d = d
        self.theta: NDArray[float64]
        self.phi: NDArray[float64]
        super().__init__(K)

    def initialize_arms(self):
        mean_vec: NDArray[float64] = np.zeros(self.d)
        covariance: NDArray[float64] = 10 * np.eye(self.d)
        feature_radius = 1 / np.sqrt(5)
        self.theta = np.random.multivariate_normal(mean_vec, covariance)
        features: NDArray[float64] = np.random.uniform(
            -feature_radius, feature_radius, size=(self.K, self.d)
        )
        self.phi = features
        return [LinearArm(feature, self.theta) for feature in features]
