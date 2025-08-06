from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg
from numpy import float64, int_
from numpy.typing import NDArray
from scipy.stats import beta, gamma, norm

from bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
)
from common import Action, Reward


class BaseBayesianState[BanditEnv: BaseBanditEnv](ABC):
    bandit_env: BanditEnv

    @abstractmethod
    def __init__(self, bandit_env: BanditEnv) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_means(self) -> NDArray[Reward]:
        raise NotImplementedError

    @abstractmethod
    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        raise NotImplementedError

    @abstractmethod
    def get_samples(self) -> NDArray[float64]:
        raise NotImplementedError

    @abstractmethod
    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        raise NotImplementedError

    @abstractmethod
    def reset_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_posterior(self, reward: Reward, action: Action) -> None:
        raise NotImplementedError


# ------------------------------------------------


class BetaBernoulliState(BaseBayesianState[BernoulliBanditEnv]):
    alphas: NDArray[int_]
    betas: NDArray[int_]
    bandit_env: BernoulliBanditEnv

    def __init__(self, bandit_env: BernoulliBanditEnv) -> None:
        self.bandit_env = bandit_env

    def reset_state(self) -> None:
        self.alphas = np.ones(self.bandit_env.k, dtype=int_)
        self.betas = np.ones(self.bandit_env.k, dtype=int_)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        self.alphas[action] += reward
        self.betas[action] += 1 - reward

    def get_means(self) -> NDArray[Reward]:
        return self.alphas / (self.alphas + self.betas)

    def get_quantiles(self, quantile: float64) -> NDArray[float64]:
        return np.asarray(beta.ppf(quantile, self.alphas, self.betas))

    def get_samples(self) -> NDArray[float64]:
        return np.random.beta(self.alphas, self.betas, size=self.bandit_env.k)

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        return np.random.beta(self.alphas[action], self.betas[action], size=size)


# ------------------------------------------------


class GammaPoissonState(BaseBayesianState[PoissonBanditEnv]):
    alphas: NDArray[int_]
    betas: NDArray[int_]
    bandit_env: PoissonBanditEnv

    def __init__(self, bandit_env: PoissonBanditEnv) -> None:
        self.bandit_env = bandit_env

    def reset_state(self) -> None:
        self.alphas = np.ones(self.bandit_env.k, dtype=int_)
        self.betas = np.ones(self.bandit_env.k, dtype=int_)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        self.alphas[action] += reward
        self.betas[action] += 1

    def get_means(self) -> NDArray[Reward]:
        return self.alphas / self.betas

    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        return np.asarray(gamma.ppf(quantile, self.alphas, scale=1 / self.betas))

    def get_samples(self) -> NDArray[float64]:
        inv_betas = 1 / self.betas
        return np.asarray(
            gamma.rvs(
                self.alphas,  # type: ignore
                scale=inv_betas,  # type: ignore
                size=(self.bandit_env.k,),
            )
        )

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        inv_beta = 1 / self.betas[action]
        return np.asarray(
            gamma.rvs(
                self.alphas[action],
                scale=inv_beta,
                size=(size,),
            )
        )


# ------------------------------------------------


class GaussianGaussianState(BaseBayesianState[GaussianBanditEnv]):
    mus: NDArray[float64]
    sigmas: NDArray[float64]
    bandit_env: GaussianBanditEnv

    def __init__(self, bandit_env: GaussianBanditEnv) -> None:
        self.bandit_env = bandit_env

    def reset_state(self):
        # hard-code initial to N(0,1)
        mu = 0.0
        sigma = 1.0

        self.mus = np.full(self.bandit_env.k, mu)
        self.sigmas = np.full(self.bandit_env.k, sigma)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        eta = self.bandit_env.arms[action].eta
        mu = self.mus[action]
        sigma = self.sigmas[action]
        inv_sigma_sq = 1 / sigma**2
        inv_eta_sq = 1 / eta**2

        inv_denom = 1 / (inv_sigma_sq + inv_eta_sq)

        self.mus[action] = (mu * inv_sigma_sq + reward * inv_eta_sq) * inv_denom
        self.sigmas[action] = np.sqrt(inv_denom)

    def get_means(self):
        return self.mus

    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        return np.asarray(norm.ppf(quantile, loc=self.mus, scale=self.sigmas))

    def get_samples(self) -> NDArray[float64]:
        return np.asarray(
            np.random.normal(self.mus, self.sigmas, size=self.bandit_env.k)
        )

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        return np.asarray(
            np.random.normal(self.mus[action], self.sigmas[action], size=size)
        )


# ------------------------------------------------


class LinearGaussianState(BaseBayesianState[LinearBanditEnv]):
    mu_vec: NDArray[float64]
    Sigma: NDArray[float64]
    bandit_env: LinearBanditEnv

    def __init__(self, bandit_env: LinearBanditEnv):
        self.bandit_env = bandit_env

    def reset_state(self) -> None:
        self.mu_vec = np.ones(self.bandit_env.d)
        self.Sigma = 10 * np.eye(self.bandit_env.d)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        # Assume eta is 1 (standard gaussian noise)
        phi_a = self.bandit_env.arms[action].feature

        # For some ungodly reason np.linalg.inv does not work with with multiprocessing
        Sigma_inv = scipy.linalg.inv(self.Sigma)
        new_Sigma = scipy.linalg.inv(Sigma_inv + np.outer(phi_a, phi_a))

        self.mu_vec = new_Sigma @ (Sigma_inv @ self.mu_vec + reward * phi_a)
        self.Sigma = new_Sigma.astype(float64)

    def get_means(self) -> NDArray[Reward]:
        return self.bandit_env.phi @ self.mu_vec

    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        means = self.bandit_env.phi @ self.mu_vec
        variances = np.sum(
            np.multiply(self.bandit_env.phi @ self.Sigma, self.bandit_env.phi),
            axis=1,
        )
        return np.asarray(norm.ppf(quantile, loc=means, scale=np.sqrt(variances)))

    def get_theta_samples(self, size: int | None = None) -> NDArray[float64]:
        return np.random.multivariate_normal(self.mu_vec, self.Sigma, size=size)

    def get_samples(self) -> NDArray[float64]:
        return self.bandit_env.phi @ self.get_theta_samples()

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        # I think this may be wrong?, but it's ok I don't use this
        return self.get_theta_samples(size=size)[action]
