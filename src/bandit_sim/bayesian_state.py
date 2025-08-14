from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg
from numpy.random import Generator
from numpy import float64, int_
from numpy.typing import NDArray
from scipy.stats import beta, gamma, norm

from .bandits import (
    BaseBanditEnv,
    BernoulliBanditEnv,
    GaussianBanditEnv,
    LinearBanditEnv,
    PoissonBanditEnv,
    BernoulliAlignmentBanditEnv,
)
from .common import Action, Reward, SampleOutput


class BaseBayesianState[BanditEnv: BaseBanditEnv](ABC):
    bandit_env: BanditEnv
    rng: Generator

    @abstractmethod
    def __init__(self, bandit_env: BanditEnv, rng: Generator) -> None:
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

    def __init__(self, bandit_env: BernoulliBanditEnv, rng: Generator) -> None:
        self.bandit_env = bandit_env
        self.rng = rng

    def reset_state(self) -> None:
        self.alphas = np.ones(self.bandit_env.K, dtype=int_)
        self.betas = np.ones(self.bandit_env.K, dtype=int_)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        self.alphas[action] += reward
        self.betas[action] += 1 - reward

    def get_means(self) -> NDArray[Reward]:
        return self.alphas / (self.alphas + self.betas)

    def get_quantiles(self, quantile: float64) -> NDArray[float64]:
        return np.asarray(beta.ppf(quantile, self.alphas, self.betas))

    def get_samples(self) -> NDArray[float64]:
        return self.rng.beta(
            self.alphas,
            self.betas,
            size=self.bandit_env.K,
        )

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        return self.rng.beta(self.alphas[action], self.betas[action], size=size)


# ------------------------------------------------


class GammaPoissonState(BaseBayesianState[PoissonBanditEnv]):
    alphas: NDArray[int_]
    betas: NDArray[int_]
    bandit_env: PoissonBanditEnv

    def __init__(self, bandit_env: PoissonBanditEnv, rng: Generator) -> None:
        self.bandit_env = bandit_env
        self.rng = rng

    def reset_state(self) -> None:
        self.alphas = np.ones(self.bandit_env.K, dtype=int_)
        self.betas = np.ones(self.bandit_env.K, dtype=int_)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        self.alphas[action] += reward
        self.betas[action] += 1

    def get_means(self) -> NDArray[Reward]:
        return self.alphas / self.betas

    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        return np.asarray(
            gamma.ppf(
                quantile, self.alphas, scale=1 / self.betas, random_state=self.rng
            )
        )

    def get_samples(self) -> NDArray[float64]:
        inv_betas = 1 / self.betas
        return np.asarray(
            gamma.rvs(
                self.alphas,  # type: ignore
                scale=inv_betas,  # type: ignore
                size=(self.bandit_env.K,),
                random_state=self.rng,
            )
        )

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        inv_beta = 1 / self.betas[action]
        return np.asarray(
            gamma.rvs(
                self.alphas[action], scale=inv_beta, size=(size,), random_state=self.rng
            )
        )


# ------------------------------------------------


class GaussianGaussianState(BaseBayesianState[GaussianBanditEnv]):
    mus: NDArray[float64]
    sigmas: NDArray[float64]
    bandit_env: GaussianBanditEnv

    def __init__(self, bandit_env: GaussianBanditEnv, rng: Generator) -> None:
        self.bandit_env = bandit_env
        self.rng = rng

    def reset_state(self) -> None:
        # hard-code initial to N(0,1)
        mu = 0.0
        sigma = 1.0

        self.mus = np.full(self.bandit_env.K, mu)
        self.sigmas = np.full(self.bandit_env.K, sigma)

    def update_posterior(self, reward: Reward, action: Action) -> None:
        eta = self.bandit_env.arms[action].eta
        mu = self.mus[action]
        sigma = self.sigmas[action]
        inv_sigma_sq = 1 / sigma**2
        inv_eta_sq = 1 / eta**2

        inv_denom = 1 / (inv_sigma_sq + inv_eta_sq)

        self.mus[action] = (mu * inv_sigma_sq + reward * inv_eta_sq) * inv_denom
        self.sigmas[action] = np.sqrt(inv_denom)

    def get_means(self) -> NDArray[float64]:
        return self.mus

    def get_quantiles(self, quantile: float64) -> NDArray[Reward]:
        return np.asarray(norm.ppf(quantile, loc=self.mus, scale=self.sigmas))

    def get_samples(self) -> NDArray[float64]:
        return np.asarray(
            self.rng.normal(self.mus, self.sigmas, size=self.bandit_env.K)
        )

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        return np.asarray(
            self.rng.normal(self.mus[action], self.sigmas[action], size=size)
        )


# ------------------------------------------------


class LinearGaussianState(BaseBayesianState[LinearBanditEnv]):
    mu_vec: NDArray[float64]
    Sigma: NDArray[float64]
    bandit_env: LinearBanditEnv

    def __init__(self, bandit_env: LinearBanditEnv, rng: Generator):
        self.bandit_env = bandit_env
        self.rng = rng

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
        return self.rng.multivariate_normal(self.mu_vec, self.Sigma, size=size)

    def get_samples(self) -> NDArray[float64]:
        return self.bandit_env.phi @ self.get_theta_samples()

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        # I think this may be wrong?, but it's ok I don't use this
        return self.get_theta_samples(size=size)[action]


# ------------------------------------------------


class BetaBernoulliAlignmentState(BaseBayesianState[BernoulliAlignmentBanditEnv]):
    alpha_phis: NDArray[int_]
    beta_phis: NDArray[int_]
    alpha_thetas: NDArray[int_]
    beta_thetas: NDArray[int_]
    mutual_info_phis: NDArray[float64]
    mutual_info_thetas: NDArray[float64]
    reward_means: NDArray[Reward]
    bandit_env: BernoulliAlignmentBanditEnv

    def __init__(self, bandit_env: BernoulliAlignmentBanditEnv, rng: Generator) -> None:
        self.bandit_env = bandit_env
        self.rng = rng

    def reset_state(self) -> None:
        self.alpha_phis = np.ones(self.bandit_env.K_env, dtype=int_)
        self.beta_phis = np.ones(self.bandit_env.K_env, dtype=int_)
        self.alpha_thetas = np.ones(self.bandit_env.K_human, dtype=int_)
        self.beta_thetas = np.ones(self.bandit_env.K_human, dtype=int_)

        phi_means = self.alpha_phis / (self.alpha_phis + self.beta_phis)
        theta_means = self.alpha_thetas / (self.alpha_thetas + self.beta_thetas)

        self.reward_means = phi_means * theta_means + (1 - phi_means) * (
            1 - theta_means
        )

        self.mutual_info_phis, self.mutual_info_thetas = self.__calculate_mutual_infos()

    def update_posterior(self, reward: Reward, action: Action) -> None:
        raise NotImplementedError

    def update_posterior_with_outcomes(
        self, output: SampleOutput, action: Action
    ) -> None:
        if self.bandit_env.is_env(action):
            index = action
            self.alpha_phis[index] += output.outcome
            self.beta_phis[index] += 1 - output.outcome
        else:
            index: Action = action - self.bandit_env.K_env
            self.alpha_thetas[index] += output.outcome
            self.beta_thetas[index] += 1 - output.outcome

        phi_mean = self.alpha_phis[index] / (
            self.alpha_phis[index] + self.beta_phis[index]
        )
        theta_mean = self.alpha_thetas[index] / (
            self.alpha_thetas[index] + self.beta_thetas[index]
        )

        self.reward_means[index] = phi_mean * theta_mean + (1 - phi_mean) * (
            1 - theta_mean
        )

        self.mutual_info_phis[index], self.mutual_info_thetas[index] = (
            self.__calculate_mutual_infos(action)
        )

    def get_means(self) -> NDArray[Reward]:
        return np.hstack([self.reward_means, -np.ones(self.bandit_env.K_human)])

    def get_quantiles(self, quantile: float64) -> NDArray[float64]: ...

    def get_samples(self) -> NDArray[float64]:
        phi_samples = self.rng.beta(
            self.alpha_phis, self.beta_phis, size=self.bandit_env.K_env
        )
        theta_samples = self.rng.beta(
            self.alpha_thetas, self.beta_thetas, size=self.bandit_env.K_human
        )

        return phi_samples * theta_samples + (1 - phi_samples) * (1 - theta_samples)

    def get_sample_for_action(self, action: Action, size: int) -> NDArray[float64]:
        if self.bandit_env.is_env(action):
            return self.rng.beta(
                self.alpha_phis[action], self.beta_phis[action], size=size
            )
        else:
            index: Action = action - self.bandit_env.K_env
            return self.rng.beta(
                self.alpha_thetas[index], self.beta_thetas[index], size=size
            )

    def __calculate_mutual_infos(
        self, action: None | Action = None
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        alpha_phis: NDArray[float64]
        beta_phis: NDArray[float64]
        alpha_thetas: NDArray[float64]
        beta_thetas: NDArray[float64]

        if action is not None:
            index: Action
            if self.bandit_env.is_env(action):
                index = action
            else:
                index = action - self.bandit_env.K_env

            alpha_phis = self.alpha_phis[index]
            beta_phis = self.beta_phis[index]
            alpha_thetas = self.alpha_thetas[index]
            beta_thetas = self.beta_thetas[index]
        else:
            alpha_phis = self.alpha_phis
            beta_phis = self.beta_phis
            alpha_thetas = self.alpha_thetas
            beta_thetas = self.beta_thetas

        p_phi_0 = beta_phis / (alpha_phis + beta_phis)
        p_phi_1 = 1 - p_phi_0
        p_theta_0 = beta_thetas / (alpha_thetas + beta_thetas)
        p_theta_1 = 1 - p_theta_0

        curr_entropy_phi = beta.entropy(alpha_phis, beta_phis)
        curr_entropy_theta = beta.entropy(alpha_thetas, beta_thetas)

        next_entropy_phi_0 = beta.entropy(alpha_phis, beta_phis + 1)
        next_entropy_phi_1 = beta.entropy(alpha_phis + 1, beta_phis)
        next_entropy_theta_0 = beta.entropy(alpha_thetas, beta_thetas + 1)
        next_entropy_theta_1 = beta.entropy(alpha_thetas + 1, beta_thetas)

        next_entropy_phi = p_phi_0 * next_entropy_phi_0 + p_phi_1 * next_entropy_phi_1
        next_entropy_theta = (
            p_theta_0 * next_entropy_theta_0 + p_theta_1 * next_entropy_theta_1
        )

        mutual_info_phi = curr_entropy_phi - next_entropy_phi
        mutual_info_theta = curr_entropy_theta - next_entropy_theta

        return mutual_info_phi, mutual_info_theta

    def get_mutual_infos(self) -> NDArray[Reward]:
        return np.hstack([self.mutual_info_phis, self.mutual_info_thetas])
