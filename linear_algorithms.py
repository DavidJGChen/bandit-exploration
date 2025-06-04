import scipy.linalg
from algorithms import BaseBayesianAlgorithm
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize_scalar

DEBUG = False

"""
Ideally this entire file should not exist, but let's just copy paste to save time.
"""


class BaseLinearAlgorithm(BaseBayesianAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)
        self.mu_vec = None
        self.Sigma = None

    def reset_bayesian_state(self):
        self.mu_vec = np.ones(self.bandit_env.d)
        self.Sigma = 10 * np.eye(self.bandit_env.d)

    def update_bayesian_posterior(self, action, reward):
        # Assume eta is 1 (standard gaussian noise
        phi_a = self.bandit_env.arms[action].feature

        # For some ungodly reason np.linalg.inv does not work with with multiprocessing
        Sigma_inv = scipy.linalg.inv(self.Sigma)
        new_Sigma = scipy.linalg.inv(Sigma_inv + np.outer(phi_a, phi_a))

        self.mu_vec = new_Sigma @ (Sigma_inv @ self.mu_vec + reward * phi_a)
        self.Sigma = new_Sigma

        # print(f"action:\t{action}")
        # print(f"phi_a:\t{phi_a}")
        # print(f"reward:\t{reward}")
        # print(f"mu_vec:\t{self.mu_vec}")
        # # print(f"sigma:\t{self.Sigma}")
        # print(f"means:\t{self.get_means()}")
        # print(f"true means:\t{np.array([arm.mean for arm in self.bandit_env.arms])}")
        # print(f"true theta:\t{self.bandit_env.theta}")

    def get_means(self):
        return self.bandit_env.phi @ self.mu_vec


# ------------------------------------------------


class EpsilonGreedyAlgorithm(BaseLinearAlgorithm):
    def __init__(self, bandit_env, epsilon_func):
        super().__init__(bandit_env)
        self.epsilon_func = epsilon_func

    def reset_state(self):
        self.reset_bayesian_state()

    def single_step(self, t):
        if np.random.rand() < self.epsilon_func(t):
            action = np.random.choice(self.K)
        else:
            action = int(np.argmax(self.get_means()))

        reward = self.bandit_env.sample(action)

        self.update_bayesian_posterior(action, reward)

        return action, reward


# ------------------------------------------------


class BayesUCBAlgorithm(BaseLinearAlgorithm):
    def __init__(self, bandit_env, c):
        self.c = c
        super().__init__(bandit_env)
        self.inv_log_factor = None

    def reset_state(self):
        self.reset_bayesian_state()
        self.inv_log_factor = 1 / np.power(np.log(self.T), self.c)

    def single_step(self, t):
        quantile = 1 - (self.inv_log_factor / (t + 1))
        means = self.bandit_env.phi @ self.mu_vec
        variances = np.sum(
            np.multiply(self.bandit_env.phi @ self.Sigma, self.bandit_env.phi),
            axis=1,
        )
        quantiles = norm.ppf(quantile, loc=means, scale=np.sqrt(variances))

        action = np.argmax(quantiles)

        reward = self.bandit_env.sample(action)

        self.update_bayesian_posterior(action, reward)

        return action, reward


# ------------------------------------------------


class ThompsonSamplingAlgorithm(BaseLinearAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)

    def reset_state(self):
        self.reset_bayesian_state()

    def single_step(self, t):
        theta_hat = np.random.multivariate_normal(self.mu_vec, self.Sigma)

        action = np.argmax(self.bandit_env.phi @ theta_hat)

        reward = self.bandit_env.sample(action)

        self.update_bayesian_posterior(action, reward)

        return action, reward


# ------------------------------------------------


class VarianceIDSAlgorithm(BaseLinearAlgorithm):
    def __init__(self, bandit_env, M, use_argmin=False):
        super().__init__(bandit_env)
        self.M = M  # number of samples for MCMC
        self.use_argmin = use_argmin
        self.thetas = None

    def reset_state(self):
        self.reset_bayesian_state()
        self.thetas = self.__calculate_thetas()

    def single_step(self, t):
        # estimated means of action parameters
        mu_hat = np.mean(self.thetas, axis=1)
        phi = self.bandit_env.phi

        # max action in each sample
        max_action = np.argmax(phi @ self.thetas, axis=0)

        # partition the sampled thetas based on which arm is optimal
        partitioned_thetas = [
            self.thetas[:, np.where(max_action == a)[0]] for a in range(self.K)
        ]

        # probability an action is optimal, approximated using number of samples where
        # it is optimal.
        p_optimal = (
            np.array([partitioned_thetas[a].shape[1] for a in range(self.K)]) / self.M
        )

        # calculate est. mean of actions conditioned on action being optimal.
        # shape = (K, K). Indexing once (cond_mu[a_star]) gives us an array
        # of means of all arms conditioned on a_star being optimal.
        cond_mu = np.nan_to_num(
            np.array(
                [
                    (
                        np.mean(thetas, axis=1)
                        if thetas.shape[1] > 0
                        else np.zeros(self.bandit_env.d)
                    )
                    for thetas in partitioned_thetas
                ]
            )
        )

        L_hat = np.sum(
            [
                p_optimal[a] * np.outer(cond_mu[a] - mu_hat, cond_mu[a] - mu_hat)
                for a in range(self.K)
            ],
            axis=0,
        )

        # estimate expected value of optimal action
        rho = np.sum([p_optimal[a] * phi[a].T @ cond_mu[a] for a in range(self.K)])
        delta = rho - phi @ mu_hat

        # calculate variance term for each arm as an expectation
        # variance = np.sum(
        #     np.array(
        #         [
        #             p_optimal[a] * (phi @ (cond_mu[a] - mu_hat)) ** 2
        #             for a in range(self.K)
        #         ]
        #     ),
        #     axis=0,
        # )

        variance = np.nan_to_num(
            np.array([phi[a].T @ L_hat @ phi[a] for a in range(self.K)])
        )

        if self.use_argmin:
            action = np.argmin(delta**2 / variance)
        else:
            action = self.__ids_action(delta, variance)

        reward = self.bandit_env.sample(action)

        # Update posterior
        self.update_bayesian_posterior(action, reward)

        # Resample thetas for updated action only.
        self.thetas = self.__calculate_thetas()

        if DEBUG:
            print(f"\n--------round {t}--------")
            print(f"mu:\t\t\t\t{self.mu_vec}")
            # print(f"times chosen:\t\t\t{self.alphas + self.betas - 2}")
            for a in range(self.K):
                print(f"--Assume action {a} is optimal--")
                print(f"mean of action {a}:\t{self.bandit_env.arms[a].mean}")
                print(f"p_optimal({a}):\t\t\t{p_optimal[a]}")
                print(f"mean vector given {a} optimal:\t{cond_mu[a]}")
                print(f"est reward given {a} optimal:\t{phi[a] @ cond_mu[a]}")
            print(f"---more stats---")
            print(f"rho_star:\t\t\t{rho}")
            print(f"delta vector:\t\t\t{delta}")
            print(f"L_hat:\t\t\t{L_hat}")
            print(f"variance:\t\t\t{variance}")
            print(f"info ratio:\t\t\t{delta**2 / variance}")
            print(f"action chosen:\t\t\t{action}")

        return action, reward

    def __ids_action(self, delta, v):
        min_ratio = None
        min_pair = None
        q_min = None
        for a1 in range(self.K - 1):
            for a2 in range(a1 + 1, self.K):
                obj = lambda q: (q * delta[a1] + (1 - q) * delta[a2]) ** 2 / (
                    q * v[a1] + (1 - q) * v[a2]
                )
                result = minimize_scalar(obj, bounds=(0, 1), method="bounded")
                info_ratio = result.fun
                q = result.x
                if min_ratio == None or info_ratio < min_ratio:
                    min_ratio = info_ratio
                    q_min = q
                    min_pair = (a1, a2)
        return min_pair[0] if np.random.random() < q_min else min_pair[1]

    def __calculate_thetas(self):
        return np.random.multivariate_normal(self.mu_vec, self.Sigma, size=self.M).T
