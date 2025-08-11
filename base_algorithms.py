from abc import ABC, abstractmethod
from collections.abc import Callable

import cvxpy as cp
import numpy as np
from numpy import float64, int_
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from bandits import BaseBanditEnv, LinearBanditEnv
from bayesian_state import BaseBayesianState
from common import Action, Reward

DEBUG = False


class BaseAlgorithm(ABC):
    t: int
    T: int
    reward_history: NDArray[Reward]
    action_history: NDArray[Action]
    bandit_env: BaseBanditEnv
    K: int
    bayesian_state: BaseBayesianState

    def __init__(
        self, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState
    ) -> None:
        self.bandit_env = bandit_env
        self.bayesian_state = bayesian_state
        self.K = self.bandit_env.K

    def run(self, T: int) -> tuple[NDArray[Reward], NDArray[Action]]:
        self.__reset_state(T)
        for t in range(T):
            reward, action = self.__single_step(t)
            self.reward_history[t] = reward
            self.action_history[t] = action
        return self.reward_history, self.action_history

    def __reset_state(self, T: int) -> None:
        self.t = 0
        self.T = T
        self.reward_history = np.zeros(T)
        self.action_history = np.zeros(T, dtype=Action)
        self.bayesian_state.reset_state()
        self.reset_algorithm_state()

    def __single_step(self, t: int) -> tuple[Reward, Action]:
        return self.single_step(t)

    @abstractmethod
    def reset_algorithm_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def single_step(self, t: int) -> tuple[Reward, Action]:
        raise NotImplementedError


# ------------------------------------------------


class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState):
        super().__init__(bandit_env, bayesian_state)

    def reset_algorithm_state(self) -> None:
        pass

    def single_step(self, t: int) -> tuple[Reward, Action]:
        action = np.int_(np.random.choice(self.K))
        return self.bandit_env.sample(action), action


# ------------------------------------------------


class EpsilonGreedyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        bayesian_state: BaseBayesianState,
        epsilon_func: Callable[[int], float],
    ) -> None:
        super().__init__(bandit_env, bayesian_state)
        self.epsilon_func = epsilon_func

    def reset_algorithm_state(self) -> None:
        pass

    def single_step(self, t: int) -> tuple[Reward, Action]:
        if np.random.rand() < self.epsilon_func(t):
            action = np.int_(np.random.choice(self.K))
        else:
            action = np.argmax(self.bayesian_state.get_means())

        reward = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior(reward, action)

        return reward, action


# ------------------------------------------------


class BayesUCBAlgorithm(BaseAlgorithm):
    inv_log_factor: float64

    def __init__(
        self, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState, c: int
    ):
        self.c = c
        super().__init__(bandit_env, bayesian_state)

    def reset_algorithm_state(self) -> None:
        self.inv_log_factor = 1 / np.power(np.log(self.T), self.c)

    def single_step(self, t: int) -> tuple[Reward, Action]:
        if t < self.K:
            action = np.int_(t)
        else:
            quantile = 1 - (self.inv_log_factor / (t + 1))
            quantiles = self.bayesian_state.get_quantiles(quantile)
            action = np.argmax(quantiles)

        reward = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior(reward, action)

        return reward, action


# ------------------------------------------------


class ThompsonSamplingAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState):
        super().__init__(bandit_env, bayesian_state)

    def reset_algorithm_state(self) -> None: ...

    def single_step(self, t: int) -> tuple[Reward, Action]:
        if t < self.K:
            action = np.int_(t)
        else:
            lambda_hats = self.bayesian_state.get_samples()
            action = np.argmax(lambda_hats)

        reward = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior(reward, action)

        return reward, action


# ------------------------------------------------


class VarianceIDSAlgorithm(BaseAlgorithm):
    thetas: NDArray[float64]

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        bayesian_state: BaseBayesianState,
        M: int,
        use_argmin: bool = False,
    ):
        super().__init__(bandit_env, bayesian_state)
        self.M = M  # number of samples for MCMC
        self.use_argmin = use_argmin
        # Not the best way to do this, but need this hack for now.
        self.is_linear = type(bandit_env) is LinearBanditEnv

    def reset_algorithm_state(self) -> None:
        self.thetas = self.__calculate_thetas()

    def single_step(self, t: int) -> tuple[Reward, Action]:
        if t < self.K:
            action = np.int_(t)
        else:
            # estimated means of action parameters
            if self.is_linear:
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
                    np.array([partitioned_thetas[a].shape[1] for a in range(self.K)])
                    / self.M
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
                        p_optimal[a]
                        * np.outer(cond_mu[a] - mu_hat, cond_mu[a] - mu_hat)
                        for a in range(self.K)
                    ],
                    axis=0,
                )

                # estimate expected value of optimal action
                rho = np.sum(
                    [p_optimal[a] * phi[a].T @ cond_mu[a] for a in range(self.K)]
                )
                delta = rho - phi @ mu_hat

                variance = np.nan_to_num(
                    np.array([phi[a].T @ L_hat @ phi[a] for a in range(self.K)])
                )
            else:
                mu = self.bayesian_state.get_means()
                # max action in each sample
                max_action = np.argmax(self.thetas, axis=0)

                # partition the sampled thetas based on which arm is optimal
                partitioned_thetas = [
                    self.thetas[:, np.where(max_action == a)[0]] for a in range(self.K)
                ]

                # probability an action is optimal, approximated using number of samples where
                # it is optimal.
                p_optimal = (
                    np.array([partitioned_thetas[a].shape[1] for a in range(self.K)])
                    / self.M
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
                                else np.zeros(self.K)
                            )
                            for thetas in partitioned_thetas
                        ]
                    )
                )

                # estimate expected value of optimal action
                rho = np.sum([p_optimal[a] * cond_mu[a, a] for a in range(self.K)])
                delta = rho - mu

                # calculate variance term for each arm as an expectation
                variance = np.sum(
                    np.array(
                        [p_optimal[a] * (cond_mu[a] - mu) ** 2 for a in range(self.K)]
                    ),
                    axis=0,
                )

            if self.use_argmin:
                action = np.nan_to_num(np.argmin(delta**2 / variance))
            else:
                action = self.__ids_action_scipy(delta, variance)
                # action = self.__ids_action_cvxpy(delta, variance)

        reward = self.bandit_env.sample(action)

        # Update posterior
        self.bayesian_state.update_posterior(reward, action)

        # Resample thetas for updated action only.
        if self.is_linear:
            self.thetas = self.bayesian_state.get_theta_samples(self.M).T
        else:
            self.thetas[action] = self.__calculate_theta(action)

        return reward, action

    def __ids_action_scipy(self, delta: NDArray[float64], v) -> int_:
        min_ratio: float | None = None
        min_pair: tuple[int, int] = (0, 0)
        q_min: float = 0.0
        for a1 in range(self.K - 1):
            for a2 in range(a1 + 1, self.K):

                def obj(q):
                    return (q * delta[a1] + (1 - q) * delta[a2]) ** 2 / (
                        q * v[a1] + (1 - q) * v[a2]
                    )

                result = minimize_scalar(obj, bounds=(0, 1), method="bounded")  # type: ignore
                info_ratio = result.fun
                q = result.x
                if min_ratio is None or info_ratio < min_ratio:
                    min_ratio = info_ratio
                    q_min = q
                    min_pair = (a1, a2)
        return np.int_(min_pair[0] if np.random.random() < q_min else min_pair[1])

    def __ids_action_cvxpy(self, delta: NDArray[float64], v) -> int_:
        min_ratio: float | None = None
        min_pair: tuple[int, int] = (0, 0)
        q_min: float = 0.0

        q = cp.Variable(nonneg=True)
        delta_a1 = cp.Parameter()
        delta_a2 = cp.Parameter()
        v_a1 = cp.Parameter()
        v_a2 = cp.Parameter()

        objective = cp.Minimize(
            cp.quad_over_lin(
                q * delta_a1 + (1 - q) * delta_a2, q * v_a1 + (1 - q) * v_a2
            )
        )
        problem = cp.Problem(objective, [q <= 1])

        for a1 in range(self.K - 1):
            for a2 in range(a1 + 1, self.K):
                delta_a1.value = delta[a1]
                delta_a2.value = delta[a2]
                v_a1.value = v[a1]
                v_a2.value = v[a2]

                problem.solve()

                info_ratio = problem.value
                opt_q = float(q.value) if q.value is not None else 0.0
                if min_ratio is None or info_ratio < min_ratio:
                    min_ratio = info_ratio
                    q_min = opt_q
                    min_pair = (a1, a2)
        return np.int_(min_pair[0] if np.random.random() < q_min else min_pair[1])

    def __calculate_thetas(self) -> NDArray[float64]:
        return np.array([self.__calculate_theta(action) for action in range(self.K)])

    def __calculate_theta(self, action: Action) -> NDArray[float64]:
        return self.bayesian_state.get_sample_for_action(action, self.M)
