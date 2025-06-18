from collections.abc import Callable
from abc import ABC, abstractmethod
from bandits import BaseBanditEnv
from bayesian_state import BaseBayesianState
from numpy.typing import NDArray
from numpy import float64, int_
import numpy as np
import scipy
from scipy.stats import gamma
from scipy.optimize import minimize_scalar

DEBUG = False

Reward = float64
Action = int_


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

    def run(self, T) -> tuple[NDArray[Reward], NDArray[Action]]:
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

    def single_step(self, t: int):
        action = np.int_(np.random.choice(self.K))
        return self.bandit_env.sample(action), action


# ------------------------------------------------


class EpsilonGreedyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        bayesian_state: BaseBayesianState,
        epsilon_func: Callable[[int], float64],
    ) -> None:
        super().__init__(bandit_env, bayesian_state)
        self.epsilon_func = epsilon_func

    def reset_state(self) -> None:
        pass

    def single_step(self, t: int):
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
        self, bandit_env: BaseBanditEnv, bayesian_state: BaseBayesianState, c: int_
    ):
        self.c = c
        super().__init__(bandit_env, bayesian_state)

    def reset_state(self):
        self.inv_log_factor = 1 / np.power(np.log(self.T), self.c)

    def single_step(self, t: int):
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

    def reset_state(self):
        pass

    def single_step(self, t: int):
        if t < self.K:
            action = np.int_(t)
        else:
            lambda_hats = self.bayesian_state.get_samples()
            action = np.argmax(lambda_hats)

        reward = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior(reward, action)

        return reward, action


# ------------------------------------------------

"""
class VarianceIDSAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env, M, use_argmin=False):
        super().__init__(bandit_env)
        self.M = M  # number of samples for MCMC
        self.use_argmin = use_argmin
        self.rates = None

    def reset_state(self):
        self.reset_bayesian_state()
        self.thetas = self.__calculate_thetas()

    def single_step(self, t):
        if t < self.K:
            action = t
        else:
            # estimated means of action parameters
            mu = self.get_means()

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
                action = np.argmin(delta**2 / variance)
            else:
                action = self.__ids_action(delta, variance)

        reward = self.bandit_env.sample(action)

        # Update posterior
        self.update_bayesian_posterior(action, reward)

        # Resample thetas for updated action only.
        self.thetas[action] = self.__calculate_theta(action)

        if DEBUG:
            print(f"\n--------round {t}--------")
            print(f"mu:\t\t\t\t{mu}")
            print(f"times chosen:\t\t\t{self.betas - 1}")
            for action in range(self.K):
                print(f"--Assume action {action} is optimal--")
                print(f"estimated mean of action {action}:\t{mu[action]}")
                print(f"p_optimal({action}):\t\t\t{p_optimal[action]}")
                print(f"mean vector given {action} optimal:\t{cond_mu[action]}")
            print(f"---more stats---")
            print(f"rho_star:\t\t\t{rho}")
            print(f"delta vector:\t\t\t{delta}")
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
        return np.array([self.__calculate_theta(action) for action in range(self.K)])

    def __calculate_theta(self, action):
        return gamma.rvs(self.alphas[action], scale=1 / self.betas[action], size=self.M)
"""
