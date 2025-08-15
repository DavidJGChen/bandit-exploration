import cvxpy as cp
import numpy as np
from icecream import ic
from numpy.random import Generator
from numpy import float64
from numpy.typing import NDArray
from collections.abc import Callable
from scipy.optimize import minimize_scalar

from .bandits import BaseBanditEnv, BernoulliAlignmentBanditEnv
from .base_algorithms import BaseAlgorithm
from .bayesian_state import BaseBayesianState, BetaBernoulliAlignmentState
from .common import Action, Reward, SampleOutput


class IDSAlignmentAlgorithm(BaseAlgorithm):
    phi_samples: NDArray[float64]
    theta_samples: NDArray[float64]
    mutual_infos_phi: NDArray[float64]
    mutual_infos_theta: NDArray[float64]

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        bayesian_state: BaseBayesianState,
        rng: Generator,
        M: int,
        use_argmin: bool = False,
    ):
        # Not the best way to do this, but need this hack for now.
        if type(bandit_env) is not BernoulliAlignmentBanditEnv:
            raise TypeError(
                f"Algorithm is only valid with {BernoulliAlignmentBanditEnv}"
            )
        if type(bayesian_state) is not BetaBernoulliAlignmentState:
            raise TypeError(
                f"Algorithm is only valid with {BetaBernoulliAlignmentState}"
            )
        self.M = M  # number of samples for MCMC
        self.use_argmin = use_argmin
        super().__init__(bandit_env, bayesian_state, rng)

    def reset_algorithm_state(self) -> None:
        self.phi_samples, self.theta_samples = self.__calculate_params()

    def single_step(self, t: int) -> tuple[Reward, Action, dict | None]:
        est_rewards = self.bayesian_state.get_means()
        mutual_infos = self.bayesian_state.get_mutual_infos()

        est_rewards_from_samples = self.phi_samples * self.theta_samples + (
            1 - self.phi_samples
        ) * (1 - self.theta_samples)

        R_star = np.mean(np.max(est_rewards_from_samples, axis=1))

        action: Action
        info_ratio: float64

        # -------------------------------------------------------------------

        argmin_fallback = False
        if not self.use_argmin:
            pi = cp.Variable(self.K)
            objective = cp.Minimize(
                cp.quad_over_lin(
                    (R_star - pi @ est_rewards) * 1000.0, pi @ mutual_infos * 1000000.0
                )
            )
            constraints = [0 <= pi, pi <= 1, cp.sum(pi) == 1.0]
            prob = cp.Problem(objective, constraints)

            try:
                prob.solve(solver="CLARABEL")

                if prob.status != "optimal":
                    ic(prob.status, t)

                raw_policy = np.maximum(pi.value, 0.0)
                policy = raw_policy / np.sum(raw_policy)
                action = self.rng.choice(self.K, 1, p=policy)[0]

                info_ratio = prob.value
            except Exception as e:
                argmin_fallback = True
                ic(e)
        else:
            argmin_fallback = True

        # -------------------------------------------------------------------

        if argmin_fallback:
            info_ratios = np.square(R_star - est_rewards) / mutual_infos
            action = np.argmin(info_ratios)
            info_ratio = info_ratios[action]

        # -------------------------------------------------------------------

        output: SampleOutput = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior_with_outcomes(output, action)

        index: Action
        if self.bandit_env.is_env(action):
            index = action
            self.phi_samples[index] = self.__calculate_param(action)
        else:
            index = action - self.bandit_env.K_env
            self.theta_samples[index] = self.__calculate_param(action)

        return (
            output.reward,
            action,
            {"argmin": argmin_fallback, "info_ratio": info_ratio},
        )

    def __calculate_params(self) -> NDArray[float64]:
        all_params = np.array(
            [self.__calculate_param(action) for action in range(self.K)]
        )
        return all_params[: self.bandit_env.K_env], all_params[self.bandit_env.K_env :]

    def __calculate_param(self, action: Action) -> NDArray[float64]:
        return self.bayesian_state.get_sample_for_action(action, self.M)


# ------------------------------------------------


class EpsilonThompsonSamplingAlignmentAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        bayesian_state: BaseBayesianState,
        rng: Generator,
        epsilon_func: Callable[[int], float],
    ):
        # Not the best way to do this, but need this hack for now.
        if type(bandit_env) is not BernoulliAlignmentBanditEnv:
            raise TypeError(
                f"Algorithm is only valid with {BernoulliAlignmentBanditEnv}"
            )
        if type(bayesian_state) is not BetaBernoulliAlignmentState:
            raise TypeError(
                f"Algorithm is only valid with {BetaBernoulliAlignmentState}"
            )
        super().__init__(bandit_env, bayesian_state, rng)
        self.epsilon_func = epsilon_func

    def reset_algorithm_state(self) -> None: ...

    def single_step(self, t: int) -> tuple[Reward, Action, dict | None]:
        action: Action
        if t < self.K:
            action = Action(t)
        else:
            est_means = self.bayesian_state.get_samples()
            env_action = np.argmax(est_means)
            if self.rng.random() < self.epsilon_func(t):
                # select corresponding human action
                action = env_action + self.bandit_env.K_env
            else:
                action = env_action

        output: SampleOutput = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior_with_outcomes(output, action)

        return output.reward, action, None
