import cvxpy as cp
import numpy as np
from icecream import ic
from numpy.random import Generator
from numpy import float64
from numpy.typing import NDArray
from collections.abc import Callable
from scipy.optimize import minimize_scalar

from bandits import BaseBanditEnv, BernoulliAlignmentBanditEnv
from base_algorithms import BaseAlgorithm
from bayesian_state import BaseBayesianState, BetaBernoulliAlignmentState
from common import Action, Reward, SampleOutput


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

        # -------------------------------------------------------------------

        # action = self.__ids_action_scipy(R_star, est_rewards, mutual_infos)

        # -------------------------------------------------------------------

        pi = cp.Variable(self.K)
        objective = cp.Minimize(
            cp.quad_over_lin(
                (R_star - pi @ est_rewards) * 1000.0, pi @ mutual_infos * 1000000.0
            )
        )
        constraints = [0 <= pi, pi <= 1, cp.sum(pi) == 1.0]
        prob = cp.Problem(objective, constraints)

        argmin = False
        info_ratio: float64
        try:
            prob.solve(solver="ECOS")

            if prob.status != "optimal":
                ic(prob.status, t)

            raw_policy = np.maximum(pi.value, 0.0)
            policy = raw_policy / np.sum(raw_policy)
            action = self.rng.choice(self.K, 1, p=policy)[0]

            info_ratio = prob.value
        except Exception as e:
            argmin = True
            ic(e)
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

        return output.reward, action, {"argmin": argmin, "info_ratio": info_ratio}
        # return output.reward, action, None

    def __ids_action_scipy(
        self,
        R_star: float64,
        est_rewards: NDArray[float64],
        mutual_infos: NDArray[float64],
    ) -> Action:
        min_ratio: float | None = None
        min_pair: tuple[Action, Action] = (0, 0)
        q_min: float = 0.0
        for a1 in range(self.K - 1):
            for a2 in range(a1 + 1, self.K):

                def obj(q):
                    return (
                        R_star - q * est_rewards[a1] + (1 - q) * est_rewards[a2]
                    ) ** 2 / (q * mutual_infos[a1] + (1 - q) * mutual_infos[a2])

                result = minimize_scalar(obj, bounds=(0, 1), method="bounded")  # type: ignore
                info_ratio = result.fun
                q = result.x
                if min_ratio is None or info_ratio < min_ratio:
                    min_ratio = info_ratio
                    q_min = q
                    min_pair = (a1, a2)
        return Action(min_pair[0] if self.rng.random() < q_min else min_pair[1])

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
