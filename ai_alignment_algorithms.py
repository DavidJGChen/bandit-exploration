import cvxpy as cp
import numpy as np
from icecream import ic
from numpy import float64
from numpy.typing import NDArray

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
        M: int,
        use_argmin: bool = False,
    ):
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
        # Not the best way to do this, but need this hack for now.
        super().__init__(bandit_env, bayesian_state)

    def reset_algorithm_state(self) -> None:
        self.phi_samples, self.theta_samples = self.__calculate_params()

    def single_step(self, t: int) -> tuple[Reward, Action]:
        est_rewards = self.bayesian_state.get_means()
        mutual_infos = self.bayesian_state.get_mutual_infos()

        est_rewards_from_samples = self.phi_samples * self.theta_samples + (
            1 - self.phi_samples
        ) * (1 - self.theta_samples)

        R_star = np.mean(np.max(est_rewards_from_samples, axis=1))

        pi = cp.Variable(self.K)
        objective = cp.Minimize(
            cp.quad_over_lin(R_star - pi @ est_rewards, pi @ mutual_infos)
        )
        constraints = [0 <= pi, pi <= 1, cp.sum(pi) == 1.0]
        prob = cp.Problem(objective, constraints)
        action: Action
        try:
            prob.solve()
            raw_policy = np.maximum(pi.value, 0.0)
            policy = raw_policy / np.sum(raw_policy)
            action = np.random.choice(self.K, 1, p=policy)[0]
        except Exception as e:
            ic(e)
            action = np.argmin(np.square(R_star - est_rewards) / mutual_infos)

        output: SampleOutput = self.bandit_env.sample(action)

        self.bayesian_state.update_posterior_with_outcomes(output, action)

        index: Action
        if self.bandit_env.is_env(action):
            index = action
            self.phi_samples[index] = self.__calculate_param(action)
        else:
            index = action - self.bandit_env.K_env
            self.theta_samples[index] = self.__calculate_param(action)

        return output.reward, action

    def __calculate_params(self) -> NDArray[float64]:
        all_params = np.array(
            [self.__calculate_param(action) for action in range(self.K)]
        )
        return all_params[: self.bandit_env.K_env], all_params[self.bandit_env.K_env :]

    def __calculate_param(self, action: Action) -> NDArray[float64]:
        return self.bayesian_state.get_sample_for_action(action, self.M)
