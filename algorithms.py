import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar

DEBUG = False

"""
This kind of object-oriented inheritance is kind of bad, but it works for now.
"""


class BaseAlgorithm:
    def __init__(self, bandit_env):
        self.bandit_env = bandit_env
        self.K = self.bandit_env.K
        self.t = 0
        self.reward_history = None
        self.action_history = None
        self.T = None

    def run(self, T):
        self.t = 0
        self.T = T
        self.reward_history = np.zeros(T)
        self.action_history = np.zeros(T)
        self.reset_state()
        for t in range(T):
            action, reward = self.single_step(t)
            self.reward_history[t] = reward
            self.action_history[t] = action
        return self.action_history, self.reward_history

    def reset_state(self):
        pass

    def single_step(self, t):
        raise NotImplementedError


class BaseBayesianAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)

    def get_means(self):
        raise NotImplementedError

    def reset_bayesian_state(self):
        raise NotImplementedError

    def update_bayesian_posterior(self, action, reward):
        raise NotImplementedError


class BaseBetaBernoulliAlgorithm(BaseBayesianAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)
        self.alphas = None
        self.betas = None

    def reset_bayesian_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)

    def update_bayesian_posterior(self, action, reward):
        self.alphas[action] += reward
        self.betas[action] += 1 - reward

    def get_means(self):
        return self.alphas / (self.alphas + self.betas)


# ------------------------------------------------


class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)

    def single_step(self, t):
        action = np.random.choice(self.K)
        return action, self.bandit_env.sample(action)


# ------------------------------------------------


class EpsilonGreedyAlgorithm(BaseBetaBernoulliAlgorithm):
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


class BayesUCBAlgorithm(BaseBetaBernoulliAlgorithm):
    def __init__(self, bandit_env, c):
        self.c = c
        super().__init__(bandit_env)
        self.inv_log_factor = None

    def reset_state(self):
        self.reset_bayesian_state()
        self.inv_log_factor = 1 / np.power(np.log(self.T), self.c)

    def single_step(self, t):
        quantile = 1 - (self.inv_log_factor / (t + 1))
        quantiles = beta.ppf(quantile, self.alphas, self.betas)

        action = np.argmax(quantiles)

        reward = self.bandit_env.sample(action)

        self.update_bayesian_posterior(action, reward)

        return action, reward


# ------------------------------------------------


class ThompsonSamplingAlgorithm(BaseBetaBernoulliAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)

    def reset_state(self):
        self.reset_bayesian_state()

    def single_step(self, t):
        theta_hats = np.random.beta(self.alphas, self.betas, size=self.K)
        action = np.argmax(theta_hats)

        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        return action, reward


# ------------------------------------------------


class VarianceIDSAlgorithm(BaseBetaBernoulliAlgorithm):
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
            np.array([partitioned_thetas[a].shape[1] for a in range(self.K)]) / self.M
        )

        # calculate est. mean of actions conditioned on action being optimal.
        # shape = (K, K). Indexing once (cond_mu[a_star]) gives us an array
        # of means of all arms conditioned on a_star being optimal.
        cond_mu = np.nan_to_num(
            np.array(
                [
                    np.mean(thetas, axis=1) if thetas.shape[1] > 0 else np.zeros(self.K)
                    for thetas in partitioned_thetas
                ]
            )
        )

        # estimate expected value of optimal action
        rho = np.sum([p_optimal[a] * cond_mu[a, a] for a in range(self.K)])
        delta = rho - mu

        # calculate variance term for each arm as an expectation
        variance = np.sum(
            np.array([p_optimal[a] * (cond_mu[a] - mu) ** 2 for a in range(self.K)]),
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
            print(f"times chosen:\t\t\t{self.alphas + self.betas - 2}")
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
        return np.random.beta(self.alphas[action], self.betas[action], size=self.M)
