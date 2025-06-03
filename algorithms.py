import bandits
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import cvxpy as cvx

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
    
# ------------------------------------------------

class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)

    def single_step(self, t):
        action = np.random.choice(self.K)
        return action, self.bandit_env.sample(action)
    
# ------------------------------------------------
    
class EpsilonGreedyAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env, epsilon_func):
        super().__init__(bandit_env)
        self.epsilon_func = epsilon_func
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)

    def reset_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)

    def single_step(self, t):
        if np.random.rand() < self.epsilon_func(t):
            action = np.random.choice(self.K)
        else:
            theta_hats = np.divide(self.alphas, self.alphas + self.betas)
            action = int(np.argmax(theta_hats))
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        return action, reward
    
# ------------------------------------------------
    
class BayesUCBAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env, c):
        self.c = c
        super().__init__(bandit_env)
        self.alphas = None
        self.betas = None
        self.inv_log_factor = None

    def reset_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)
        self.inv_log_factor = 1 / np.power(np.log(self.T), self.c)

    def single_step(self, t):
        quantile = 1 - (self.inv_log_factor / (t + 1))
        quantiles = beta.ppf(quantile, self.alphas, self.betas)

        action = np.argmax(quantiles)
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        return action, reward
    
# ------------------------------------------------

class ThompsonSamplingAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env):
        super().__init__(bandit_env)
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)

    def reset_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)

    def single_step(self, t):
        theta_hats = np.random.beta(self.alphas, self.betas, size=self.K)
        action = np.argmax(theta_hats)
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        return action, reward
    
# ------------------------------------------------

class VarianceIDSAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env, M):
        super().__init__(bandit_env)
        self.M = M # number of samples for MCMC
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)
        self.thetas = None

    def reset_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)
        self.thetas = self.__calculate_thetas()

    def single_step(self, t):
        # estimated means of action parameters
        # mu = np.mean(self.thetas, axis=1)
        mu = self.alphas / (self.alphas + self.betas)

        max_action = np.argmax(self.thetas, axis=0) # max action in each sample

        # partition the sampled thetas based on which arm is optimal
        partitioned_thetas = [self.thetas[:, np.where(max_action == action)[0]] for action in range(self.K)]

        p_optimal = np.array([partitioned_thetas[action].shape[1] for action in range(self.K)]) / self.M

        # calculate estimated mean of all actions conditioned on arms being optimal
        # shape = (K, K), where first dimension represents the conditional optimal arm.
        cond_mu = np.nan_to_num(np.array([np.mean(thetas, axis=1) for thetas in partitioned_thetas]))

        # estimate expected value of optimal action
        rho_star = np.sum([p_optimal[action] * cond_mu[action, action] for action in range(self.K)])
        delta = rho_star - mu

        variance = np.sum(np.array([p_optimal[action] * (cond_mu[action] - mu)**2 for action in range(self.K)]), axis=0)

        # print(f"\n--------round {t}--------")
        # print(f"mu:\t\t\t\t{mu}")
        # print(f"times chosen:\t\t\t{self.alphas + self.betas - 2}")
        # for action in range(self.K):
        #     print(f"--Assume action {action} is optimal--")
        #     print(f"estimated mean of action {action}:\t{mu[action]}")
        #     print(f"p_optimal({action}):\t\t\t{p_optimal[action]}")
        #     print(f"mean vector given {action} optimal:\t{cond_mu[action]}")
        # print(f"---more stats---")
        # print(f"rho_star:\t\t\t{rho_star}")
        # print(f"delta vector:\t\t\t{delta}")
        # print(f"variance:\t\t\t{variance}")
        # print(f"info ratio:\t\t\t{delta**2 / variance}")

        action = self.__ids_action(delta, variance)
        # action = np.random.choice(self.K)
        # print(f"action chosen:\t\t\t{action}")
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward
        self.thetas[action] = self.__calculate_theta(action)

        return action, reward
    
    def __ids_action(self, delta, v):
        min_ratio = None
        min_pair = None
        q_min = None
        for a1 in range(self.K - 1):
            for a2 in range(a1 + 1, self.K):
                obj = lambda q: (q * delta[a1] + (1 - q) * delta[a2])**2 / (q * v[a1] + (1 - q) * v[a2])
                result = minimize_scalar(obj, bounds=(0,1), method='bounded')
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