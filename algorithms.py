import bandits
import numpy as np
from scipy.stats import beta

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
        action = int(np.argmax(theta_hats))
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        return action, reward
    
# ------------------------------------------------

class IDSAlgorithm(BaseAlgorithm):
    def __init__(self, bandit_env, M):
        super().__init__(bandit_env)
        self.M = M
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)
        self.thetas = None

    def reset_state(self):
        self.alphas = np.ones(self.K).astype(int)
        self.betas = np.ones(self.K).astype(int)
        self.thetas = self.__calculate_thetas()

    def single_step(self, t):
        action = None # TODO: continue implementation
        
        reward = self.bandit_env.sample(action)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward
        self.thetas[action] = self.__calculate_theta(action)
        return action, reward
    
    def __calculate_thetas(self):
        return np.array([self.__calculate_theta(action) for action in range(self.K)])
        
    def __calculate_theta(self, action):
        return np.random.beta(self.alphas[action], self.betas[action], size=self.M)