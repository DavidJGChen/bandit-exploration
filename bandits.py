
import numpy as np
# TODO: add types

class BaseArm:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        raise NotImplementedError

class BaseBanditEnv:
    def __init__(self, K):
        self.K = K
        self.arms = self.initialize_arms()
        self.optimal_action = np.argmax([arm.mean for arm in self.arms])
        self.optimal_mean = self.arms[self.optimal_action].mean
        
    def initialize_arms(self):
        raise NotImplementedError

    def sample(self, action):
        try:
            return self.arms[action].sample()
        except KeyError as e:
            print('Action invalid:', e)
            raise

class BernoulliBanditEnv(BaseBanditEnv):

    class BernoulliArm(BaseArm):
        def __init__(self, theta):
            self.theta = theta
            super().__init__(theta)

        def sample(self):
            return 1.0 if np.random.rand() < self.theta else 0.0

    def initialize_arms(self):
        thetas = np.random.uniform(0, 1, self.K)
        return [self.BernoulliArm(theta) for theta in thetas]

