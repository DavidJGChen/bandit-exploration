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
            print("Action invalid:", e)
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


class GaussianBanditEnv(BaseBanditEnv):

    class GaussianArm(BaseArm):
        def __init__(self, mu, eta):
            self.mu = mu
            self.eta = eta
            super().__init__(mu)

        def sample(self):
            return np.random.normal(self.mu, self.eta)

    def initialize_arms(self):
        mus = np.random.normal(0, 1, self.K)
        etas = np.ones(self.K)
        return [self.GaussianArm(mu, eta) for mu, eta in zip(mus, etas)]


class PoissonBanditEnv(BaseBanditEnv):

    class PoissonArm(BaseArm):
        def __init__(self, rate):
            self.rate = rate
            super().__init__(rate)

        def sample(self):
            return np.random.poisson(self.rate)

    def initialize_arms(self):
        rates = np.random.exponential(1.0, size=self.K)
        return [self.PoissonArm(rate) for rate in rates]


class LinearBanditEnv(BaseBanditEnv):
    def __init__(self, K, d):
        self.d = d
        super().__init__(K)

    class LinearArm(BaseArm):
        def __init__(self, feature, theta):
            self.feature = feature
            super().__init__(feature @ theta)

        def sample(self):
            return self.mean + np.random.standard_normal()

    def initialize_arms(self):
        mean_vec = np.zeros(self.d)
        covariance = 10 * np.eye(self.d)
        feature_radius = 1 / np.sqrt(5)
        self.theta = np.random.multivariate_normal(mean_vec, covariance)
        features = np.random.uniform(
            -feature_radius, feature_radius, size=(self.K, self.d)
        )
        self.phi = features
        return [self.LinearArm(feature, self.theta) for feature in features]
