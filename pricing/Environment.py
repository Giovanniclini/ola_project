import numpy as np


# we set the environment class given the number of arms and for each arm its probability
class Environment:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

# each pulled arm returns a random variable (given a max 1) depending on its probability
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
