from Environment import Environment
import numpy as np


class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):  # in this case probabilities is a matrix for each arm in a certain phase becuase the mean changes
        super().__init__(n_arms, probabilities)  # the constructor of the parent class
        self.t = 0
        n_phases = len(probabilities)
        self.phase_size = horizon/n_phases

    def round(self, pulled_arm):  # we compute in which phase we are, and we get the mean (from probabilities) of the pulled arm in a certain phase
        current_phase = int(self.t/self.phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)  # we need the mean for the binomial distribution
        self.t += 1
        return reward


