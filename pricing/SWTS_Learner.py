from TS_Learner import TS_Learner
import numpy as np


class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm) # it counts from the end of the array
            cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.beta_parameters[arm, 0] = cum_rew + 1.0  # if n_samples > 0 it can be 0 or sum of 1
            self.beta_parameters[arm, 1] = n_samples - cum_rew + 1.0
