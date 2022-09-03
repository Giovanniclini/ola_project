import numpy as np

class Learner:
    def __init__(self, n_arms):
        #number of arms = number of price configurations (6 each time)
        self.n_arms = n_arms
        #time step
        self.t = 0
        #collected rewards per arm
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        #global collected rewards
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)