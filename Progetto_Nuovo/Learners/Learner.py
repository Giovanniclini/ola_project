import numpy as np


class Learner:
    def __init__(self, n_arms, n_products):
        # number of arms = number of price configurations (6 each time)
        self.n_arms = n_arms
        self.n_products = n_products
        # time step
        self.t = 0
        # collected rewards per arm
        self.rewards_per_arm = x = [[[] for i in range(n_arms)] for j in range(n_products)]
        # global collected rewards
        self.collected_rewards = [np.array([]) for i in range(n_products)]

    def update_observations(self, pulled_arm, reward, product):
        self.rewards_per_arm[pulled_arm][product].append(reward)
        self.collected_rewards[product] = np.append(self.collected_rewards, reward)