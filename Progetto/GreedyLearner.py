from Learner import *


class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # at each time t, computes the estimation of the reward
        self.expected_rewards = np.zeros(n_arms)

    # selects the arm to pull, by maximizing the expected rewards array
    def pull_arm(self):
        if self.t < self.n_arms:   # we are saying that at each round we pull arm with index +1
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # indexes of the arms with the maximum reward
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t - 1) + reward) / self.t
        # the update of the reward in just an average
