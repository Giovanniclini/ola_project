import numpy as np
import matplotlib.pyplot as plt
from LinearMabEnvironment import LinearMabEnvirnoment
from LinearUCBLearner import LinUCBLearner

#import LinearMabEnvironment from LinearMabEnvironment
#import LinUCBLearner from LinearUCBLearner

if __name__ == '__main__':
    n_arms = 10
    T = 1000
    n_experiments = 100
    lin_ucb_rewards_per_experiments = []

    env = LinearMabEnvirnoment(n_arms=n_arms, dim=10)

    for e in range(0, n_experiments):
        lin_ucb_learner = LinUCBLearner(arms_features=env.arms_features)
        for t in range(0, T):
            pulled_arm = lin_ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            lin_ucb_learner.update(pulled_arm, reward)
        lin_ucb_rewards_per_experiments.append(lin_ucb_learner.collected_rewards)


    opt = env.opt()
    plt.figure(0)
    plt.plot(np.cumsum(np.mean(opt-lin_ucb_rewards_per_experiments, axis=0)),'r')
    plt.show()