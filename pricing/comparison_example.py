import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from Greedy_Learner import *

arms = 4
p = np.array([0.15, 0.1, 0.1, 0.35])   # parameters of the four bornoulli distrubution
opt = p[3]

T = 300

n_experiments = 100
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Environment(n_arms=arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=arms)
    gr_learner = Greedy_Learner(n_arms=arms)
    for t in range(0, T):

        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')  # it is the mean of the regret (the difference) per experiment where we have 300 rounds
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()





