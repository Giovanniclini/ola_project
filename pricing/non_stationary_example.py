import numpy as np
from matplotlib import pyplot as plt
from Non_Stationary_Environment import Non_Stationary_Environment
from SWTS_Learner import SWTS_Learner
from TS_Learner import TS_Learner


n_arms = 4
n_phases = 4
p = np.array([[0.15, 0.1, 0.2, 0.35],
             [0.45, 0.21, 0.2, 0.35],
             [0.1, 0.1, 0.5, 0.15],
             [0.1, 0.21, 0.1, 0.15]])
T = 500
phases_len = int(T/n_phases)
n_experiments = 100
ts_reward_per_experiment = []
swts_reward_per_experiment = []
windows_size = int(T**0.5)

if __name__ == '__main__':
    for e in range(n_experiments):
        ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
        ts_learner = TS_Learner(n_arms=n_arms)

        swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
        swts_learner = SWTS_Learner(n_arms=n_arms, window_size=windows_size)

        for t in range(T):
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            pulled_arm = swts_learner.pull_arm()
            reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, reward)

        ts_reward_per_experiment.append(ts_learner.collected_rewards)
        swts_reward_per_experiment.append(swts_learner.collected_rewards)

    ts_istantaneous_regret = np.zeros(T)
    swts_istantaneous_regret = np.zeros(T)
    opt_per_phases = p.max(axis=1)  # it takes the best mean between each arms per phase
    optimum_per_round = np.zeros(T)

    for i in range(n_phases):
        t_index = range(i*phases_len, (i+1)*phases_len)  # it takes the list of index of T for a certain phase
        optimum_per_round[t_index] = opt_per_phases[i]  # it gives to a certain phase (all indexes) the optimal mean
        ts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_reward_per_experiment, axis=0)[t_index]
        swts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_reward_per_experiment, axis=0)[t_index]

    plt.figure(0)
    plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
    plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
    plt.plot(optimum_per_round, 'k--')  # tratteggiato
    plt.legend(['TS', 'SWTS', 'Optimum'])
    plt.ylabel('Reward')
    plt.xlabel('t')
    plt.show()

    plt.figure(1)
    plt.plot(np.cumsum(ts_istantaneous_regret), 'r')
    plt.plot(np.cumsum(swts_istantaneous_regret), 'b')
    plt.legend(['TS', 'SWTS'])
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.show()














