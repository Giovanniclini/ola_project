from Progetto_Nuovo.Learners.Learner import *
import numpy as np
import math as m


class UCBLearner(Learner):
    def __init__(self, n_prices, n_products):
        super().__init__(n_prices, n_products)
        self.means = np.zeros((n_prices, n_products))
        self.upper_bound = np.matrix(np.ones((n_prices, n_products)) * np.inf)

    def pull_arm(self):
        pulled_config_indexes = np.argmax(self.means + self.upper_bound)
        return pulled_config_indexes

    def update(self, pulled_config, bought, seen, tot_seen, tot_samples):
        for product in range(len(pulled_config)):
            self.update_observations(pulled_config[product], bought[product], product)
            self.means[product, pulled_config[product]] = (self.means[product, pulled_config[product]] * seen + bought[product]) / tot_seen
            self.upper_bound[product, pulled_config[product]] = m.sqrt((2 * m.log10(tot_samples)) / seen)

    def conversion_rate(self):
        return self.means + self.upper_bound


    def UCB1(exp_reward, T):
        n_arms = len(exp_reward)
        opt = np.max(exp_reward)
        idx_opt = np.argmax(exp_reward)
        deltas = opt - exp_reward
        deltas = np.array([delta for delta in deltas if delta > 0])

        ucb1_criterion = np.zeros(n_arms)
        expected_payoffs = np.zeros(n_arms)
        number_of_pulls = np.zeros(n_arms)

        regret = np.array([])
        pseudo_regret = np.array([])

        for t in range(1, T + 1):
            # Select an arm
            if t < n_arms:
                pulled_arm = t  # round robin for the first n steps
            else:
                idxs = np.argwhere(ucb1_criterion == ucb1_criterion.max()).reshape(-1)  # there can be more arms with max value
                pulled_arm = np.random.choice(idxs)

            # Pull an arm
            reward = np.random.binomial(1, exp_reward[pulled_arm])
            if pulled_arm != idx_opt:
                reward_opt = np.random.binomial(1, exp_reward[idx_opt])
            else:
                reward_opt = reward

            # Update UCB1
            number_of_pulls[pulled_arm] = number_of_pulls[pulled_arm] + 1
            expected_payoffs[pulled_arm] = ((expected_payoffs[pulled_arm] * (number_of_pulls[pulled_arm] - 1.0) + reward) /
                                            number_of_pulls[pulled_arm])  # update sample mean for the selected arm
            for k in range(0, n_arms):
                ucb1_criterion[k] = expected_payoffs[k] + np.sqrt(2 * np.log(t) / number_of_pulls[k])

            # Store results
            regret = np.append(regret, reward_opt - reward)
            pseudo_regret = np.append(pseudo_regret, opt - exp_reward[pulled_arm])
        return regret, pseudo_regret, deltas, expected_payoffs