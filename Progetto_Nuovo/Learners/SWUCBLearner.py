from Progetto_Nuovo.Learners.Learner import *
import numpy as np
import math as m


class SWUCBLearner(Learner):
    def __init__(self, n_prices, n_products):
        super().__init__(n_prices, n_products)
        self.means = np.zeros((n_products, n_prices))
        self.upper_bound = np.matrix(np.ones((n_products, n_prices)) * np.inf)

    def pull_arm(self):
        pulled_config_indexes = np.argmax(self.means + self.upper_bound, axis=1)
        return pulled_config_indexes

    def update(self, pulled_config, bought, seen, tot_seen_window, tot_samples, tot_samples_round, reward):
        self.check_end_window()
        for product in range(len(pulled_config)):
            self.update_observations(pulled_config[product], bought[product], product, reward)
            self.means[product, pulled_config[product]] = (self.means[product, pulled_config[product]] *
                                                           seen[product, pulled_config[product]] + bought[product]) / \
                                                          tot_seen_window[product, pulled_config[product]]
            self.upper_bound[product, pulled_config[product]] = m.sqrt(
                (2 * m.log10(min(tot_samples[product], tot_samples_round[product])) /
                 tot_seen_window[product, pulled_config[product]]))
        self.t += 1

    def check_end_window(self):
        if self.t % 25 == 0:
            self.means = np.zeros((self.n_products, self.n_arms))
            self.upper_bound = np.matrix(np.ones((self.n_products, self.n_arms)) * np.inf)

    def conversion_rate(self):
        return self.means + self.upper_bound


