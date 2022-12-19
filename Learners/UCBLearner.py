from Learners.Learner import *
import numpy as np
import math as m


class UCBLearner(Learner):
    def __init__(self, n_prices, n_products):
        super().__init__(n_prices, n_products)
        self.means = np.zeros((n_products, n_prices))
        self.upper_bound = np.matrix(np.ones((n_products, n_prices)) * np.inf)

    def pull_arm(self):
        pulled_config_indexes = np.argmax(self.means + self.upper_bound, axis=1)
        return pulled_config_indexes

    def update(self, pulled_config, bought, seen, tot_seen, tot_samples, reward):
        for product in range(len(pulled_config)):
            self.update_observations(pulled_config[product], bought[product], product, reward)
            self.means[product, pulled_config[product]] = (self.means[product, pulled_config[product]] *
                                                           seen[product, pulled_config[product]] + bought[product]) / \
                                                          tot_seen[product, pulled_config[product]]
            self.upper_bound[product, pulled_config[product]] = m.sqrt((2 * m.log10(tot_samples[product])) /
                                                                       tot_seen[product, pulled_config[product]])

    def conversion_rate(self):
        return self.means + self.upper_bound


    