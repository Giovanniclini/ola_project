from Progetto_Nuovo.Learners.Learner import *
import numpy as np
import math as m


class CDUCBLearner(Learner):
    def __init__(self, n_prices, n_products, range=6, delta=0.2):
        super().__init__(n_prices, n_products)
        self.means = np.zeros((n_products, n_prices))
        self.upper_bound = np.matrix(np.ones((n_products, n_prices)) * np.inf)
        self.phase = 0
        self.phase_sizes = []
        self.range = range
        self.delta = delta
        self.conversion_rates = np.zeros((n_products, n_prices))

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
            self.conversion_rates[product][pulled_config[product]].append(bought[product]/seen[product, pulled_config[product]])

    def conversion_rate(self):
        return self.means + self.upper_bound

    def detect_change(self, pulled_config):
        if self.t > 12:
            change_detected = False
            if not change_detected:
                for product in range(len(pulled_config)):
                    last_mean = np.mean(self.conversion_rates[product][pulled_config[product]][:-range])
                    new_mean = np.mean(self.conversion_rates[product][pulled_config[product]][-range:])

                    if last_mean/new_mean >= self.delta or new_mean/last_mean >= self.delta:
                        change_detected = True
                        self.phase += 1
                        self.means = np.zeros((n_products, n_prices))
                        self.upper_bound = np.matrix(np.ones((n_products, n_prices)) * np.inf)
                        self.phase_sizes.append(self.t)



