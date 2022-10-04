from Learner import *
import math


class TSLearner(Learner):
    def __init__(self, n_prices, n_products):
        super().__init__(n_prices, n_products)
        self.beta_parameters = np.ones((n_products, n_prices, 2))

    def pull_arm(self):
        pulled_config_indxes = np.argmax(np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1]), axis=1)
        return pulled_config_indxes

    def update(self, pulled_config, bought, total):
        # increase time
        self.t += 1
        for product in range(len(pulled_config)):
            self.update_observations(pulled_config[product], bought[product], product)
            self.beta_parameters[product, pulled_config[product], 0] = self.beta_parameters[product, pulled_config[product], 0] + bought[product]
            self.beta_parameters[product, pulled_config[product], 1] = self.beta_parameters[product, pulled_config[product], 1] + total - bought[product]

    def succ_prob_arm(self, arm):
        arm_successes = self.beta_parameters[arm][0]
        arm_failures = self.beta_parameters[arm][1]
        return arm_successes / (arm_successes + arm_failures)

    def expected_value(self, arm, candidate_value):
        arm_prob_success = self.succ_prob_arm(arm)
        expected_value = arm_prob_success * candidate_value
        return expected_value

    def best_arm(self, candidates_values):
        best = 0
        best_value = self.expected_value(0,candidates_values[0])
        for i in range(self.n_arms):
            if self.expected_value(i, candidates_values[i]) > best_value:
                best = i
                bast_value = self.expected_value(i, candidates_values[i])
        return best, best_value

    def best_arm_lower_bound(self, candidates_values):
        best_arm = self.best_arm(candidates_values)
        exp_value = self.expected_value(best_arm, candidates_values[best_arm])
        succ_arm = self.prob_succ_arm(best_arm)
        alfa_best_arm = self.beta_parameters[best_arm][0]
        beta_best_arm = self.beta_parameters[best_arm][1]
        minus = -pow(-math.log(succ_arm * (1 - succ_arm)) / (2 * (alfa_best_arm + beta_best_arm)), 0.5)
        # TODO: sostituisci (alfa_best_arm + beta_best_arm) con il self.t
        # NEW!
        # minus = -pow(-math.log(succ_arm * (1-succ_arm)) / (2 * (self.t)), 0.5)
        return exp_value - minus