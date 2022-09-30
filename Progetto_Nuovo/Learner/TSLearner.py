from Learner import *
import math

class TSLearner(Learner):
    def __init__(self, n_prices, n_products):
        super().__init__(n_prices)
        self.beta_parameters = np.ones((n_prices, n_products, 2))  # size of array is number of arms*2, and betas are initialized to one

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        # with argmax select the index of the maximum value and with the function
        # random beta we update the value of beta parameters
        return idx

    # update beta_parameters after pulled arm
    def update(self, price, arm, bought, total):
        self.t += 1
        self.update_observations(price, bought)
        self.beta_parameters[price, arm, 0] = self.beta_parameters[price, arm, 0] + bought
        # first parameter counts how many successes we have
        self.beta_parameters[price, arm, 1] = self.beta_parameters[price, arm, 1] + total - bought
        # second parameters does the opposite

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