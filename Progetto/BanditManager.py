import UCBLearner
from Environment import *
from TSLearner import *
from Learner import *

class BanditManager:
    def __init__(self, id, T, n_experiments, opt, idx_opt, probabilities, customers):
        self.id = id
        self.env = None
        self.learner = None
        self.T = T
        self.n_experiments = n_experiments
        self.opt = opt
        self.idx_opt = idx_opt
        self.conversion_rates = [0., 0., 0., 0., 0., 0.]
        self.profit = [0., 0., 0., 0., 0., 0.]
        self.regrets, self.pseudo_regrets = np.zeros((n_experiments, T)), np.zeros((n_experiments, T))
        self.probabilities = probabilities
        self.n_customers = ((customers[0].number_of_customers + customers[1].number_of_customers + customers[2].number_of_customers) * 5)
        self.expected_payoffs = []
        self.deltas = 0.
        self.exp_clairvoyant = []
        self.best_clairvoyant = 0.
        self.regret = []
        self.campaigns_margin = [0., 0., 0., 0., 0., 0.]

    def initBandit(self, n_arms, campaigns, optimal_campaign_aggregate, level):
        if self.id == "ts":
            self.env = Environment(n_arms=n_arms, probabilities=self.probabilities)
            self.learner = TSLearner(n_arms=n_arms)
            for i in range(6):
                if i == 5:
                    self.learner.beta_parameters[i, 0] += campaigns[optimal_campaign_aggregate].aggregate_sales
                    self.learner.beta_parameters[i, 1] += campaigns[optimal_campaign_aggregate].aggregate_no_sales
                else:
                    self.learner.beta_parameters[i, 0] += campaigns[level + i].aggregate_sales
                    self.learner.beta_parameters[i, 1] += campaigns[level + i].aggregate_no_sales
        for i in range(6):
            if i == 6:
                self.campaigns_margin[i] = campaigns[optimal_campaign_aggregate].average_margin_for_sale
            else:
                self.campaigns_margin[i] = campaigns[level + i].average_margin_for_sale

    def executeBandit(self):
        if self.id == "ts":
            for t in range(0, self.T):
                pulled_arm = self.learner.pull_arm()
                reward = self.env.round(pulled_arm)
                self.learner.update(pulled_arm, reward)
        elif self.id == "ucb1":
            for e in range(0, self.n_experiments):
                self.regrets[e], self.pseudo_regrets[e], self.deltas, self.expected_payoffs = UCBLearner.UCB1(self.probabilities, self.T)

    def evaluateProfit(self, level, campaigns, optimal_campaign_aggregate):
        if self.id == "ts":
            for i in range(6):
                self.conversion_rates[i] = self.learner.beta_parameters[i, 0] / self.T
                if i == 5:
                    self.profit[i] = self.conversion_rates[i] * self.n_customers * campaigns[optimal_campaign_aggregate].average_margin_for_sale
                else:
                    self.profit[i] = self.conversion_rates[i] * self.n_customers * campaigns[level + i].average_margin_for_sale
        elif self.id == "ucb1":
            for i in range(6):
                if i == 5:
                    self.profit[i] = self.expected_payoffs[i] * self.n_customers * campaigns[optimal_campaign_aggregate].average_margin_for_sale
                else:
                    self.profit[i] = self.expected_payoffs[i] * self.n_customers * campaigns[level + i].average_margin_for_sale

    def clairvoyant_aggregate(self):
        # per ogni categoria calcolo il valore atteso di ogni arm aggregando le tre classi
        self.exp_clairvoyant = np.multiply(self.conversion_rates, self.campaigns_margin)
        self.best_clairvoyant = np.max(self.exp_clairvoyant)




