from termcolor import colored
import numpy as np
from BanditManager import *


class OptimizationAlgorithm:
    def __init__(self, aggregate, learners, user_class, optimal_campaign, customers, campaigns):
        self.aggregate = aggregate
        self.learners = learners
        self.profit_increase = 0.
        self.check_increase = True
        self.user_class = user_class
        self.optimal_campaign = optimal_campaign
        self.customers = customers
        self.campaigns = campaigns
        self.profit = [0., 0., 0., 0., 0., 0]
        self.probabilities = [0., 0., 0., 0., 0., 0.]

    def evaluate_aggregate_conversion_rate(self, customer_class, campaign):
        if customer_class.global_history[campaign] is not None:
            # for each history (relative to one costumer) in the global history
            for history in customer_class.global_history[campaign]:
                # check if the current step in global history is not null (just to be sure)
                if history is not None:
                    # for each step in the history
                    for history_step in history:
                        # if contains a 1, i.e., a product was bought
                        if 1 in history_step:
                            prod = history_step.index(1)
                            # increase the global number of sales of the customer class involved
                            customer_class.number_of_clicks[campaign] += 1
                            customer_class.number_of_clicks_per_product[campaign][prod] += 1
        for prod in range(5):
            customer_class.conversion_rate_per_product[campaign][prod] = customer_class.number_of_clicks_per_product[
                                                                    campaign][prod] / customer_class.number_of_customers
        print(customer_class.conversion_rate_per_product[campaign])


    def create_bandit(self, bandit_id, level):
        if bandit_id == 0:
            bandit_id = "ts"
        elif bandit_id == 1:
            bandit_id = "ucb1"
        self.learners.append(BanditManager(id=bandit_id, T=self.customers.number_of_customers, n_experiments=1,
                                           opt=[1., 1., 1., 1., 1., 1.], idx_opt=self.optimal_campaign,
                                           probabilities=self.probabilities,
                                           n_customers=self.customers.number_of_customers))
        self.learners[-1].initBandit(n_arms=6, campaigns=self.campaigns,
                                     optimal_campaign_aggregate=self.optimal_campaign, level=level)

    def execute_bandit(self, bandit_index, level):
        self.learners[bandit_index].executeBandit()
        self.learners[bandit_index].evaluateProfit(level=level, campaigns=self.campaigns,
                                                   optimal_campaign_aggregate=self.optimal_campaign)
        self.learners[bandit_index].clairvoyant_aggregate()

    def assign_probabilities(self, i, idx):
        self.evaluate_aggregate_conversion_rate(self.customers, idx)
        for prod in range(5):
            self.probabilities[i] = self.customers.conversion_rate_per_product[idx][prod]

    def assign_values(self, level, bandit_index):
        configurations = [0, 0, 0, 0, 0, 0]
        self.profit = [0., 0., 0., 0., 0., 0]
        for i in range(6):
            idx = level + i
            if i == 0:
                idx = self.optimal_campaign
            configurations[i] = 0
            configurations[i] = np.copy(self.campaigns[idx].configuration)
            if bandit_index != -1:
                self.assign_probabilities(i, idx)
            else:
                self.evaluate_profit(i, idx)
        return configurations

    def evaluate_profit(self, i, idx):
        for prod in range(5):
            self.profit[i] += (self.customers[self.user_class].units_purchased_per_product[idx][prod] * self.campaigns[idx].average_margin_for_price_in_configuration[prod])

    def execute(self, bandit_index):
        for level in range(0, 15, 5):
            if not self.check_increase:
                return
            print(colored('\n\n---------------------------- LEVEL {0} ----------------------------', 'blue',
                          attrs=['bold']).format(int(level / 5)))
            configurations = self.assign_values(level, bandit_index)
            print('\nInitial optimal configuration is: {0}'.format(configurations[0]))
            if bandit_index != -1 and self.aggregate:
                self.create_bandit(bandit_index, level)
                self.execute_bandit(bandit_index, level)
                print(self.learners[bandit_index].profit)
                self.profit = self.learners[bandit_index].profit
            max_profit_idx = 0
            possible_optimal = self.profit.index(max(self.profit))
            self.profit_increase = (self.profit[possible_optimal] / self.profit[max_profit_idx]) - 1
            if possible_optimal != max_profit_idx and self.profit_increase > 0.:
                max_profit_idx = possible_optimal
                self.optimal_campaign = level + max_profit_idx
                print(colored('Better solution found. Current marginal increase {0:.2%}', 'green', attrs=['bold']).format(self.profit_increase))
                print('The best configuration is number {0}: {1}  '.format(self.optimal_campaign, self.campaigns[
                    self.optimal_campaign].configuration))
            else:
                self.check_increase = False
                print(colored('No better solution found. Current marginal increase {0:.2%}', 'red', attrs=['bold']).format(self.profit_increase))
                print('The best configuration is number {0}: {1}  '.format(self.optimal_campaign, self.campaigns[
                    self.optimal_campaign].configuration))

