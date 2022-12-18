from Social_Influence.SocialInfluence import *


class ContextualEnvironment:
    def __init__(self, n_arms, lambda_coeff, n_prod):
        self.n_arms = n_arms
        self.customer_class = None
        self.lambda_coeff = lambda_coeff
        self.n_prod = n_prod

    def round(self, configuration, prices, alpha_ratios, item_sold_mean):
        prices_configuration = np.zeros(self.n_prod)
        for product in range(self.n_prod):
            prices_configuration[product] = prices[product][configuration[product]]
        simulator = SocialInfluence(self.lambda_coeff, alpha_ratios, item_sold_mean, self.customer_class,
                                    prices_configuration, self.n_prod, configuration)
        simulator.simulation()
        return simulator.reward, simulator.bought, simulator.actual_users

    def select_costumer_class(self, customer_class):
        self.customer_class = customer_class

