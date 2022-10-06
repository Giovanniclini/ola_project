from Progetto_Nuovo.Social_Influence.SocialInfluence import *


class Environment:
    def __init__(self, n_arms, customer_class, lambda_coeff, n_prod):
        self.n_arms = n_arms
        self.customer_class = customer_class
        self.lambda_coeff = lambda_coeff
        self.n_prod = n_prod

    def round(self, configuration, prices):
        prices_configuration = np.zeros(self.n_prod)
        for product in range(self.n_prod):
            prices_configuration[product] = prices[configuration[product]][product]
        simulator = SocialInfluence(self.lambda_coeff, self.customer_class, prices_configuration, self.n_prod)
        return simulator.reward, simulator.units_sold
