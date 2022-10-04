from Social_Influence.SocialInfluence import *


class Environment:
    def __init__(self, n_arms, customer_class, lambda_coeff, n_prod):
        self.n_arms = n_arms
        self.customer_class = customer_class
        self.lambda_coeff = lambda_coeff
        self.n_prod = n_prod

    def round(self, configuration):
        # TODO: convertire indici dei prezzi in valori
        simulator = SocialInfluence(self.lambda_coeff, self.customer_class, configuration, self.n_prod)
        return simulator.reward, simulator.units_sold
