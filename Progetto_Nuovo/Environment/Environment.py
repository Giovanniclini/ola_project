from Progetto_Nuovo.Social_Influence.SocialInfluence import *


class Environment:
    def __init__(self, n_arms, probabilities, customer_class, lambda_coeff, n_prod):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.customer_class = customer_class
        self.lambda_coeff = lambda_coeff
        self.n_prod = n_prod

    def round(self, configuration):
        simulator = SocialInfluence(self.customer_class.number_of_customers, self.customer_class.alpha_probabilities,
                                    self.lambda_coeff, self.customer_class.social_influence_transition_probability_matrix,
                                    self.customer_class, configuration, self.n_prod)
        return simulator.reward
