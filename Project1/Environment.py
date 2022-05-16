import numpy as np

class Environment:
    def __init__(self, n_arms, prices_probabilities, customer_class):
        self.n_arms = n_arms #20 arms
        self.prices_probabilities = prices_probabilities # probabilità associate ai prezzi
        self.customer_class = customer_class

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm]) #Bernulli distribution is modeled as a binomial with input 1
        return reward

    def select_class_cust(self, customer_class):
        class_cust = np.random.choice(customer_class)
        return class_cust


