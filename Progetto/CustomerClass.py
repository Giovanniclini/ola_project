import numpy as np

class CustomerClass():
    def __init__(self, class_id):
        self.id = class_id
        # assign a random number of customers
        self.number_of_customers = np.random.randint(10000, 100000)
        # assign a random value to alpha probabilities
        self.alpha_probabilities = np.random.dirichlet(np.ones(6), size=1)
        # assign a random value to the total number of purchasable product units
        self.max_number_of_purchases = np.random.randint(10000, 100000)
        # assign a random value to the reservation price for each product
        self.reservation_prices = [np.random.uniform(400., 1300.) for _ in range(5)]
        # keep track of number of units sold of each product
        self.units_purchased_for_each_product = np.zeros(5)
        # assign a random value to the social influence transition probability matrix to navigate among the products
        self.social_influence_transition_probability_matrix = np.random.uniform(0.01, 0.99, (5, 5))
        # assign empty global history
        self.global_history = []

    # a function that returns the total number of purchases among all the products
    def total_purchases(self):
        return np.sum(self.units_purchased_for_each_product, axis=1)