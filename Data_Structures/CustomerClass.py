import random

import numpy as np


class CustomerClass:
    def __init__(self):
        self.conversion_rates = np.zeros((5, 4))
        # assign a random number of customers
        self.number_of_customers = 0
        # assign a random value to alpha probabilities
        self.alpha_probabilities = np.zeros(6)
        # assign a mean of number of item sold
        self.item_sold_mean = np.zeros((5, 4))
        # assign a random value to the reservation price for each product
        self.reservation_prices = [0 for _ in range(5)]
        # keep track of units clicked starting from an edge
        self.units_clicked_starting_from_a_primary = np.zeros((5, 5))
        # assign graph table empty, it seems that must replace the social_influence_transition_probability_matrix
        self.graph_probabilities = np.zeros((5, 5))

