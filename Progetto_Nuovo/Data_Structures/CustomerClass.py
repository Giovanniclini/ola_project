import random

import numpy as np


class CustomerClass:
    def __init__(self):
        self.conversion_rates = np.zeros((5, 4))
        # assign a random number of customers
        self.number_of_customers = np.random.randint(180, 220)
        # assign a random value to alpha probabilities
        self.alpha_probabilities = np.zeros(6)
        # assign a mean of number of item sold
        self.item_sold_mean = np.zeros((5, 4))
        # assign a random value to the reservation price for each product
        self.reservation_prices = [np.random.uniform(50., 301.) for _ in range(5)]
        # keep track of the products clicked
        self.units_clicked_for_each_product = np.zeros(5)
        # keep track of number of units sold of each product (global)
        self.units_purchased_for_each_product = np.zeros(5)
        self.units_purchased_per_product_per_campaign = np.random.randint(0, 20, size=(17, 5))
        # assign graph table empty, it seems that must replace the social_influence_transition_probability_matrix
        self.graph_probabilities = np.zeros((5, 5))
        # assign empty global history
        self.global_history = []

    def assign_values(self, units, index):
        self.units_purchased_for_each_product[index] += units
        # self.units_purchased_per_product_per_campaign = units_purchased_per_product_per_campaign
