import random

import numpy as np


class CustomerClass:
    def __init__(self, class_id):
        self.id = class_id
        # assign a random number of customers
        self.number_of_customers = np.random.randint(180, 220)
        # assign a random value to alpha probabilities
        self.alpha_probabilities = np.zeros(6)
        # assign a mean of number of item sold
        self.item_sold_mean = np.zeros(5)
        # assign a random value to the reservation price for each product
        self.reservation_prices = [np.random.uniform(50., 301.) for _ in range(5)]
        # keep track of number of units sold of each product (global)
        self.units_purchased_for_each_product = np.zeros(5)
        self.units_purchased_per_product_per_campaign = np.random.randint(0, 20, size=(17, 5))
        # assign graph table empty, it seems that must replace the social_influence_transition_probability_matrix
        self.graph_probabilities = np.zeros((5, 5))
        # assign a random value to the social influence transition probability matrix to navigate among the products
        self.social_influence_transition_probability_matrix = np.random.uniform(0, 1, (5, 5))
        # put zero the diagonal because a product already seen cannot be seen anymore
        np.fill_diagonal(self.social_influence_transition_probability_matrix, 0)
        # assign empty global history
        self.global_history = []


    def assign_values(self, units, index):
        self.units_purchased_for_each_product[index] += units
        # self.units_purchased_per_product_per_campaign = units_purchased_per_product_per_campaign
