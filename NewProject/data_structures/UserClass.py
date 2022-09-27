import numpy as np


class UserClass:
    def __init__(self, class_id, number_of_customers, alphas, reservation_prices, global_history, graph_probabilities):
        self.id = class_id
        # assign a random number of customers
        self.number_of_customers = number_of_customers
        # assign a random value to alpha probabilities [0, 0, 0, 0, 0, 0]
        self.alpha_ratios = alphas
        # assign a random value to the reservation price for each product
        self.reservation_prices = reservation_prices
        # assign empty global history arrays of history
        self.global_history = global_history
        # keep track of number of units sold (global) 0
        self.units_purchased = np.zeros(17)
        # keep track of number of units sold per product (global) [0 ,0, 0, 0, 0]
        self.units_purchased_per_product = np.zeros((17,5))
        # assign a random value to the social influence transition probability matrix to navigate among the products
        self.graph_probabilities = graph_probabilities
        # assign conversion rate for each campaign
        self.conversion_rate_of_campaign = np.zeros(17)
        # assign conversion rate for each product of each price campaign
        self.conversion_rate_per_product = np.zeros((17,5))
        # assign the total number of clicks
        self.number_of_clicks = np.zeros(17)
        # assign the total number of clicks
        self.number_of_clicks_per_product = np.zeros((17,5))
