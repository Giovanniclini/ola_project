import numpy as np


class UserClass:
    def __init__(self, class_id, number_of_customers, alphas, reservation_prices, global_history, number_of_purchases,
                 number_of_purchases_per_product, graph_probabilities, conversion_rate_of_campaign,
                 conversion_rate_per_product, number_of_clicks, number_of_clicks_per_product):
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
        self.units_purchased = number_of_purchases
        # keep track of number of units sold per product (global) [0 ,0, 0, 0, 0]
        self.units_purchased_per_product = number_of_purchases_per_product
        # assign a random value to the social influence transition probability matrix to navigate among the products
        self.graph_probabilities = graph_probabilities
        # assign conversion rate for each campaign
        self.conversion_rate_of_campaign = conversion_rate_of_campaign
        # assign conversion rate for each product of each price campaign
        self.conversion_rate_per_product = conversion_rate_per_product
        # assign the total number of clicks
        self.number_of_clicks = number_of_clicks
        # assign the total number of clicks
        self.number_of_clicks_per_product = number_of_clicks_per_product
