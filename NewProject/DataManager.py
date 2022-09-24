import json
import random as random
import numpy as np
from PricingCampaign import *

class DataManager:
    def __init__(self, number_of_users, user_classes, number_of_configurations, reservation_prices,
                 transition_probabilities):
        self.number_of_users = number_of_users
        self.user_classes = user_classes
        self.number_of_configurations = number_of_configurations
        self.reservation_prices = reservation_prices
        self.transition_probabilities = transition_probabilities
        self.lambda_coefficient = 0.8

    def generate_dict_with_user_class(self):
        user_class = [0, 0, 0]
        user_class[random.randint(0, self.user_classes-1)] = 1
        user_class_number = user_class.index(1)
        return dict(class_0=user_class[0], class_1=user_class[1], class_2=user_class[2]), user_class_number

    def generate_alpha_realization(self):
        alpha = random.randint(-1, 4)
        return alpha

    def generate_dict_with_units_purchased_per_product(self, units_purchased_per_product):
        return dict(product_0=units_purchased_per_product[0], product_1=units_purchased_per_product[1],
                    product_2=units_purchased_per_product[2], product_3=units_purchased_per_product[3],
                    product_4=units_purchased_per_product[4])

    def generate_history(self, user_class, alpha, price_configuration):
        history = []
        units_purchased_per_product = [0, 0, 0, 0, 0]
        if alpha == -1:
            return history, units_purchased_per_product
        prob_matrix = np.copy(self.transition_probabilities[user_class])
        np.fill_diagonal(prob_matrix, 0)
        initial_active_nodes = [0, 0, 0, 0, 0]
        initial_active_nodes[alpha] = 1
        active_nodes = initial_active_nodes
        newly_active_nodes = active_nodes
        t = 0
        n_steps_max = 5
        if self.reservation_prices[user_class][active_nodes.index(1)] < price_configuration[active_nodes.index(1)]:
            return history, units_purchased_per_product
        history = [newly_active_nodes]
        for i in range(prob_matrix.shape[1]):
            if i in np.array(np.where(newly_active_nodes == 1)):
                for j in range(prob_matrix.shape[0]):
                    prob_matrix[j][i] = 0
        order_of_parallel_product = [active_nodes]
        while t < n_steps_max and len(order_of_parallel_product) != 0:
            first_secondary_node = np.random.randint(0, 5)
            second_secondary_node = np.random.randint(0, 5)
            # repeat assignment until the two products are different
            while second_secondary_node == first_secondary_node:
                second_secondary_node = np.random.randint(0, 5)
            # select from the probability matrix only the rows related to the active nodes
            transition_probabilities_from_the_active_node = (prob_matrix.T * order_of_parallel_product[0]).T
            p_row = transition_probabilities_from_the_active_node[order_of_parallel_product[0].index(1)]
            if np.all((p_row == 0)):
                history2 = []
                for i in range(len(history)):
                    if type(history[i]) is np.ndarray:
                        history2.append(history[i].tolist())
                    else:
                        history2.append(history[i])
                return history2, units_purchased_per_product
            # update the value of the transition probability related to the second secondary product
            p_row[second_secondary_node] = p_row[second_secondary_node] * self.lambda_coefficient
            # assign false to all the activated edges array to keep track of the one that will be selected (
            # clicked secondary product)
            activated_edges = [False for _ in range(5)]
            # one or two secondary products are chosen num_prod_clicked = np.random.randint(1, 3) random choice
            # of the index of the secondary product selected by the user (select one, but maybe could be two)
            num_prod_clicked = np.random.randint(1, 3)
            indx = random.choices(np.arange(0, 5), p_row, k=num_prod_clicked)
            # check if the probability related to the chosen index if > 0.0; if yes, activate the edge (set true)
            for z in range(num_prod_clicked):
                if (p_row[indx[z]]) > 0.0:
                    activated_edges[indx[z]] = True
            # if ((p != 0) == activated_edges) it is False, empty the matrix
            prob_matrix = prob_matrix * ((transition_probabilities_from_the_active_node != 0) == activated_edges)
            # assign 0 to the new active nodes (reset values)
            newly_active_nodes = np.zeros(5)
            # for each product, find the chosen one
            for i in range(5):
                # if the chosen secondary product is found, let the costumer actually buy it, by updated the
                # values related to the units sold and the revenue
                if activated_edges[i] and self.reservation_prices[user_class][i] >= price_configuration[i]:
                    # assign a random amount of units of product purchased by the user
                    units_purchased_per_product[i] += np.random.randint(1, 40)
                    # assign 1 to the new active nodes
                    newly_active_nodes[i] = 1
            order_of_parallel_product.pop(0)
            for i in range(prob_matrix.shape[1]):
                if i in np.array(np.where(newly_active_nodes == 1)):
                    # from all the nodes to the new active one
                    for j in range(prob_matrix.shape[0]):
                        prob_matrix[j][i] = 0
                    # update the active nodes
                    active_nodes = [0, 0, 0, 0, 0]
                    active_nodes[i] = 1
                    # update the history
                    history = np.concatenate((history, [active_nodes]), axis=0)
                    # update time step
                    t = t + 1
                    # list of product for manage the parallel case
                    order_of_parallel_product.append(active_nodes)
        # return the history
        history2 = []
        for i in range(len(history)):
            if type(history[i]) is np.ndarray:
                history2.append(history[i].tolist())
            else:
                history2.append(history[i])
        return history2, units_purchased_per_product

    def generate_file(self, price_configurations):
        users = list()
        for configuration in range(self.number_of_configurations):
            for c in range(self.number_of_users):
                user_class_dict, user_class = self.generate_dict_with_user_class()
                alpha = self.generate_alpha_realization()
                history, units_purchased_per_product = self.generate_history(user_class, alpha, price_configurations[configuration])
                units_purchased_dict = self.generate_dict_with_units_purchased_per_product(units_purchased_per_product)
                user = dict(id=c, config_id=configuration, user_class=user_class_dict, alpha_parameter=alpha,
                            history=history, units_purchased=units_purchased_dict)
                users.append(dict(user=user))
        return users

    def generate_configuration_levels(self, prices):
        price_configurations2 = [prices[:, 0]]
        for price in range(0, int((self.number_of_configurations + 5) / 5) - 1):  # for each price (except the lowest)
            for product in range(int((self.number_of_configurations + 5) / 4)):
                configuration_level = np.copy(np.array(prices[:, price]))
                configuration_level[product] = np.copy(prices[product][price + 1])
                price_configurations2.append(configuration_level)
        price_configurations2.append(prices[:, 3])
        return price_configurations2

    def generate_pricing_campaigns(self, configurations, production_costs):
        campaigns = []
        for c in range(len(configurations)):
            margin = np.subtract(configurations[c], production_costs)
            campaigns.append(PricingCampaign(c, np.mean(margin), configurations[c], margin))
        return campaigns

