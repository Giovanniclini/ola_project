import json

import numpy as np

from Progetto_Nuovo.Data_Structures.CustomerClass import *


def get_customer_class_from_json(filename):
    file = open(filename)
    data = json.load(file)
    customer_class = CustomerClass()
    customer_class.number_of_customers = data["n_users"]
    customer_class.alpha_probabilities = data["average_alphas"]
    customer_class.reservation_prices = data["reservation_prices"]
    customer_class.graph_probabilities = data["graph_probabilities"]
    customer_class.item_sold_mean = data["average_items_sold"]
    customer_class.conversion_rates = data["conversion_rates"]
    return customer_class


def get_json_from_binary_feature(binary_feature):
    if binary_feature == [0, 0]:
        return 0
    elif binary_feature == [1, 0] or binary_feature == [0, 1]:
        return 1
    elif binary_feature == [1, 1]:
        return 2
    elif binary_feature == [-1, -1]:
        return 3
    elif binary_feature == [0, -1]:
        return 4
    return 5


def evaluate_rewards_per_combination(configurations, customer_class, prices):
    rewards_per_combination = []
    for i in range(len(configurations)):
        current_reward = 0.
        for product in range(5):
            for price in range(4):
                current_reward += prices[product][price] * customer_class.conversion_rates[product][price] * \
                                  customer_class.alpha_probabilities[product] * \
                                  customer_class.item_sold_mean[product][price]
        rewards_per_combination.append(current_reward)
    return max(rewards_per_combination)


def evaluate_contextual_clairvoyant(configurations, customer_classes, prices):
    rewards_per_customer_class = []
    for customer_class in customer_classes:
        rewards_per_customer_class.append(evaluate_rewards_per_combination(configurations, customer_class, prices))
    return rewards_per_customer_class


def evaluate_clairvoyant(configurations, max_units_sold, reservation_prices, n_users):
    max_reward = 0.
    for config in configurations:
        opt_reward = 0.
        # for each product
        for product in range(5):
            if reservation_prices[product] >= config[product]:
                # evaluate the reward for
                opt_reward += max_units_sold * n_users * config[product]
        # average reward over all the simulations
        opt_reward = opt_reward / n_users
        if opt_reward > max_reward:
            max_reward = opt_reward
    return max_reward


def evaluate_abrupt_changes_clairvoyant(configurations, max_units_sold, reservation_prices, n_users, n_phases):
    max_reward = np.zeros(n_phases)
    for phase in range(n_phases):
        for config in configurations:
            opt_reward = 0.
            # for each product
            for product in range(5):
                if reservation_prices[phase][product] >= config[product]:
                    # evaluate the reward for
                    opt_reward += max_units_sold * n_users * config[product]
            # average reward over all the simulations
            opt_reward = opt_reward / n_users
            if opt_reward > max_reward[phase]:
                max_reward[phase] = opt_reward
    return max_reward


def get_customer_class_from_json_aggregate(file_name_class_1, file_name_class_2, file_name_class_3):
    # Load file customer class non aggregate
    file_class_1 = open(file_name_class_1)
    file_class_2 = open(file_name_class_2)
    file_class_3 = open(file_name_class_3)

    # Extract data from JSON for user class
    data_class_1 = json.load(file_class_1)
    data_class_2 = json.load(file_class_2)
    data_class_3 = json.load(file_class_3)

    # Assign aggregate values tu agg customer class
    customer_class_aggregate = CustomerClass()
    # Agg number of customers
    customer_class_aggregate.number_of_customers = data_class_1['n_users'] + data_class_2['n_users'] + data_class_3['n_users']
    # Agg average alphas
    temp_matrix = data_class_1['n_users'] * np.asarray(data_class_1['average_alphas']) + data_class_2['n_users'] * np.asarray(data_class_2['average_alphas']) + data_class_3['n_users'] * np.asarray(data_class_3['average_alphas'])
    customer_class_aggregate.alpha_probabilities = np.asarray(temp_matrix) / customer_class_aggregate.number_of_customers
    # Agg reservation prices
    temp_matrix = (data_class_1['n_users'] * np.asarray(data_class_1['reservation_prices'])) + (data_class_2['n_users'] * np.asarray(data_class_2['reservation_prices'])) + (data_class_3['n_users'] * np.asarray(data_class_3['reservation_prices']))
    customer_class_aggregate.reservation_prices = np.asarray(temp_matrix) / customer_class_aggregate.number_of_customers
    # Agg graph probabilities
    temp_matrix = data_class_1['n_users'] * np.asarray(data_class_1['graph_probabilities']) + data_class_2['n_users'] * np.asarray(data_class_2['graph_probabilities']) + data_class_3['n_users'] * np.asarray(data_class_3['graph_probabilities'])
    customer_class_aggregate.graph_probabilities = np.asarray(temp_matrix) / customer_class_aggregate.number_of_customers
    # Agg average item sold
    temp_matrix = data_class_1['n_users'] * np.asarray(data_class_1['average_items_sold']) + data_class_2['n_users'] * np.asarray(data_class_2['average_items_sold']) + data_class_3['n_users'] * np.asarray(data_class_3['average_items_sold'])
    customer_class_aggregate.item_sold_mean = np.asarray(temp_matrix) / customer_class_aggregate.number_of_customers
    return customer_class_aggregate


def get_customer_class_from_json_aggregate_unknown_graph(file_name_class_1, file_name_class_2, file_name_class_3):
    # Load file customer class non aggregate
    file_class_1 = open(file_name_class_1)
    file_class_2 = open(file_name_class_2)
    file_class_3 = open(file_name_class_3)

    # Extract data from JSON for user class
    data_class_1 = json.load(file_class_1)
    data_class_2 = json.load(file_class_2)
    data_class_3 = json.load(file_class_3)

    # Assign aggregate values tu agg customer class
    customer_class_aggregate = CustomerClass()
    # Agg number of customers
    customer_class_aggregate.number_of_customers = data_class_1['n_users'] + data_class_2['n_users'] + data_class_3['n_users']
    # Agg average alphas
    temp_matrix = (np.asarray(data_class_1['average_alphas']) * data_class_1['n_users']) + (np.asarray(data_class_2['average_alphas']) * data_class_2['n_users']) + (np.asarray(data_class_3['average_alphas']) * data_class_3['n_users'])
    customer_class_aggregate.alpha_probabilities = temp_matrix / customer_class_aggregate.number_of_customers
    # Agg reservation prices
    temp_matrix = (np.asarray(data_class_1['reservation_prices']) * data_class_1['n_users']) + (np.asarray(data_class_2['reservation_prices']) * data_class_2['n_users']) + (np.asarray(data_class_3['reservation_prices']) * data_class_3['n_users'])
    customer_class_aggregate.reservation_prices = temp_matrix / customer_class_aggregate.number_of_customers
    # NO AGGREGATE GRAPH BUT ONE GRAPH INITIALIZED TO 1
    customer_class_aggregate.graph_probabilities = np.ones((5, 5))
    np.fill_diagonal(customer_class_aggregate.graph_probabilities, 0)
    # Agg average item sold
    temp_matrix = (np.asarray(data_class_1['average_items_sold']) * data_class_1['n_users']) + (np.asarray(data_class_2['average_items_sold']) * data_class_2['n_users']) + (np.asarray(data_class_3['average_items_sold']) * data_class_3['n_users'])
    customer_class_aggregate.item_sold_mean = temp_matrix // customer_class_aggregate.number_of_customers

    return customer_class_aggregate


def get_customer_class_one_feature(user_classes):
    # Assign aggregate values tu agg customer class
    customer_class_feature_zero = CustomerClass()
    customer_class_feature_one = CustomerClass()

    # Agg number of customers
    # class 0
    customer_class_feature_zero.number_of_customers = user_classes[0].number_of_customers + user_classes[2].number_of_customers
    # class 1
    customer_class_feature_one.number_of_customers = user_classes[1].number_of_customers + user_classes[2].number_of_customers
    # Agg average alphas
    # class 0
    temp_matrix = user_classes[0].number_of_customers * np.asarray(user_classes[0].alpha_probabilities) + user_classes[2].number_of_customers * np.asarray(user_classes[2].alpha_probabilities)
    customer_class_feature_zero.alpha_probabilities = np.asarray(temp_matrix) / customer_class_feature_zero.number_of_customers
    # class 1
    temp_matrix = user_classes[1].number_of_customers * np.asarray(user_classes[1].alpha_probabilities) + user_classes[
        2].number_of_customers * np.asarray(user_classes[2].alpha_probabilities)
    customer_class_feature_one.alpha_probabilities = np.asarray(temp_matrix) / customer_class_feature_one.number_of_customers

    # Agg reservation prices
    # class 0
    temp_matrix = user_classes[0].number_of_customers * np.asarray(user_classes[0].reservation_prices) + user_classes[2].number_of_customers * np.asarray(user_classes[2].reservation_prices)
    customer_class_feature_zero.reservation_prices = np.asarray(temp_matrix) / customer_class_feature_zero.number_of_customers
    # class 1
    temp_matrix = user_classes[1].number_of_customers * np.asarray(user_classes[1].reservation_prices) + user_classes[
        2].number_of_customers * np.asarray(user_classes[2].reservation_prices)
    customer_class_feature_one.reservation_prices = np.asarray(temp_matrix) / customer_class_feature_one.number_of_customers

    # Agg graph probabilities
    # class 0
    temp_matrix = user_classes[0].number_of_customers * np.asarray(user_classes[0].graph_probabilities) + user_classes[2].number_of_customers * \
                  np.asarray(user_classes[2].graph_probabilities)
    customer_class_feature_zero.graph_probabilities = np.asarray(temp_matrix) / customer_class_feature_zero.number_of_customers
    # class 1
    temp_matrix = user_classes[1].number_of_customers * np.asarray(user_classes[1].graph_probabilities) + user_classes[
        2].number_of_customers * np.asarray(user_classes[2].graph_probabilities)
    customer_class_feature_one.graph_probabilities = np.asarray(temp_matrix) / customer_class_feature_one.number_of_customers

    # Agg average item sold
    # class 0
    temp_matrix = user_classes[0].number_of_customers * np.asarray(user_classes[0].item_sold_mean) + user_classes[2].number_of_customers * np.asarray(user_classes[2].item_sold_mean)
    customer_class_feature_zero.item_sold_mean = np.asarray(temp_matrix) / customer_class_feature_zero.number_of_customers
    # class 1
    temp_matrix = user_classes[1].number_of_customers * np.asarray(user_classes[1].item_sold_mean) + user_classes[
        2].number_of_customers * np.asarray(user_classes[2].item_sold_mean)
    customer_class_feature_one.item_sold_mean = np.asarray(temp_matrix) / customer_class_feature_one.number_of_customers

    return customer_class_feature_zero, customer_class_feature_one


def get_prices_from_json(filename):
    file = open(filename)
    data = json.load(file)
    prices = data["prices"]
    return prices

