import json
from Progetto_Nuovo.Data_Structures.CustomerClass import *


def get_customer_class_from_json(filename):
    file = open(filename)
    data = json.load(file)
    customer_class = CustomerClass(0)
    customer_class.number_of_customers = data["n_users"]
    customer_class.alpha_probabilities = data["average_alphas"]
    customer_class.reservation_prices = data["reservation_prices"]
    customer_class.social_influence_transition_probability_matrix = data["graph_probabilities"]
    return customer_class

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
    customer_class_aggregate = CustomerClass(0)
    # Agg number of customers
    customer_class_aggregate.number_of_customers = data_class_1['n_users'] + data_class_2['n_users'] + data_class_3['n_users']
    # Agg average alphas
    temp_matrix = data_class_1['n_users'] * data_class_1['average_alphas'] + data_class_2['n_users'] * data_class_2['average_alphas'] + data_class_3['n_users'] * data_class_3['average_alphas']
    customer_class_aggregate.alpha_probabilities = temp_matrix / customer_class_aggregate.number_of_customers
    # Agg reservation prices
    temp_matrix = data_class_1['n_users'] * data_class_1['reservation_prices'] + data_class_2['n_users'] * data_class_2['reservation_prices'] + data_class_3['n_users'] * data_class_3['reservation_prices']
    customer_class_aggregate.alpha_probabilities = temp_matrix / customer_class_aggregate.number_of_customers
    # Agg graph probabilities
    temp_matrix = data_class_1['n_users'] * data_class_1['graph_probabilities'] + data_class_2['n_users'] * data_class_2['graph_probabilities'] + data_class_3['n_users'] * data_class_3['grapg_probabilities']
    customer_class_aggregate.alpha_probabilities = temp_matrix / customer_class_aggregate.number_of_customers

    return customer_class_aggregate

def get_graph_from_json(filename):
    return None


def get_prices_from_json(filename):
    file = open(filename)
    data = json.load(file)
    prices = data["prices"]
    return prices


def evaluate_aggregate_conversion_rates(customer_classes):
    return None


def evaluate_aggregate_alphas(customer_classes):
    return None


def evaluate_aggregate_graph_probabilities(customer_classes):
    return None


def generate_configurations(prices):
    return None
