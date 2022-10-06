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
