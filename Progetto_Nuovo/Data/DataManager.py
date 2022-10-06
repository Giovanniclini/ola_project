import json


def get_customer_class_from_json(filename):
    file = open(filename)
    data = json.load(file)
    return None


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
