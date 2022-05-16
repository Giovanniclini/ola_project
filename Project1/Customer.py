import numpy as np

class Customer:
    def __init__(self, demand_curves, n_users, alpha_ratios, n_products, graph_probabilities):
        self.alphas_ratios = alpha_ratios
        self.n_users = n_users
        self.n_products = n_products
        self.demand_curves = demand_curves
        self.graph_probabilities = graph_probabilities
