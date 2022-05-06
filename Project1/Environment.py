import numpy as np

class Environment:
    def __init__(self, n_product_types, product_prices, n_users, reservation_price):
        self.n_product_types = n_product_types #int
        self.product_prices =  product_prices #array
        self.n_user = n_users #int
        self.carts = [[] for i in range(n_product_types)] #array/matrice
        self.reservation_price = reservation_price #array of int
