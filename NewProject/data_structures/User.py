class User:

    def __init__(self, user_id, day, class_id, alphas, histories, units_purchased_per_config,
                 units_purchased_per_product_per_config):
        self.id = user_id
        self.day = day
        self.class_id = class_id
        self.alphas = alphas
        self.histories = histories
        self.units_purchased_per_config = units_purchased_per_config
        self.units_purchased_per_product_per_config = units_purchased_per_product_per_config
