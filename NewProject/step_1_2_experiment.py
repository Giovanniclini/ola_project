from DataManager import *
from UserClass import *


days = 31
number_of_user_classes = 3
number_of_users_per_class = [400, 500, 300]
reservation_prices = [[20, 15, 30, 25, 40],
                      [25, 20, 35, 30, 45],
                      [30, 25, 40, 35, 50]]

number_of_configurations = 17
price_configuration = [20, 12, 36, 40, 55]

if __name__ == '__main__':
    user_classes = []
    transition_probabilities = []
    for c in range(number_of_user_classes):
        user_class = UserClass(c, number_of_users_per_class[c], [0., 0., 0., 0., 0., 0.], reservation_prices[c], [], 0, [0, 0, 0, 0, 0],
                               np.random.uniform(0, 1, (5, 5)), 0., [0., 0., 0., 0., 0.], 0, [0, 0, 0, 0, 0])
        user_classes.append(user_class)
        transition_probabilities.append(user_class.graph_probabilities)

    data_manager = DataManager(sum(number_of_users_per_class), number_of_user_classes, number_of_configurations, reservation_prices, transition_probabilities)
    data_manager.generate_file(1, price_configuration, 1)