from itertools import product
from Progetto_Nuovo.Data.DataManager import *


def generate_configuration_levels(prices, number_of_configurations):
    price_configurations = []
    price_configuration_indexes = []
    for price in range(0, int((number_of_configurations+5)/5) - 1):  # for each price (except the lowest)
        for prod in range(int((number_of_configurations+5)/4)):
            configuration_level = np.copy((np.array(prices))[:, price])
            configuration_level_indexes = np.copy(np.array([price for _ in range(5)]))
            configuration_level[prod] = np.copy(prices[prod][price + 1])
            configuration_level_indexes[prod] = price + 1
            price_configurations.append(configuration_level)
            price_configuration_indexes.append(configuration_level_indexes)
    # set the 15th configuration
    price_configurations.append((np.array(prices))[:, 0])
    price_configuration_indexes.append(np.array([0 for _ in range(5)]))
    return price_configurations, price_configuration_indexes


def initialization_step2(prices, number_of_configurations, classes):
    margin = 40
    # define all the price configuration levels
    price_configurations, price_configuration_indexes = np.copy(generate_configuration_levels(prices,
                                                                                              number_of_configurations))
    # define the margin for each price configuration as 18% of prices
    margins_for_each_configuration = np.copy(price_configurations)
    for i in range(len(margins_for_each_configuration)):
        margins_for_each_configuration[i] = np.copy(margins_for_each_configuration[i] - margin)
    # define the margin means for each price configurations
    margin_means_for_configuration = np.zeros(number_of_configurations)
    for c in range(number_of_configurations):
        margin_means_for_configuration[c] = np.mean(margins_for_each_configuration[c])
    # create all customer classes
    customers = []
    for c in range(3):
        cust = get_customer_class_from_json(classes[c])
        customers.append(cust)
    return price_configurations, price_configuration_indexes, customers


def generate_all_configurations(prices):
    price_configurations = product(*prices)
    return list(price_configurations)


def initialization_other_steps(prices):
    margin = [3, 2, 4, 1, 2]
    price_configurations = np.copy(generate_all_configurations(prices))
    margins_for_each_configuration = np.copy(price_configurations)
    for i in range(len(margins_for_each_configuration)):
        for prod in range(5):
            margins_for_each_configuration[i] = np.copy(margins_for_each_configuration[i][prod] - margin[prod])
    return price_configurations
