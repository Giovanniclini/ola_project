import numpy as np
from Data_Structures.CustomerClass import CustomerClass
from Social_Influence.SocialInfluence import *
from optimizationAlgorithm import *


def generate_configuration_levels(prices, number_of_configurations):
    price_configurations = []
    for price in range(0, int((number_of_configurations+5)/5) - 1):  # for each price (except the lowest)
        for product in range(int((number_of_configurations+5)/4)):
            configuration_level = np.copy(np.array(prices[:, price]))
            configuration_level[product] = np.copy(prices[product][price+1])
            price_configurations.append(configuration_level)
    return price_configurations


def initialization(prices, number_of_configurations):
    margin = 40
    # define all the price configuration levels
    price_configurations = np.copy(generate_configuration_levels(prices, number_of_configurations))
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
        customers.append(CustomerClass(c))
    return price_configurations, customers