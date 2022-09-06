import numpy as np
from PricingCampaign import *
from CustomerClass import *
from optimizationAlgorithm import *


def generate_configuration_levels(prices, number_of_configurations):
    price_configurations2 = []
    for price in range(0, int((number_of_configurations+5)/5) - 1): # for each price (except the lowest)
      for product in range(int((number_of_configurations+5)/4)):
        configuration_level = np.copy(np.array(prices[:,price]))
        configuration_level[product] = np.copy(prices[product][price+1])
        price_configurations2.append(configuration_level)
    return price_configurations2

def initialization(prices, number_of_configurations):
    # define all the price configuration levels
    price_configurations = np.copy(generate_configuration_levels(prices, number_of_configurations))
    # define the margin for each price configuration as 18% of prices
    margins_for_each_configuration = np.copy(price_configurations)
    initial_price_configuration_margin = np.copy(initial_price_configuration * margin)
    maximum_configuration_margin = np.copy(maximum_configuration * margin)
    for i in range(len(margins_for_each_configuration)):
        margins_for_each_configuration[i] = np.copy(margins_for_each_configuration[i] * margin)

    # define the margin means for each price configurations
    margin_means_for_configuration = np.zeros(number_of_configurations)
    for c in range(number_of_configurations):
        margin_means_for_configuration[c] = np.mean(margins_for_each_configuration[c])
    mean_initial_price_configuration_margin = np.mean(initial_price_configuration_margin)
    mean_maximum_configuration_margin = np.mean(maximum_configuration_margin)

    # create all of the pricing campaigns (one for each price configuration = 1 + 15 + 1 = initial + 5 levels + maximum)
    for c in range(number_of_configurations):
        campaigns.append(PricingCampaign(c, margin_means_for_configuration[c], price_configurations[c],
                                          margins_for_each_configuration[c]))
    campaigns.append(PricingCampaign(15, mean_initial_price_configuration_margin, initial_price_configuration,
                                      initial_price_configuration_margin))
    campaigns.append(
        PricingCampaign(16, maximum_configuration_margin, maximum_configuration, maximum_configuration_margin))

    # create all of the customer classes
    for c in range(number_of_customer_classes):
        customers.append(CustomerClass(c))

    for camp in campaigns:
        for cus in customers:
            for prod in range(number_of_products):
                camp.sales_per_product[cus.id][prod] = np.random.randint(cus.number_of_customers/2, cus.number_of_customers)
                camp.sales[cus.id] += camp.sales_per_product[cus.id][prod]
            #camp.no_sales[cus.class_id] = cus.number_of_customers - camp.sales[cus.class_id]
    return price_configurations, customers, campaigns


