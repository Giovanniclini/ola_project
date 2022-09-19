import numpy as np
from PricingCampaign import *
from CustomerClass import CustomerClass
from optimizationAlgorithm import *


def generate_configuration_levels(prices, number_of_configurations):
    price_configurations2 = []
    for price in range(0, int((number_of_configurations+5)/5) - 1):  # for each price (except the lowest)
        for product in range(int((number_of_configurations+5)/4)):
            configuration_level = np.copy(np.array(prices[:, price]))
            configuration_level[product] = np.copy(prices[product][price+1])
            price_configurations2.append(configuration_level)
    return price_configurations2


def initialization(prices, number_of_configurations, step):
    initial_price_configuration = prices[:, 0]
    maximum_configuration = prices[:, 3]
    margin = 40
    # define all the price configuration levels
    price_configurations = np.copy(generate_configuration_levels(prices, number_of_configurations))
    # define the margin for each price configuration as 18% of prices
    margins_for_each_configuration = np.copy(price_configurations)
    initial_price_configuration_margin = np.copy(initial_price_configuration - margin)
    maximum_configuration_margin = np.copy(maximum_configuration - margin)
    for i in range(len(margins_for_each_configuration)):
        margins_for_each_configuration[i] = np.copy(margins_for_each_configuration[i] - margin)

    # define the margin means for each price configurations
    margin_means_for_configuration = np.zeros(number_of_configurations)
    for c in range(number_of_configurations):
        margin_means_for_configuration[c] = np.mean(margins_for_each_configuration[c])
    mean_initial_price_configuration_margin = np.mean(initial_price_configuration_margin)
    mean_maximum_configuration_margin = np.mean(maximum_configuration_margin)

    # create all of the pricing campaigns (one for each price configuration = 1 + 15 + 1 = initial + 5 levels + maximum)
    for c in range(number_of_configurations):
        campaigns.append(PricingCampaign(c, margin_means_for_configuration[c], price_configurations[c], margins_for_each_configuration[c]))
    campaigns.append(PricingCampaign(15, mean_initial_price_configuration_margin, initial_price_configuration,initial_price_configuration_margin))
    campaigns.append(PricingCampaign(16, mean_maximum_configuration_margin, maximum_configuration, maximum_configuration_margin))

    # create all customer classes
    for c in range(number_of_customer_classes):
        customers.append(CustomerClass(c))
    if step == 2:
        for camp in campaigns:
            for cus in customers:
                for prod in range(5):
                    camp.sales_per_product[cus.id][prod] = np.random.randint(cus.number_of_customers/2,
                                                                             cus.number_of_customers)
                    camp.sales[cus.id] += camp.sales_per_product[cus.id][prod]
                # camp.no_sales[cus.class_id] = cus.number_of_customers - camp.sales[cus.class_id]
    return price_configurations, customers, campaigns

def generate_social_influence_data(level, ts_configurations, ucb_configurations, ts_p, ucb_p, campaigns, number_of_products, customers, ts_optimal_campaign_aggregate, ucb_optimal_campaign_aggregate):
    for i in range(6):
        # actual index
        ts_idx = level + i
        ucb_idx = level + i
        # if i = 0 is the optimal configuration (campaign)
        if i == 5:
            ts_idx = ts_optimal_campaign_aggregate
            ucb_idx = ucb_optimal_campaign_aggregate
        # reset value otherwise at each iteration remain the same (why? colab?)
        ts_configurations[i] = 0
        # assign actual value of config
        ts_configurations[i] = np.copy(campaigns[ts_idx].configuration)
        ucb_configurations[i] = 0
        # assign actual value of config
        ucb_configurations[i] = np.copy(campaigns[ucb_idx].configuration)
        # simulate social influence episodes to generate the conversion rates for the price configurations
        social = SocialInfluence()
        for c in range(3):
            if campaigns[ts_idx].sales[c] == 0:
                social.run_social_influence_simulation(number_of_products, campaigns[ts_idx].configuration, c,
                                                       campaigns[ts_idx], True, customers)
            if campaigns[ucb_idx].sales[c] == 0:
                social.run_social_influence_simulation(number_of_products, campaigns[ucb_idx].configuration, c,
                                                       campaigns[ucb_idx], True, customers)
        for prod in range(number_of_products):
            for c in range(3):
                customers[c].units_purchased_for_each_product[prod] = 0
        # assign aggregate conversion rate evaluated in social influence
        ts_p[i] = np.copy(campaigns[ts_idx].aggregate_conversion_rate)
        ucb_p[i] = np.copy(campaigns[ucb_idx].aggregate_conversion_rate)
    return ts_configurations, ucb_configurations, ts_p, ucb_p