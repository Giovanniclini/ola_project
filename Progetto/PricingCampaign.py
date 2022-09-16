# one campaign for each price configuration
import numpy as np


class PricingCampaign:
    def __init__(self, campaign_id, average_margin_for_configuration, configuration, margins_for_configuration):
        self.id = campaign_id
        # assign the configuration to the pricing campaign
        self.configuration = np.copy(configuration)
        # assign a random conversion rate between 1% and 20% for this campaign, TBD with Social Influence for each
        # customer class
        self.conversion_rate = np.random.uniform(0.01, 0.5, 3)
        # assign a random conversion rate between 1% and 20% for this campaign, for each price belonging to the price
        # configuration for each customer class
        self.conversion_rate_for_each_product = np.random.uniform(0.01, 0.5, (3, 5))
        # assign an amount of profit per successful sale for this campaign
        self.average_margin_for_sale = np.copy(average_margin_for_configuration)
        # assign an amount of profit per successful sale for this campaign for each price belonging to the price
        # configuration
        self.average_margin_for_price_in_configuration = np.copy(margins_for_configuration)
        # track the number of successes and failures
        self.sales = np.zeros(3)
        self.no_sales = np.zeros(3)
        self.sales_per_product = np.zeros((3, 5))

        # define same data for aggregate model
        self.marginal_profit = np.zeros(3)
        self.aggregate_sales = 0.
        self.aggregate_no_sales = 0.
        self.aggregate_sales_per_product = np.zeros(5)
        self.aggregate_no_sales_per_product = np.zeros(5)
        self.aggregate_units_sold_per_product = [0, 0, 0, 0, 0]
        self.aggregate_conversion_rate = np.random.uniform(0.01, 0.5)
        self.aggregate_conversion_rate_for_each_product = np.random.uniform(0.01, 0.5, 5)

        # history returned for each customer
        self.global_history = [[] for _ in range(3)]


# define a function to try an pricing campaign on a customer, TDB Social Inflence
def try_campaign(campaign, customer_class):
    if np.random.random() <= campaign.conversion_rate[customer_class]:
        campaign.sales[customer_class] += 1
    else:
        campaign.no_sales[customer_class] += 1


