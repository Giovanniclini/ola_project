# one campaign for each price configuration
import numpy as np


class PricingCampaign:
    def __init__(self, campaign_id, average_margin_for_configuration, configuration, margins_for_configuration):
        self.id = campaign_id
        self.configuration = np.copy(configuration)
        self.average_margin_for_sale = np.copy(average_margin_for_configuration)
        self.average_margin_for_price_in_configuration = np.copy(margins_for_configuration)



