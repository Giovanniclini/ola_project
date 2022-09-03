import numpy as np
from termcolor import colored
from CustomerClass import *
from PricingCampaign import *
from SocialInfluence import *
#define time of experiment (time step is one day, price changes every day)
T = 1 * 365
#define number of price configurations (price changes)
number_of_configurations = 15
#define four prices for each of the five products (row = four prices for a product)
prices = np.array([[np.random.uniform(100., 1200.) for _ in range(4)] for _ in range(5)])
#sort prices from lowest to highest for each product (axis = 1 = row)
prices.sort(axis = 1)
#define initial configuration with lowest prices
initial_price_configuration = prices[:,0]
#define the configuration with highest prices
maximum_configuration = prices[:, 3]
#define number of products
number_of_products = 5
#define number of pricing campaigns, 5 + 1, one for each price configuration + optimal
number_of_pricing_campaigns = 5 + 1
#define number of customer classes
number_of_customer_classes = 3
#define a list to hold the pricing campaign objects
campaigns = []
#define a list to hold the customer class objects
customers = []
#define known margin, same for each product (at the moment)
margin = 0.18
#define the number of arms
n_arms = number_of_pricing_campaigns
#initialize optimal campaigns
optimal_campaign = [15, 15, 15]
#old campaign used to check difference with the new found at each iteration
old_optimal_campaign = [15, 15, 15]
#assign number of experiments
n_experiments = 365
#assign the index of the optimal config for each level (always 6 config at each level, optimal is the last one)
max_profit_idx = 5
#define configurations
price_configurations = []
socialInfluence = SocialInfluence()
#a function that generates all the possible price configurations (by increasing levels of prices)
def generate_configuration_levels(prices, number_of_configurations):
    price_configurations2 = []
    for price in range(0, int((number_of_configurations+5)/5) - 1): # for each price (except the lowest)
      for product in range(int((number_of_configurations+5)/4)):
        configuration_level = np.copy(np.array(prices[:,price]))
        configuration_level[product] = np.copy(prices[product][price+1])
        price_configurations2.append(configuration_level)
    return price_configurations2

def initialization():
    # define all the price configuration levels
    price_configurations = generate_configuration_levels(prices, number_of_configurations)
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

def optimizationProblem():
    initialization()
    # print all the prices
    print('All the available prices are: \n{0}'.format(prices))
    # print all the price configurations
    print('\nAll the available configurations are: ')
    for config in price_configurations:
        print(config)
    # print all the reservations prices
    print('\nAll the reservation prices are: ')
    for c in range(number_of_customer_classes):
        print(customers[c].reservation_prices)
    for level in range(0, 15, 5):
        print(colored('\n\n---------------------------- LEVEL {0} ----------------------------', 'blue',
                      attrs=['bold']).format(int(level / 5)))
        # for each customer class
        for customer_class in range(number_of_customer_classes):
            print(colored('\nCustomer class {0}:', 'red', attrs=['bold']).format(customer_class))
            # assign init value to profit increase variable
            profit_increase = 0.
            # assign init value to conversion rates
            p = [0., 0., 0., 0., 0., 0.]
            # assign init value to profits
            profit = [0., 0., 0., 0., 0., 0.]
            # assign init value to configurations
            configurations = [0., 0., 0., 0., 0., 0.]
            # for each element of the arrays, set the correct value
            for i in range(6):
                # actual index
                idx = level + i
                # if i = 0 is the optimal configuration (campaign)
                if i == 5:
                    idx = optimal_campaign[customer_class]
                # reset value otherwise at each iteration remain the same (why? colab?)
                configurations[i] = 0
                # assign actual value of config
                configurations[i] = np.copy(campaigns[idx].configuration)
                # simulate social inflence episodes to generate the conversion rates for the price configurations
                if campaigns[idx].sales[customer_class] == 0:
                    socialInfluence.run_social_influence_simulation(5, campaigns[idx].configuration, customer_class, campaigns[idx], False, False, False, customers)
                # assign conversion rate evaluated in social influence
                p[i] = campaigns[idx].conversion_rate[customer_class]
                # assign actual profit value (average)
                profit[i] = campaigns[idx].conversion_rate[customer_class] * campaigns[idx].average_profit_per_sale
            # reset value
            max_profit_idx = 5
            print('\nInitial optimal configuration is: {0}'.format(configurations[5]))
            # assign value to the old optimal campaign for later comparison
            old_optimal_campaign[customer_class] = optimal_campaign[customer_class]
            # assign simulation time to number of customer of the customer class
            T = customers[customer_class].number_of_customers #OCCHIO IL TIME ORIZON VA CAMBIATO
            # if the new optimal is different w.r.t the old one, then update values
            possible_optimal = np.where(max(profit))
            if (possible_optimal[0][0] != max_profit_idx):
                profit_increase = (profit[possible_optimal[0][0]] / profit[max_profit_idx]) - 1
                max_profit_idx = possible_optimal[0][0]
                optimal_campaign[customer_class] = level + max_profit_idx
            else:
                # otherwise, no better solution was found
                # here the algorithm should terminate for the current customer class
                print('No better solution found')
                return
            print('The best configuration is number {0}: {1}  '.format(optimal_campaign[customer_class], campaigns[
                optimal_campaign[customer_class]].configuration))
            if profit_increase > 0:
                print(
                    colored('Current marginal increase {0:.2%}', 'green', attrs=['bold']).format(profit_increase))

            else:
                print(colored('TS current marginal increase {0:.2%}', 'red', attrs=['bold']).format(profit_increase))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 2 ----------------------------', 'blue', attrs=['bold']))
    optimizationProblem()
    print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
    # aggregate_run(False)
    print(colored('\n\n---------------------------- STEP 4 ----------------------------', 'blue', attrs=['bold']))
    # aggregate_run(True)
    print(colored('\n\n---------------------------- STEP 5 ----------------------------', 'blue', attrs=['bold']))
    #professor_run(True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
