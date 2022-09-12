from generateData import *
from termcolor import colored
from SocialInfluence import *
from Environment import *
from TSLearner import *
from UCBLearner import *
from CustomerClass import CustomerClass
from StatsManager import *

# define number of price configurations (price changes)
number_of_configurations = 15
# define four prices for each of the five products (row = four prices for a product)
prices = np.array([[np.random.uniform(100., 1200.) for _ in range(4)] for _ in range(5)])
# sort prices from lowest to highest for each product (axis = 1 = row)
prices.sort(axis=1)
# define initial configuration with the lowest prices
initial_price_configuration = prices[:,0]
# define the configuration with the highest prices
maximum_configuration = prices[:, 3]
# define number of products
number_of_products = 5
# define number of pricing campaigns, 5 + 1, one for each price configuration + optimal
number_of_pricing_campaigns = 5 + 1
# define number of customer classes
number_of_customer_classes = 3
# define a list to hold the pricing campaign objects
campaigns = []
# define a list to hold the customer class objects
customers = []
# define the price configurations
price_configurations = []
# define known margin, same for each product (at the moment)
margin = 0.18
# initialize optimal campaigns
optimal_campaign = [15, 15, 15]
# old campaign used to check difference with the new found at each iteration
old_optimal_campaign = [15, 15, 15]
# assign the index of the optimal config for each level (always 6 config at each level, optimal is the last one)
max_profit_idx = 5
# marginal increase check
check = [True, True, True]


def optimizationProblem(step, T):
    price_configurations, customers, campaigns = initialization(prices, number_of_configurations, step)
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
            if not check[customer_class]:
                continue
            print(colored('\nCustomer class {0}:', 'red', attrs=['bold']).format(customer_class))
            # assign init value to profit increase variable
            profit_increase = 0.
            # assign init value to profits
            profit = [0., 0., 0., 0., 0., 0.]
            # assign init value to configurations
            configurations = [0., 0., 0., 0., 0., 0.]
            # assign init value to conversion rates
            ts_p = [0., 0., 0., 0., 0., 0.]
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
                # simulate social influence episodes to generate the conversion rates for the price configurations
                social = SocialInfluence()
                if step == 3:
                    if campaigns[idx].sales[customer_class] == 0:
                        social.run_social_influence_simulation(number_of_products, campaigns[idx].configuration,
                                                               customer_class, campaigns[idx], True, customers)
                    for prod in range(number_of_products):
                        for c in range(3):
                            profit[i] += social.evaluate_profit_aggregate(customers[c], campaigns[idx],
                                                                          configurations[i])
                            print(customers[c].units_purchased_for_each_product[prod])
                            customers[c].units_purchased_for_each_product[prod] = 0
                # assign aggregate conversion rate evaluated in social influence
                ts_p[i] = np.copy(campaigns[idx].aggregate_conversion_rate)
                # ucb_p = np.copy(campaigns[idx].aggregate_conversion_rate_per_product)
                if step == 2:
                    # assign actual profit value (average)
                    for prod in range(number_of_products):
                        profit[i] += campaigns[idx].sales_per_product[customer_class][prod] * \
                                     campaigns[idx].average_margin_for_price_in_configuration[prod]
            if step == 3:
                n_repetitions = 5
                regrets, pseudo_regrets = np.zeros((n_repetitions, T)), np.zeros((n_repetitions, T))
                for i in range(n_repetitions):
                    regrets[i], pseudo_regrets[i], deltas = UCB1(ts_p, T)
                printUCBBound(regrets, pseudo_regrets, T, n_repetitions, deltas)
                '''
                ts_env = Environment(n_arms=number_of_products, probabilities=ts_p)
                ts_learner = TS_Learner(n_arms=number_of_products)

                ts_opt = [1., 1., 1., 1., 1., 1.]
                ts_idx_opt = np.argmax(ts_p)
                for t in range(0, T):
                    # Thompson Sampling
                    pulled_arm = ts_learner.pull_arm()
                    reward = ts_env.round(pulled_arm)
                    ts_learner.update(pulled_arm, reward)
                
                # print("Il regret è: ", np.cumsum(ts_opt - ts_learner.collected_rewards))
            # reset value
            '''
            max_profit_idx = 5
            #print('\nInitial optimal configuration is: {0}'.format(configurations[5]))
            # assign value to the old optimal campaign for later comparison
            old_optimal_campaign[customer_class] = optimal_campaign[customer_class]
            # assign simulation time to number of customer of the customer class
            T = customers[customer_class].number_of_customers #OCCHIO IL TIME ORIZON VA CAMBIATO
            # if the new optimal is different w.r.t the old one, then update values
            possible_optimal = np.where(max(profit))
            if possible_optimal[0][0] != max_profit_idx:
                profit_increase = (profit[possible_optimal[0][0]] / profit[max_profit_idx]) - 1
                max_profit_idx = possible_optimal[0][0]
                optimal_campaign[customer_class] = level + max_profit_idx
                if profit_increase > 0.:
                    print(colored('Current marginal increase {0:.2%}', 'green', attrs=['bold']).format(profit_increase))
                else:
                    check[customer_class] = False
            else:
                # otherwise, no better solution was found
                # here the algorithm should terminate for the current customer class
                print('No better solution found')
                print(colored('Current marginal increase {0:.2%}', 'red', attrs=['bold']).format(profit_increase))
            print('The best configuration is number {0}: {1}  '.format(optimal_campaign[customer_class], campaigns[
                optimal_campaign[customer_class]].configuration))


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 2 ----------------------------', 'blue', attrs=['bold']))
    optimizationProblem(step=3, T=1000)
