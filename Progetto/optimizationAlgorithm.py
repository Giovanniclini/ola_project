from generateData import *
from termcolor import colored
from SocialInfluence import *
from BanditManager import *

# define number of price configurations (price changes)
number_of_configurations = 15
# define four prices for each of the five products (row = four prices for a product)
#prices = np.array([[np.random.uniform(100., 1200.) for _ in range(4)] for _ in range(5)])
prices = np.array([[100, 150, 200, 250], [80, 120, 160, 200], [110, 140, 170, 200], [150, 190, 230, 270], [50, 90, 130, 170]])
# sort prices from lowest to highest for each product (axis = 1 = row)
prices.sort(axis=1)
# define initial configuration with the lowest prices
initial_price_configuration = prices[:, 0]
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
ts_optimal_campaign_aggregate = 15
ucb_optimal_campaign_aggregate = 15


def step3(step, T):
    global ts_optimal_campaign_aggregate
    global ucb_optimal_campaign_aggregate
    check_aggregate = [True, True]
    price_configurations, customers, campaigns = initialization(prices, number_of_configurations, step)
    # print all the prices
    print('All the available prices are: \n{0}'.format(prices))
    # print all the price configurations
    print('\nAll the available configurations are: ')
    for config in price_configurations:
        print(config)
    print('\nAll the reservation prices are: ')
    for c in range(number_of_customer_classes):
        print(customers[c].reservation_prices)
    for level in range(0, 15, 5):
        if not check_aggregate[0] and not check_aggregate[1]:
            break
        print(colored('\n\n---------------------------- LEVEL {0} ----------------------------', 'blue',
                      attrs=['bold']).format(int(level / 5)))
        # assign init value to configurations
        ts_configurations = [0., 0., 0., 0., 0., 0.]
        # assign init value to conversion rates
        ts_p = [0., 0., 0., 0., 0., 0.]
        # assign init value to configurations
        ucb_configurations = [0., 0., 0., 0., 0., 0.]
        # assign init value to conversion rates
        ucb_p = [0., 0., 0., 0., 0., 0.]
        # for each element of the arrays, set the correct value
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
                    social.run_social_influence_simulation(number_of_products, campaigns[ts_idx].configuration, c, campaigns[ts_idx], True, customers)
                if campaigns[ucb_idx].sales[c] == 0:
                    social.run_social_influence_simulation(number_of_products, campaigns[ucb_idx].configuration, c, campaigns[ucb_idx], True, customers)
            for prod in range(number_of_products):
                for c in range(3):
                    customers[c].units_purchased_for_each_product[prod] = 0
            # assign aggregate conversion rate evaluated in social influence
            ts_p[i] = np.copy(campaigns[ts_idx].aggregate_conversion_rate)
            ucb_p[i] = np.copy(campaigns[ucb_idx].aggregate_conversion_rate)

        #UCB1 LEARNER
        if check_aggregate[1]:
            ucb1 = BanditManager(id="ucb1", T=T, n_experiments=5, opt=None, idx_opt=None, probabilities=ucb_p, customers=customers)
            ucb1.executeBandit()
            ucb1.evaluateProfit(level=level, campaigns=campaigns, optimal_campaign_aggregate=ucb_optimal_campaign_aggregate)
            ucb_max_profit_idx = 5
            ucb_possible_optimal = ucb1.profit.index(max(ucb1.profit))
            ucb_profit_increase = (ucb1.profit[ucb_possible_optimal] / ucb1.profit[max_profit_idx]) - 1
            if check_aggregate[1] and ucb_possible_optimal != ucb_max_profit_idx and ucb_profit_increase > 0.:
                ucb_max_profit_idx = ucb_possible_optimal
                ucb_optimal_campaign_aggregate = level + ucb_max_profit_idx
                print('\nUCB1 optimal configuration is: {0}'.format(ucb_configurations[5]))
                print(colored('UCB1 current marginal increase {0:.2%}', 'green', attrs=['bold']).format(ucb_profit_increase))
            else:
                if check_aggregate[1]:
                    print(colored('UCB1 no better solution found: current marginal increase {0:.2%}', 'red',attrs=['bold']).format(ucb_profit_increase))
                    print('The best UCB1 configuration is number {0}: {1}  '.format(ucb_optimal_campaign_aggregate, campaigns[ucb_optimal_campaign_aggregate].configuration))
                check_aggregate[1] = False

        #TS LEARNER
        if check_aggregate[0]:
            ts_opt = [1., 1., 1., 1., 1., 1.]
            ts_idx_opt = np.argmax(ts_p)
            ts = BanditManager(id="ts", T=T, n_experiments=5, opt=ts_opt, idx_opt=ts_idx_opt, probabilities=ts_p, customers=customers)
            ts.initBandit(n_arms=6, campaigns=campaigns, optimal_campaign_aggregate=ts_optimal_campaign_aggregate, level=level)
            ts.executeBandit()
            ts.evaluateProfit(level=level, campaigns=campaigns, optimal_campaign_aggregate=ts_optimal_campaign_aggregate)
            ts_max_profit_idx = 5
            ts_possible_optimal = ts.profit.index(max(ts.profit))
            ts_profit_increase = (ts.profit[ts_possible_optimal] / ts.profit[max_profit_idx]) - 1
            if check_aggregate[0] and ts_possible_optimal != ts_max_profit_idx and ts_profit_increase > 0.:
                ts_max_profit_idx = ts_possible_optimal
                ts_optimal_campaign_aggregate = level + ts_max_profit_idx
                print('\nThompson Sampling optimal configuration is: {0}'.format(ts_configurations[5]))
                print(colored('Thompson Sampling current marginal increase {0:.2%}', 'green', attrs=['bold']).format(ts_profit_increase))
            else:
                if check_aggregate[0]:
                    # otherwise, no better solution was found
                    # here the algorithm should terminate for the current customer class
                    print(colored('Thompson Sampling no better solution found: current marginal increase {0:.2%}', 'red', attrs=['bold']).format(ts_profit_increase))
                    print('The best Thompson Sampling configuration is number {0}: {1}  '.format(ts_optimal_campaign_aggregate, campaigns[ts_optimal_campaign_aggregate].configuration))
                check_aggregate[0] = False

def step4(step, T):
    return

def optimizationProblem(step, T):
    check = [True, True, True]
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
        print(colored('\n\n---------------------------- LEVEL {0} ----------------------------', 'blue', attrs=['bold']).format(int(level / 5)))
        # chek if profit increase was made
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
                # assign aggregate conversion rate evaluated in social influence
                ts_p[i] = np.copy(campaigns[idx].aggregate_conversion_rate)
                # ucb_p = np.copy(campaigns[idx].aggregate_conversion_rate_per_product)
                # assign actual profit value (average)
                for prod in range(number_of_products):
                    profit[i] += campaigns[idx].sales_per_product[customer_class][prod] * campaigns[idx].average_margin_for_price_in_configuration[prod]
            max_profit_idx = 5
            print('\nInitial optimal configuration is: {0}'.format(configurations[5]))
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
                print('The best configuration is number {0}: {1}  '.format(optimal_campaign[customer_class], campaigns[optimal_campaign[customer_class]].configuration))


if __name__ == '__main__':
    step = 3
    if step == 2:
        print(colored('\n\n---------------------------- STEP 2 ----------------------------', 'blue', attrs=['bold']))
        optimizationProblem(step=step, T=1000)
    elif step == 3:
        print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
        step3(step=step, T=1000)
    elif step == 4:
        print(colored('\n\n---------------------------- STEP 4 ----------------------------', 'blue', attrs=['bold']))
        step4(step=step, T=1000)

