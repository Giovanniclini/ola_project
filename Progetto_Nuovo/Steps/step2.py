from Progetto_Nuovo.generateData import *
from termcolor import colored
from Progetto_Nuovo.Social_Influence.SocialInfluence import *

# define number of price configurations (price changes)
number_of_configurations = 15
# define four prices for each of the five products (row = four prices for a product)
prices = np.array([[np.random.uniform(20., 300.) for _ in range(4)] for _ in range(5)])
# sort prices from lowest to highest for each product (axis = 1 = row)
prices.sort(axis=1)
# define number of products
number_of_products = 5
# define number of customer classes
number_of_customer_classes = 3
# define a list to hold the customer class objects
customers = []
# define the price configurations
price_configurations = []
# define known margin, same for each product (at the moment)
margin = [15, 19, 18, 7, 14]
# initialize optimal campaigns
optimal_config = [15, 15, 15]
# old profit
old_profit = 0.
# maximum profit for every costumer class
maximum_profit = [0., 0., 0.]


def optimizationProblem():
    check = [True, True, True]
    price_configurations, customers = initialization_step2(prices, number_of_configurations)
    # StatsManager.printData(price_configurations, customers, prices, number_of_customer_classes)
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
            profit = [0., 0., 0., 0., 0.]
            if level == 0:
                social = SocialInfluence(0.5, customers[customer_class], price_configurations[15], number_of_products)
                social.simulation()
                for prod in range(number_of_products):
                    maximum_profit[customer_class] += (price_configurations[15][prod] - margin[prod]) * social.units_sold[prod]
            # for each configuration of the arrays, set the correct value
            for i in range(5):
                # actual index
                idx = level + i
                # assign actual profit value (average)
                social = SocialInfluence(0.5, customers[customer_class], price_configurations[idx], number_of_products)
                social.simulation()
                for prod in range(number_of_products):
                    profit[i] += (price_configurations[idx][prod] - margin[prod]) * social.units_sold[prod]
            # if the new optimal is different w.r.t the old one, then update values
            possible_optimal = np.where(profit == np.max(profit))
            if profit[possible_optimal[0][0]] > maximum_profit[customer_class] and check[customer_class]:
                profit_increase = (profit[possible_optimal[0][0]] / maximum_profit[customer_class]) - 1
                max_profit_idx = possible_optimal[0][0]
                maximum_profit[customer_class] = profit[max_profit_idx]
                optimal_config[customer_class] = level + max_profit_idx
                if profit_increase > 0.:
                    print(colored('Current marginal increase {0:.2%}', 'green', attrs=['bold']).format(profit_increase))
                    print('The best configuration is number {0}: {1}  '.
                          format(optimal_config[customer_class], price_configurations[optimal_config[customer_class]]))
                else:
                    check[customer_class] = False
            else:
                # otherwise, no better solution was found
                # here the algorithm should terminate for the current customer class
                print('No better solution found')
                print(colored('Current marginal increase {0:.2%}', 'red', attrs=['bold']).format(profit_increase))
                print('The best configuration is number {0}: {1}  '.
                      format(optimal_config[customer_class], price_configurations[optimal_config[customer_class]]))


if __name__ == '__main__':
    optimizationProblem()
