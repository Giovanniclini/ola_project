from generateData import *
from termcolor import colored
from Social_Influence.SocialInfluence import *

# define number of price configurations (price changes)
number_of_configurations = 15
# define four prices for each of the five products (row = four prices for a product)
prices = get_prices_from_json("../Data/prices.json")
# define number of products
number_of_products = 5
# define number of customer classes
number_of_customer_classes = 3
# define a list to hold the customer class objects
customers = []
# define the price configurations
price_configurations = []
# define known margin, same for each product (at the moment)
margin = [3, 3, 3, 3, 3]
# initialize optimal campaigns
optimal_config = [15, 15, 15]
# old profit
old_profit = 0.
# maximum profit for every costumer class
maximum_profit = [0., 0., 0.]
# filename class 1
class_1 = "../Data/user_class_1.json"
# filename class 2
class_2 = "../Data/user_class_2.json"
# filename class 3
class_3 = "../Data/user_class_3.json"
# array of classes
classes = [class_1, class_2, class_3]


def optimizationProblem():
    check = [True, True, True]
    price_configurations, price_configuration_indexes, customers = initialization_step2(prices, number_of_configurations,
                                                                                        classes)
    # StatsManager.printData(price_configurations, customers, prices, number_of_customer_classes)
    for level in range(0, 15, 5):
        print(colored('\n\n---------------------------- LEVEL {0} ----------------------------', 'blue', attrs=['bold']).format(int(level / 5)))
        # check if profit increase was made
        for customer_class in range(number_of_customer_classes):
            if not check[customer_class]:
                continue
            print(colored('\nCustomer class {0}:', 'red', attrs=['bold']).format(customer_class))
            # assign init value to profit increase variable
            profit_increase = 0.
            # assign init value to profits
            profit = [0., 0., 0., 0., 0.]
            if level == 0:
                social = SocialInfluence(0.5, customers[customer_class].alpha_probabilities,
                                         customers[customer_class].item_sold_mean, customers[customer_class],
                                         price_configurations[15], number_of_products, price_configuration_indexes[15])
                social.simulation()
                for prod in range(number_of_products):
                    maximum_profit[customer_class] += (price_configurations[15][prod] - margin[prod]) * \
                                                      social.units_sold[prod]
            # for each configuration of the arrays, set the correct value
            for i in range(5):
                # actual index
                idx = level + i
                # assign actual profit value (average)
                social = SocialInfluence(0.5, customers[customer_class].alpha_probabilities,
                                         customers[customer_class].item_sold_mean, customers[customer_class],
                                         price_configurations[idx], number_of_products, price_configuration_indexes[idx])
                social.simulation()
                for prod in range(number_of_products):
                    profit[i] += (price_configurations[idx][prod] - margin[prod]) * social.units_sold[prod]
            # if the new optimal is different w.r.t the old one, then update values
            possible_optimal = np.where(profit == np.max(profit))
            if profit[possible_optimal[0][0]] > maximum_profit[customer_class] and check[customer_class]:
                profit_increase = (profit[possible_optimal[0][0]] / maximum_profit[customer_class]) - 1
                print(maximum_profit[customer_class])
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
