from DataManager import *
from NewProject.data_structures.UserClass import *
import pandas as pd
from OptimizationAlgorithm import *

days = 31
number_of_user_classes = 3
number_of_users_per_class = [400, 500, 300]
reservation_prices = [[20, 15, 30, 25, 40],
                      [25, 20, 35, 30, 45],
                      [30, 25, 40, 35, 50]]
production_costs = [3, 4, 9, 8, 2]
number_of_configurations = 17
prices = np.array([[10, 15, 20, 25], [8, 12, 16, 20], [11, 14, 17, 20], [15, 19, 23, 27], [5, 9, 13, 17]])
# sort prices from lowest to highest for each product (axis = 1 = row)
prices.sort(axis=1)
# define initial configuration with the lowest prices
initial_price_configuration = prices[:, 0]
# define the configuration with the highest prices
maximum_configuration = prices[:, 3]



if __name__ == '__main__':
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    user_classes = []
    transition_probabilities = []
    price_configurations = []
    aggregate_user_class = UserClass(-1, sum(number_of_users_per_class), [], reservation_prices, [], [])
    for c in range(number_of_user_classes):
        user_class = UserClass(c, number_of_users_per_class[c], [0., 0., 0., 0., 0., 0.], reservation_prices[c], [], np.random.uniform(0, 1, (5, 5)))
        user_classes.append(user_class)
        transition_probabilities.append(user_class.graph_probabilities)
    data_manager = DataManager(sum(number_of_users_per_class), number_of_user_classes, number_of_configurations,
                               reservation_prices, transition_probabilities)
    price_configurations = data_manager.generate_configuration_levels(prices)
    users = data_manager.generate_file(price_configurations)
    campaigns = data_manager.generate_pricing_campaigns(price_configurations, production_costs)
    day = 1
    data = dict(day=day, users=users)
    filename = "configurations_data/configurations.json"
    with open(filename, "a") as f:
        json.dump(data, f)
    d = None
    with open(filename) as f:
        d = json.load(f)
    dataframe = pd.json_normalize(d['users'], max_level=2)
    dataframe.columns = ['id', 'config_id', 'class_0', 'class_1', 'class_2', 'alpha', 'history', 'units_purch_prod_0',
                         'units_purch_prod_1', 'units_purch_prod_2', 'units_purch_prod_3', 'units_purch_prod_4']
    print(dataframe.head(10))
    units_purchased_per_product = []
    units_purchased = np.zeros(17)
    for c in range(number_of_configurations):
        units = np.zeros(5)
        for p in range(5):
            units[p] = dataframe.loc[dataframe['config_id'] == c, 'units_purch_prod_'+str(p)].sum()
            units_purchased[c] += units[p]
        units_purchased_per_product.append(units)
        aggregate_user_class.global_history.append(dataframe.loc[dataframe['config_id'] == c, 'history'])
    aggregate_user_class.units_purchased_per_product = units_purchased_per_product
    aggregate_user_class.units_purchased = units_purchased
    print("User Class Aggregate: \n Number of customers: {1}; \n Alpha ratios {2}; \n Reservation prices {3}; \n Units purchased {4}; \n Units purchased per product {5} \n".format(aggregate_user_class.id, aggregate_user_class.number_of_customers, aggregate_user_class.alpha_ratios, aggregate_user_class.reservation_prices, aggregate_user_class.units_purchased, aggregate_user_class.units_purchased_per_product))
    print(colored('\nCustomer class {0}:', 'red', attrs=['bold']).format(aggregate_user_class.id))
    optimization_ts = OptimizationAlgorithm(True, [], aggregate_user_class.id, 0, aggregate_user_class, campaigns)
    optimization_ts.execute(0)