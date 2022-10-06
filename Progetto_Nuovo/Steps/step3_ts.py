from termcolor import colored
from Progetto_Nuovo.Environment.Environment import *
from Progetto_Nuovo.Learners.TSLearner import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.generateData import *


n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 30
number_of_experiments = 5
graph_filename = "../Data/graph.json"
prices_filename = "../Data/prices.json"
user_class_filename = "../Data/user_class_aggregate.json"
max_units_sold = 20


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
    # assign graph from json file
    graph = get_graph_from_json(graph_filename)
    # assign prices per products from json file
    prices = get_prices_from_json(prices_filename)
    # generate all the possible price configurations
    configurations = initialization_other_steps(prices)
    # generate the customer class from json (aggregate)
    customer_class = get_customer_class_from_json(user_class_filename)
    # init environment
    env = Environment(n_prices, customer_class, lambda_coefficient, n_products)
    # init reward collection for each experiment
    rewards_per_experiment = []

    clairvoyant = evaluate_clairvoyant(configurations, max_units_sold, customer_class.reservation_prices[0],
                                       customer_class.number_of_customers)
    # for each experiment
    for e in range(number_of_experiments):
        # for each day
        # init Thompson Sampling learner
        learner = TSLearner(n_prices, n_products)
        for t in range(0, number_of_days):
            # pull prices belonging to a configuration (super arm)
            pulled_config_indexes = learner.pull_arm()
            # collect reward trough e-commerce simulation for all the users
            reward, units_sold, total_seen = env.round(pulled_config_indexes, prices)
            #total_seen = customer_class.number_of_customers * 5
            # update TS learner parameters
            learner.update(pulled_config_indexes, units_sold, total_seen, reward)
        # append collected reward of current experiment
        rewards_per_experiment.append(learner.collected_rewards)

        #printTSBeta(learner.beta_parameters[:, 0, :], rewards_per_experiment[0])
        #print("SOS", learner.collected_rewards[0])
        printTSRegret(learner, clairvoyant)

