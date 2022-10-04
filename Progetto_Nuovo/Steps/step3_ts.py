from termcolor import colored
from Progetto_Nuovo.Environment.Environment import *
from Progetto_Nuovo.Learner.TSLearner import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.generateData import *


n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 90
number_of_experiments = 10
filename =


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
    graph = get_graph_from_json(filename)
    prices = get_prices_from_json(filename)
    configurations = initialization_other_steps(prices)
    customer_class = get_customer_class_from_json(filename)

    env = Environment(n_prices, customer_class, lambda_coefficient, n_products)
    learner = TSLearner(n_prices, n_products)

    total_seen = customer_class.n_users * 5
    rewards_per_experiment = []
    for e in range(number_of_experiments):
        for t in range(0, number_of_days):
            pulled_config_indexes = learner.pull_arm()
            reward = env.round(pulled_config_indexes, prices)
            learner.update(pulled_config_indexes, reward, total_seen)
        rewards_per_experiment.append(learner.collected_rewards)

