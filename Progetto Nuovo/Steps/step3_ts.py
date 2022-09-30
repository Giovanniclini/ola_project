from termcolor import colored
from Environment.Environment import *
from Learner.TSLearner import *
from Data.DataManager import *
from Data.StatsManager import *


n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 90
number_of_experiments = 10
filename =


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
    customer_class = get_customer_class_from_json(filename)
    graph = get_graph_from_json(filename)

    env = Environment()
    learner = TSLearner()

    rewards_per_experiment = []
    for e in range(number_of_experiments):
        for t in range(0, number_of_days):
            pulled_arm = learner.pull_arm()
            reward = env.round(pulled_arm)
            learner.update(pulled_arm, reward)
        rewards_per_experiment.append(learner.collected_rewards)

