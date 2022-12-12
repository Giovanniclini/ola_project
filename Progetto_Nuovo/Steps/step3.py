from termcolor import colored
from Progetto_Nuovo.Environment.Environment import *
from Progetto_Nuovo.Learners.TSLearner import *
from Progetto_Nuovo.Learners.UCBLearner import *
from Progetto_Nuovo.generateData import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from tqdm import tqdm

n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 200
number_of_experiments = 5
prices_filename = "../Data/prices.json"
user_class_1 = "../Data/user_class_1.json"
user_class_2 = "../Data/user_class_2.json"
user_class_3 = "../Data/user_class_3.json"
max_units_sold = 1.5

if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 3 ----------------------------', 'blue', attrs=['bold']))
    # assign prices per products from json file
    prices = get_prices_from_json(prices_filename)
    # generate all the possible price configurations
    configurations = initialization_other_steps(prices)
    # generate the customer class from json (aggregate)
    customer_class = get_customer_class_from_json_aggregate(user_class_1, user_class_2, user_class_3)

    # init reward collection for each experiment TS
    rewards_per_experiment_ts = []
    # init reward collection for each experiment UCB
    rewards_per_experiment_ucb = []

    clairvoyant = evaluate_clairvoyant(configurations, max_units_sold, customer_class.reservation_prices[0],
                                       customer_class.number_of_customers)

    for e in range(number_of_experiments):
        # init environment
        env = Environment(n_prices, customer_class, lambda_coefficient, n_products)
        # init Thompson Sampling learner
        ts_learner = TSLearner(n_prices, n_products)
        # init UCB-1
        ucb_learner = UCBLearner(n_prices, n_products)
        total_seen_ucb = np.zeros((n_products, n_prices))
        total_seen_product_ucb = np.zeros(n_products)
        # for each day
        for t in tqdm(range(0, number_of_days)):
            alpha_ratios = np.random.dirichlet(customer_class.alpha_probabilities)
            item_sold_mean = customer_class.item_sold_mean

            # THOMPSON SAMPLING
            # pull prices belonging to a configuration (super arm)
            pulled_config_indexes_ts = ts_learner.pull_arm()
            # collect reward trough e-commerce simulation for all the users
            reward_ts, units_sold_ts, total_seen_ts = env.round(pulled_config_indexes_ts, prices, alpha_ratios,
                                                                item_sold_mean)
            # update TS learner parameters
            ts_learner.update(pulled_config_indexes_ts, units_sold_ts, np.sum(total_seen_ts[1:]), reward_ts)

            # UCB-1
            pulled_config_indexes_ucb = ucb_learner.pull_arm()
            pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
            reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios
                                                                         , item_sold_mean)
            # seen since day before
            total_seen_since_daybefore_ucb = np.copy(total_seen_ucb)
            # seen til now
            for product in range(len(pulled_config_indexes_ucb)):
                total_seen_ucb[product, pulled_config_indexes_ucb[product]] += np.sum(total_seen_daily_ucb[1:])
                total_seen_product_ucb[product] += np.sum(total_seen_daily_ucb[1:])
            ucb_learner.update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_daybefore_ucb, total_seen_ucb,
                               total_seen_product_ucb, reward_ucb)
        rewards_per_experiment_ts.append(ts_learner.collected_rewards)
        rewards_per_experiment_ucb.append(ucb_learner.collected_rewards)

    printRegret(rewards_per_experiment_ucb, rewards_per_experiment_ts, clairvoyant, "Comparing Regret")
    printReward(rewards_per_experiment_ts, clairvoyant, "TS")
    printReward(rewards_per_experiment_ucb, clairvoyant, "UCB")
