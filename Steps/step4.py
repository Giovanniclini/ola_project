from termcolor import colored
from Environment.Environment import *
from Learners.TSLearner import *
from Learners.UCBLearner import *
from generateData import *
from Data.DataManager import *
from Data.StatsManager import *
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
max_units_sold = 0.75


def estimate_alpha_ratios(old_starts, starts):
    new_starts = old_starts + starts
    return new_starts / np.sum(new_starts)


def estimate_items_for_each_product(mean, seen_since_day_before, unit_sold, total_sold):
    if mean * seen_since_day_before + unit_sold == 0:
        return 0.
    return (mean * seen_since_day_before + unit_sold) / total_sold


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 4 ----------------------------', 'blue', attrs=['bold']))

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
        total_seen_ts = np.zeros((n_products, n_prices))
        total_sold_product_ts = np.zeros((n_products, n_prices))
        total_sold_product_ucb = np.zeros((n_products, n_prices))

        # initialize alpha ratios for TS
        alpha_ratios_ts = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        # initialize alpha ratios for UCB
        alpha_ratios_ucb = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        # starts of day before TS
        old_starts_ts = np.zeros(6)
        # starts of day before UCB
        old_starts_ucb = np.zeros(6)
        # number of unit bought for each price
        unit_bought = np.ones((n_products, n_prices))
        # mean of product TS
        mean_product_ts = np.zeros((n_products, n_prices))
        # mean of product UCB
        mean_product_ucb = np.zeros((n_products, n_prices))
        # initialize item sold mean TS
        item_sold_mean_ts = [[3/2 for _ in range(n_prices)] for _ in range(n_products)]
        # initialize item sold mean UCB
        item_sold_mean_ucb = [[3/2 for _ in range(n_prices)] for _ in range(n_products)]
        # for each day
        for t in tqdm(range(0, number_of_days)):
            # ---------------------------------------THOMPSON SAMPLING----------------------------------
            # pull prices belonging to a configuration (super arm)
            pulled_config_indexes_ts = ts_learner.pull_arm()
            # collect reward trough e-commerce simulation for all the users
            reward_ts, units_sold_ts, total_seen_daily_ts = env.round(pulled_config_indexes_ts, prices, alpha_ratios_ts,
                                                                      item_sold_mean_ts)
            # update TS learner parameters
            ts_learner.update(pulled_config_indexes_ts, units_sold_ts, np.sum(total_seen_daily_ts[1:]), reward_ts)

            # ------------------------------------------alpha ratio-------------------------------------

            # call alpha estimate for TS
            alpha_ratios_ts = estimate_alpha_ratios(old_starts_ts, total_seen_daily_ts)
            old_starts_ts += total_seen_daily_ts

            # ------------------------------------------unit bought-------------------------------------

            # bought since day before
            total_bought_since_day_before_ts = np.copy(total_sold_product_ts)

            # seen til now
            for p in range(len(pulled_config_indexes_ts)):
                total_seen_ts[p, pulled_config_indexes_ts[p]] += np.sum(total_seen_daily_ts[1:])
                total_sold_product_ts[p, pulled_config_indexes_ts[p]] += units_sold_ts[p]
                # call unit bought for product estimate
                if total_sold_product_ts[p, pulled_config_indexes_ts[p]] > 0:
                    item_sold_mean_ts[p][pulled_config_indexes_ts[p]] = estimate_items_for_each_product(
                        item_sold_mean_ts[p][pulled_config_indexes_ts[p]],
                        total_bought_since_day_before_ts[p, pulled_config_indexes_ts[p]], units_sold_ts[p],
                        total_sold_product_ts[p, pulled_config_indexes_ts[p]])
            # ---------------------------------------------UCB-------------------------------------------

            # UCB
            pulled_config_indexes_ucb = ucb_learner.pull_arm()
            pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
            reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices,
                                                                         alpha_ratios_ucb, item_sold_mean_ucb)
            # seen since day before
            total_seen_since_day_before_ucb = np.copy(total_seen_ucb)
            # seen til now
            for p in range(len(pulled_config_indexes_ucb)):
                total_seen_ucb[p, pulled_config_indexes_ucb[p]] += np.sum(total_seen_daily_ucb[1:])
                total_seen_product_ucb[p] += np.sum(total_seen_daily_ucb[1:])
            ucb_learner.update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_day_before_ucb,
                               total_seen_ucb, total_seen_product_ucb, reward_ucb)

            # ------------------------------------------alpha ratio-------------------------------------

            # call alpha estimate for UCB
            alpha_ratios_ucb = estimate_alpha_ratios(old_starts_ucb, total_seen_daily_ucb)
            old_starts_ucb += total_seen_daily_ucb

            # ------------------------------------------unit bought-------------------------------------

            # seen since day before
            total_bought_since_day_before_ucb = np.copy(total_sold_product_ucb)
            # seen til now
            for p in range(len(pulled_config_indexes_ucb)):
                total_sold_product_ucb[p, pulled_config_indexes_ucb[p]] += units_sold_ucb[p]
                # call unit bought for product estimate
                if total_sold_product_ucb[p, pulled_config_indexes_ucb[p]] > 0:
                    item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]] = estimate_items_for_each_product(
                        item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]],
                        total_bought_since_day_before_ucb[p][pulled_config_indexes_ucb[p]],
                        units_sold_ucb[p], total_sold_product_ucb[p][pulled_config_indexes_ucb[p]])
        # append collected reward of current experiment TS
        rewards_per_experiment_ts.append(ts_learner.collected_rewards)
        # append collected reward of current experiment UCB
        rewards_per_experiment_ucb.append(ucb_learner.collected_rewards)

    printRegret(rewards_per_experiment_ucb, rewards_per_experiment_ts, clairvoyant, "Comparing Regret")
    printReward(rewards_per_experiment_ts, clairvoyant, "TS Rewards")
    printReward(rewards_per_experiment_ucb, clairvoyant, "UCB Rewards")


