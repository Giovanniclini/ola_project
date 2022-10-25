from termcolor import colored
from Progetto_Nuovo.Environment.ContextualEnvironment import ContextualEnvironment
from Progetto_Nuovo.Learners.TSLearner import *
from Progetto_Nuovo.Learners.UCBLearner import *
from Progetto_Nuovo.generateData import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.Data_Structures.ContextClass import *

n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 200
number_of_experiments = 5
graph_filename = "../Data/graph.json"
prices_filename = "../Data/prices.json"
user_class_1 = "../Data/user_class_1.json"
user_class_2 = "../Data/user_class_2.json"
user_class_3 = "../Data/user_class_3.json"
max_units_sold = 1


def estimate_alpha_ratios(old_starts, starts):
    new_starts = old_starts + starts
    return new_starts / np.sum(new_starts)


def estimate_items_for_each_product(mean, seen_since_day_before, unit_sold, total_sold):
    if mean * seen_since_day_before + unit_sold == 0:
        return 0.
    if total_sold == 0:
        print("ciao")
    return (mean * seen_since_day_before + unit_sold) / total_sold


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 7 ----------------------------', 'blue', attrs=['bold']))

    # assign graph from json file
    graph = get_graph_from_json(graph_filename)
    # assign prices per products from json file
    prices = get_prices_from_json(prices_filename)
    # generate all the possible price configurations
    configurations = initialization_other_steps(prices)
    # generate the customer class from json (aggregate)
    customer_classes = [get_customer_class_from_json(user_class_1), get_customer_class_from_json(user_class_2),
                        get_customer_class_from_json(user_class_3)]
    # init reward collection for each experiment TS
    rewards_per_experiment_ts = []
    # init reward collection for each experiment UCB
    rewards_per_experiment_ucb = []

    # TODO: fare clairvoyant per classi non aggregate
    clairvoyant = evaluate_clairvoyant(configurations, max_units_sold, customer_class.reservation_prices[0],
                                       customer_class.number_of_customers)

    for e in range(number_of_experiments):
        # init environment
        env = ContextualEnvironment(n_prices, customer_class, lambda_coefficient, n_products)

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
        item_sold_mean_ts = [[3 / 2 for _ in range(n_prices)] for _ in range(n_products)]
        # initialize item sold mean UCB
        item_sold_mean_ucb = [[3 / 2 for _ in range(n_prices)] for _ in range(n_products)]
        # for each day

        ucb_learners = []
        ts_learners = []

        ucb_split_rewards = []
        ts_split_rewards = []

        ts_collected_rewards = []
        ucb_collected_rewards = []

        for t in range(0, number_of_days):
            if t % 14 == 0:
                # UCB LEARNER
                ucb_learners = []
                context_ucb = ContextClass()
                if t == 0:
                    context_ucb.split()
                elif t == 14:
                    ucb_collected_rewards.append(ucb_split_rewards)
                    context_ucb.assign_father_lower_bound(ucb_split_rewards)
                    context_ucb.split()
                else:
                    check, l_reward, r_reward = context_ucb.evaluate_split_condition(ucb_split_rewards)
                    context_ucb.split(check, l_reward, r_reward)
                    if check:
                        if l_reward > r_reward:
                            ucb_collected_rewards.append(ucb_split_rewards[0])
                            context_ucb.father_lower_bound = l_reward
                        else:
                            ucb_collected_rewards.append(ucb_split_rewards[1])
                            context_ucb.father_lower_bound = r_reward
                    else:
                        context_ucb.pending_list_lower_bounds.append(ucb_collected_rewards[-1])
                        context_ucb.pending_list_prob(t)
                        context_ucb.split(check, l_reward, r_reward)

                for split in context_ucb.current_split:
                    ucb_learners.append(UCBLearner(n_prices, n_products))

                # TS LEARNER
                ts_learners = []
                context_ts = ContextClass()
                if t == 0:
                    context_ts.split()
                elif t == 14:
                    ucb_collected_rewards.append(ts_split_rewards)
                    context_ts.assign_father_lower_bound(ts_split_rewards)
                    context_ts.split()
                else:
                    check, l_reward, r_reward = context_ts.evaluate_split_condition(ts_split_rewards)
                    context_ts.split(check, l_reward, r_reward)
                    if check:
                        if l_reward > r_reward:
                            ucb_collected_rewards.append(ts_split_rewards[0])
                            context_ts.father_lower_bound = l_reward
                        else:
                            ucb_collected_rewards.append(ts_split_rewards[1])
                            context_ts.father_lower_bound = r_reward
                    else:
                        context_ts.pending_list_lower_bounds.append(ts_split_rewards[-1])
                        context_ts.pending_list_prob(t)
                        context_ts.split(check, l_reward, r_reward)

                for split in context_ts.current_split:
                    ts_learners.append(TSLearner(n_prices, n_products))

            # ---------------------------------------THOMPSON SAMPLING----------------------------------
            for ts_learner in ts_learners:
                pulled_config_indexes_ts = ts_learner.pull_arm()
                reward_ts, units_sold_ts, total_seen_daily_ts = env.round(pulled_config_indexes_ts, prices, alpha_ratios_ts, item_sold_mean_ts)
                ts_split_rewards.append(reward_ts)
                ts_learner.update(pulled_config_indexes_ts, units_sold_ts, np.sum(total_seen_daily_ts[1:]), reward_ts)
                alpha_ratios_ts = estimate_alpha_ratios(old_starts_ts, total_seen_daily_ts)
                total_bought_since_day_before_ts = np.copy(total_sold_product_ts)
                for p in range(len(pulled_config_indexes_ts)):
                    total_seen_ts[p, pulled_config_indexes_ts[p]] += np.sum(total_seen_daily_ts[1:])
                    total_sold_product_ts[p, pulled_config_indexes_ts[p]] += units_sold_ts[p]
                    item_sold_mean_ts[p][pulled_config_indexes_ts[p]] = estimate_items_for_each_product(
                        item_sold_mean_ts[p][pulled_config_indexes_ts[p]],
                        total_bought_since_day_before_ts[p, pulled_config_indexes_ts[p]], units_sold_ts[p],
                        total_sold_product_ts[p, pulled_config_indexes_ts[p]])

            # ---------------------------------------------UCB-------------------------------------------
            for ucb_learner in ucb_learners:
                pulled_config_indexes_ucb = ucb_learner.pull_arm()
                pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
                reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios_ucb, item_sold_mean_ucb)
                ucb_split_rewards.append(reward_ucb)
                total_seen_since_day_before_ucb = np.copy(total_seen_ucb)
                for p in range(len(pulled_config_indexes_ucb)):
                    total_seen_ucb[p, pulled_config_indexes_ucb[p]] += np.sum(total_seen_daily_ucb[1:])
                    total_seen_product_ucb[p] += np.sum(total_seen_daily_ucb[1:])
                ucb_learner.update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_day_before_ucb,
                                   total_seen_ucb, total_seen_product_ucb, reward_ucb)
                alpha_ratios_ucb = estimate_alpha_ratios(old_starts_ucb, total_seen_daily_ucb)
                total_bought_since_day_before_ucb = np.copy(total_sold_product_ucb)
                for p in range(len(pulled_config_indexes_ucb)):
                    total_sold_product_ucb[p, pulled_config_indexes_ucb[p]] += units_sold_ucb[p]
                    item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]] = estimate_items_for_each_product(
                        item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]],
                        total_bought_since_day_before_ucb[p][pulled_config_indexes_ucb[p]],
                        units_sold_ucb[p], total_sold_product_ucb[p][pulled_config_indexes_ucb[p]])

        # append collected reward of current experiment TS
        rewards_per_experiment_ts.append(ts_collected_rewards)
        # append collected reward of current experiment UCB
        rewards_per_experiment_ucb.append(ucb_collected_rewards)

        printReward(rewards_per_experiment_ts, clairvoyant)
        printReward(rewards_per_experiment_ucb, clairvoyant)

        printRegret(rewards_per_experiment_ts, clairvoyant)
        printRegret(rewards_per_experiment_ucb, clairvoyant)
