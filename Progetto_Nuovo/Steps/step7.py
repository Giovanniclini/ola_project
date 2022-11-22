from termcolor import colored
from Progetto_Nuovo.Environment.ContextualEnvironment import ContextualEnvironment
from Progetto_Nuovo.Learners.TSLearner import *
from Progetto_Nuovo.Learners.UCBLearner import *
from Progetto_Nuovo.generateData import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.Data_Structures.ContextClass import *
from tqdm import tqdm

n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 211
number_of_experiments = 5
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
    return (mean * seen_since_day_before + unit_sold) / total_sold


if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 7 ----------------------------', 'blue', attrs=['bold']))
    # INIT PARAMETERS
    prices = get_prices_from_json(prices_filename)
    configurations = initialization_other_steps(prices)
    customer_classes = [get_customer_class_from_json(user_class_1), get_customer_class_from_json(user_class_2),
                        get_customer_class_from_json(user_class_3),
                        get_customer_class_from_json_aggregate(user_class_1, user_class_2, user_class_3)]
    customer_class_zero, customer_class_one = get_customer_class_one_feature(customer_classes)
    customer_classes.append(customer_class_zero)
    customer_classes.append(customer_class_one)

    rewards_per_experiment_ts = []
    rewards_per_experiment_ucb = []

    # EVALUATE CLAIRVOYANT
    clairvoyant = evaluate_contextual_clairvoyant(configurations, customer_classes, prices)

    for e in range(number_of_experiments):
        # init environment
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
        env = ContextualEnvironment(n_prices, lambda_coefficient, n_products)

        ts_collected_rewards = []
        ucb_collected_rewards = []

        ucb_split_rewards = [[], []]
        ts_split_rewards = [[], []]

        context_ts = ContextClass()
        context_ucb = ContextClass()

        ucb_father_reward = []
        ts_father_reward = []

        optimum_found = [False, False]
        optimum_split = [None, None]
        optimum_learner = [None, None]
        optimum_father_reward = [None, None]
        for t in tqdm(range(0, number_of_days)):
            # every 14 days run context and do a split
            if t % 14 == 0:
                # UCB LEARNER
                if not optimum_found[0]:
                    if t == 0:
                        context_ucb.split()
                    elif t == 14:
                        ucb_father_reward.append(ucb_split_rewards[0])
                        ucb_collected_rewards.append(ucb_split_rewards[0])
                        context_ucb.assign_father_lower_bound(ucb_split_rewards[0])
                        context_ucb.split()
                    else:
                        split = context_ucb.current_split
                        check, l_reward, r_reward = context_ucb.evaluate_split_condition(ucb_split_rewards[0],
                                                                                         ucb_split_rewards[1], t)
                        context_ucb.split(check, l_reward, r_reward)
                        # if the split is worth...
                        if check:
                            # if the left node is better than the right node it becomes the father node
                            if l_reward > r_reward:
                                if len(ucb_learners) > 0:
                                    optimum_split[0] = split[0]
                                    optimum_learner[0] = ucb_learners[0]
                                    optimum_father_reward[0] = ucb_split_rewards[0]
                                # 0 -> left, 1 -> right
                                ucb_father_reward.append(ucb_split_rewards[0])
                                ucb_collected_rewards.append(ucb_split_rewards[0])
                                context_ucb.father_lower_bound = l_reward
                                context_ucb.pending_list_lower_bounds.append(
                                    context_ucb.lower_bound(ucb_split_rewards[1], 5, 14))
                                context_ucb.pending_list_prob.append(context_ucb.assign_prob_context_occur(t))
                            else:
                                if len(ucb_learners) > 0:
                                    optimum_split[0] = split[1]
                                    optimum_learner[0] = ucb_learners[1]
                                    optimum_father_reward[0] = ucb_split_rewards[1]
                                ucb_father_reward.append(ucb_split_rewards[1])
                                ucb_collected_rewards.append(ucb_split_rewards[1])
                                context_ucb.father_lower_bound = r_reward
                                context_ucb.pending_list_lower_bounds.append(
                                    context_ucb.lower_bound(ucb_split_rewards[0], 5, 14))
                                context_ucb.pending_list_prob.append(context_ucb.assign_prob_context_occur(t))
                        # if the split isn't worth...
                        else:
                            if len(context_ucb.pending_list) > 0:
                                ucb_collected_rewards.append(ucb_father_reward.pop(0))
                                context_ucb.father_lower_bound = context_ucb.pending_list_lower_bounds[0]
                                context_ucb.pending_list_lower_bounds.pop(0)
                                context_ucb.pending_list_prob.pop(0)
                            else:
                                ucb_collected_rewards.append(optimum_father_reward[0])
                                context_ucb.current_split = optimum_split[0]
                                optimum_found[0] = True
                else:
                    ucb_collected_rewards.append(ucb_split_rewards[0])

                # TS LEARNER
                if not optimum_found[1]:
                    if t == 0:
                        context_ts.split()
                    elif t == 14:
                        ts_father_reward.append(ts_split_rewards[0])
                        ts_collected_rewards.append(ts_split_rewards[0])
                        context_ts.assign_father_lower_bound(ts_split_rewards[0])
                        context_ts.split()
                    else:
                        split = context_ts.current_split
                        check, l_reward, r_reward = context_ts.evaluate_split_condition(ts_split_rewards[0],
                                                                                        ts_split_rewards[1], t)
                        context_ts.split(check, l_reward, r_reward)
                        if check:
                            if l_reward > r_reward:
                                # 0 -> left, 1 -> right
                                if len(ts_learners) > 1:
                                    optimum_split[1] = split[0]
                                    optimum_learner[1] = ts_learners[0]
                                    optimum_father_reward[1] = ts_split_rewards[0]
                                ts_father_reward.append(ts_split_rewards[0])
                                ts_collected_rewards.append(ts_split_rewards[0])
                                context_ts.father_lower_bound = l_reward
                                context_ts.pending_list_lower_bounds.append(
                                    context_ts.lower_bound(ts_split_rewards[1], 5, 14))
                                context_ts.pending_list_prob.append(context_ts.assign_prob_context_occur(t))
                            else:
                                if len(ts_learners) > 1:
                                    optimum_split[1] = split[1]
                                    optimum_learner[1] = ts_learners[1]
                                    optimum_father_reward[1] = ts_split_rewards[1]
                                ts_father_reward.append(ts_split_rewards[1])
                                ts_collected_rewards.append(ts_split_rewards[1])
                                context_ts.father_lower_bound = r_reward
                                context_ts.pending_list_lower_bounds.append(
                                    context_ts.lower_bound(ts_split_rewards[0], 5, 14))
                                context_ts.pending_list_prob.append(context_ts.assign_prob_context_occur(t))

                        else:
                            if len(context_ts.pending_list) > 0:
                                ts_collected_rewards.append(list(ts_father_reward.pop(0)))
                                context_ts.father_lower_bound = context_ts.pending_list_lower_bounds[0]
                                context_ts.pending_list_lower_bounds.pop(0)
                                context_ts.pending_list_prob.pop(0)
                            else:
                                ts_collected_rewards.append(optimum_father_reward[1])
                                context_ts.current_split = optimum_split[1]
                                optimum_found[1] = True
                else:
                    ts_collected_rewards.append(ts_split_rewards[0])

                ucb_split_rewards = [[], []]
                ts_split_rewards = [[], []]

            # ---------------------------------------THOMPSON SAMPLING----------------------------------
            i = 0
            if not optimum_found[1]:
                ts_learners = []
            if not optimum_found[0]:
                ucb_learners = []
            for split in context_ts.current_split:
                if not optimum_found[1]:
                    ts_learners.append(TSLearner(n_prices, n_products))
                pulled_config_indexes_ts = ts_learners[i].pull_arm()
                env.select_costumer_class(customer_classes[get_json_from_binary_feature(split)])
                reward_ts, units_sold_ts, total_seen_daily_ts = env.round(pulled_config_indexes_ts, prices,
                                                                          alpha_ratios_ts, item_sold_mean_ts)
                ts_split_rewards[i].append(reward_ts)
                ts_learners[i].update(pulled_config_indexes_ts, units_sold_ts, np.sum(total_seen_daily_ts[1:]), reward_ts)
                alpha_ratios_ts = estimate_alpha_ratios(old_starts_ts, total_seen_daily_ts)
                old_starts_ts += total_seen_daily_ts
                total_bought_since_day_before_ts = np.copy(total_sold_product_ts)
                for p in range(len(pulled_config_indexes_ts)):
                    total_seen_ts[p, pulled_config_indexes_ts[p]] += np.sum(total_seen_daily_ts[1:])
                    total_sold_product_ts[p, pulled_config_indexes_ts[p]] += units_sold_ts[p]
                    item_sold_mean_ts[p][pulled_config_indexes_ts[p]] = estimate_items_for_each_product(
                        item_sold_mean_ts[p][pulled_config_indexes_ts[p]],
                        total_bought_since_day_before_ts[p, pulled_config_indexes_ts[p]], units_sold_ts[p],
                        total_sold_product_ts[p, pulled_config_indexes_ts[p]])
                i += 1

            # ---------------------------------------------UCB-------------------------------------------
            i = 0
            for split in context_ucb.current_split:
                if not optimum_found[0]:
                    ucb_learners.append(UCBLearner(n_prices, n_products))
                pulled_config_indexes_ucb = ucb_learners[i].pull_arm()
                pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
                env.select_costumer_class(customer_classes[get_json_from_binary_feature(split)])
                reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices,
                                                                             alpha_ratios_ucb, item_sold_mean_ucb)
                ucb_split_rewards[i].append(reward_ucb)
                total_seen_since_day_before_ucb = np.copy(total_seen_ucb)
                for p in range(len(pulled_config_indexes_ucb)):
                    total_seen_ucb[p, pulled_config_indexes_ucb[p]] += np.sum(total_seen_daily_ucb[1:])
                    total_seen_product_ucb[p] += np.sum(total_seen_daily_ucb[1:])
                ucb_learners[i].update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_day_before_ucb,
                                       total_seen_ucb, total_seen_product_ucb, reward_ucb)
                alpha_ratios_ucb = estimate_alpha_ratios(old_starts_ucb, total_seen_daily_ucb)
                old_starts_ucb += total_seen_daily_ucb
                total_bought_since_day_before_ucb = np.copy(total_sold_product_ucb)
                for p in range(len(pulled_config_indexes_ucb)):
                    total_sold_product_ucb[p, pulled_config_indexes_ucb[p]] += units_sold_ucb[p]
                    item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]] = estimate_items_for_each_product(
                        item_sold_mean_ucb[p][pulled_config_indexes_ucb[p]],
                        total_bought_since_day_before_ucb[p][pulled_config_indexes_ucb[p]],
                        units_sold_ucb[p], total_sold_product_ucb[p][pulled_config_indexes_ucb[p]])
                i += 1
        rewards_per_experiment_ucb.append(ucb_collected_rewards)
        rewards_per_experiment_ts.append(ts_collected_rewards)
    print_contextual_graphs(rewards_per_experiment_ucb, rewards_per_experiment_ts, clairvoyant)
