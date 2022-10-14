from termcolor import colored
from Progetto_Nuovo.Environment.Environment import *
from Progetto_Nuovo.Learners.TSLearner import *
from Progetto_Nuovo.Learners.SWUCBLearner import *
from Progetto_Nuovo.generateData import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.Learners.CDUCBLearner import *
from tqdm import tqdm

n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 100
window = 25
number_of_experiments = 5
graph_filename = "../Data/graph.json"
prices_filename = "../Data/prices.json"
user_class_filename = "../Data/user_class_aggregate.json"
max_units_sold = 2

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

    # init reward collection for each experiment CD-UCB
    rewards_per_experiment_cducb = []
    # init reward collection for each experiment UCB
    rewards_per_experiment_ucb = []

    clairvoyant = evaluate_clairvoyant(configurations, max_units_sold, customer_class.reservation_prices[0],
                                       customer_class.number_of_customers)

    for e in range(number_of_experiments):
        print("Experiment {0}...".format(e))
        # init environment
        env = Environment(n_prices, customer_class, lambda_coefficient, n_products)
        # init CD-UCB learner
        cd_ucb_learner = CDUCBLearner(n_prices, n_products)
        # init UCB-1
        ucb_learner = SWUCBLearner(n_prices, n_products)

        total_seen_cducb = np.zeros((n_products, n_prices))
        total_seen_product_cducb = np.zeros(n_products)

        total_seen_ucb = np.zeros((n_products, n_prices))
        total_seen_product_ucb = np.zeros(n_products)
        # for each day
        for t in tqdm(range (0, number_of_days)):
            alpha_ratios = np.random.dirichlet(customer_class.alpha_probabilities)
            item_sold_mean = customer_class.item_sold_mean

            # initialize for the next window of 25 days
            if number_of_days % 25 == 0:
                total_seen_ucb = np.zeros((n_products, n_prices))
                total_seen_product_ucb = np.zeros(n_products)

            # UCB-1
            pulled_config_indexes_ucb = ucb_learner.pull_arm()
            pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
            reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios
                                                                         , item_sold_mean)
            # CD-UCB
            pulled_config_indexes_cducb = cd_ucb_learner.pull_arm()
            pulled_config_indexes_cducb = np.array(np.transpose(pulled_config_indexes_cducb))[0]
            reward_cducb, units_sold_cducb, total_seen_daily_cducb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios
                                                                         , item_sold_mean)


            # seen since day before
            total_seen_since_daybefore_ucb = np.copy(total_seen_ucb)
            # seen since day before
            total_seen_since_daybefore_cducb = np.copy(total_seen_cducb)
            # seen til now
            for product in range(len(pulled_config_indexes_ucb)):
                total_seen_ucb[product, pulled_config_indexes_ucb[product]] += np.sum(total_seen_daily_ucb[1:])
                total_seen_product_ucb[product] += np.sum(total_seen_daily_ucb[1:])
            ucb_learner.update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_daybefore_ucb, total_seen_ucb,
                               total_seen_product_ucb, reward_ucb)

            for product in range(len(pulled_config_indexes_cducb)):
                total_seen_cducb[product, pulled_config_indexes_cducb[product]] += np.sum(total_seen_daily_cducb[1:])
                total_seen_product_cducb[product] += np.sum(total_seen_daily_cducb[1:])
            cd_ucb_learner.update(pulled_config_indexes_cducb, units_sold_cducb, total_seen_since_daybefore_cducb, total_seen_cducb,
                               total_seen_product_cducb, reward_cducb)
            cd_ucb_learner.detect_change(pulled_config_indexes_cducb)

        # append collected reward of current experiment TS
        rewards_per_experiment_cducb.append(ts_learner.collected_rewards)
        # append collected reward of current experiment UCB
        rewards_per_experiment_ucb.append(ucb_learner.collected_rewards)
        # printTSBeta(learner.beta_parameters[:, 0, :], rewards_per_experiment[0])
        # print("SOS", learner.collected_rewards[0])
    printReward(rewards_per_experiment_cducb, clairvoyant)
    printReward(rewards_per_experiment_ucb, clairvoyant)

    evaluate_mean_std_rewards(rewards_per_experiment_cducb)
    printRegret(rewards_per_experiment_cducb, clairvoyant)
    printRegret(rewards_per_experiment_ucb, clairvoyant)


