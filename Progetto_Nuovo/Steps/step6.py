from termcolor import colored
from Progetto_Nuovo.Environment.Environment import *
from Progetto_Nuovo.Learners.SWUCBLearner import *
from Progetto_Nuovo.generateData import *
from Progetto_Nuovo.Data.DataManager import *
from Progetto_Nuovo.Data.StatsManager import *
from Progetto_Nuovo.Learners.CDUCBLearner import *
from Progetto_Nuovo.Environment.NonStationaryEnvironment import *
from tqdm import tqdm

n_prices = 4
n_products = 5
lambda_coefficient = 0.2
number_of_days = 200
window = 25
number_of_experiments = 5
prices_filename = "../Data/prices.json"
user_class_filename = "../Data/user_class_aggregate_abrupt_changes.json"
max_units_sold = 1
n_phases = 4

if __name__ == '__main__':
    print(colored('\n\n---------------------------- STEP 6 ----------------------------', 'blue', attrs=['bold']))
    cd_ucb_phase = 0
    sw_ucb_phase = 0
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

    for e in range(number_of_experiments):
        # init environment
        env = NonStationaryEnvironment(n_prices, customer_class, lambda_coefficient, n_products)
        # init CD-UCB learner
        cd_ucb_learner = CDUCBLearner(n_prices, n_products)
        # init UCB-1
        ucb_learner = SWUCBLearner(n_prices, n_products)

        total_seen_cducb = np.zeros((n_products, n_prices))
        total_seen_product_cducb = np.zeros(n_products)

        total_seen_ucb = np.zeros((n_products, n_prices))
        total_seen_product_ucb = np.zeros(n_products)
        total_seen_product_per_window_ucb = np.zeros(n_products)
        # for each day
        for t in tqdm(range(0, number_of_days)):
            alpha_ratios = np.random.dirichlet(customer_class.alpha_probabilities)
            item_sold_mean = customer_class.item_sold_mean

            # UCB-1
            pulled_config_indexes_ucb = ucb_learner.pull_arm()
            pulled_config_indexes_ucb = np.array(np.transpose(pulled_config_indexes_ucb))[0]
            reward_ucb, units_sold_ucb, total_seen_daily_ucb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios
                                                                         , item_sold_mean, sw_ucb_phase)
            # CD-UCB
            pulled_config_indexes_cducb = cd_ucb_learner.pull_arm()
            pulled_config_indexes_cducb = np.array(np.transpose(pulled_config_indexes_cducb))[0]
            reward_cducb, units_sold_cducb, total_seen_daily_cducb = env.round(pulled_config_indexes_ucb, prices, alpha_ratios, item_sold_mean, cd_ucb_phase)


            # seen since day before
            total_seen_since_daybefore_ucb = np.copy(total_seen_ucb)
            # seen since day before
            total_seen_since_daybefore_cducb = np.copy(total_seen_cducb)
            # seen til now
            for product in range(len(pulled_config_indexes_ucb)):
                total_seen_ucb[product, pulled_config_indexes_ucb[product]] += np.sum(total_seen_daily_ucb[1:])
                total_seen_product_ucb[product] += np.sum(total_seen_daily_ucb[1:])
                total_seen_product_per_window_ucb[product] += np.sum(total_seen_daily_ucb[1:])
            ucb_learner.update(pulled_config_indexes_ucb, units_sold_ucb, total_seen_since_daybefore_ucb, total_seen_ucb,
                               total_seen_product_ucb, total_seen_product_per_window_ucb, reward_ucb)

            for product in range(len(pulled_config_indexes_cducb)):
                total_seen_cducb[product, pulled_config_indexes_cducb[product]] += np.sum(total_seen_daily_cducb[1:])
                total_seen_product_cducb[product] += np.sum(total_seen_daily_cducb[1:])
            cd_ucb_learner.update(pulled_config_indexes_cducb, units_sold_cducb, total_seen_since_daybefore_cducb,
                                  total_seen_cducb, total_seen_product_cducb, reward_cducb)
            if cd_ucb_learner.detect_change(pulled_config_indexes_cducb):
                if cd_ucb_phase < 3:
                    cd_ucb_phase += 1
                total_seen_cducb = np.zeros((n_products, n_prices))
                total_seen_product_per_window_cducb = np.zeros(n_products)
                # initialize for the next window of 25 days
            if t != 0 and t % window == 0:
                if sw_ucb_phase < 3:
                    sw_ucb_phase += 1
                total_seen_ucb = np.zeros((n_products, n_prices))
                total_seen_product_per_window_ucb = np.zeros(n_products)

        # append collected reward of current experiment UCB
        rewards_per_experiment_ucb.append(ucb_learner.collected_rewards)
        rewards_per_experiment_cducb.append(cd_ucb_learner.collected_rewards)
        # printTSBeta(learner.beta_parameters[:, 0, :], rewards_per_experiment[0])
        # print("SOS", learner.collected_rewards[0])
    #printReward(rewards_per_experiment_cducb, clairvoyant)
    #printReward(rewards_per_experiment_ucb, clairvoyant)

    #evaluate_mean_std_rewards(rewards_per_experiment_cducb)
    #printRegret(rewards_per_experiment_cducb, clairvoyant)
    #printRegret(rewards_per_experiment_ucb, clairvoyant)

    clairvoyant_phases = evaluate_abrupt_changes_clairvoyant(configurations, max_units_sold, customer_class.reservation_prices,
                                       customer_class.number_of_customers, n_phases)

    cd_ucb_instantaneus_regret = np.zeros(number_of_days)
    sw_ucb_instantaneus_regret = np.zeros(number_of_days)
    optimum_per_round = np.zeros(number_of_days)

    for i in range(n_phases):
        t_index = range(i*window, (i+1)*window)
        optimum_per_round[t_index] = clairvoyant_phases[i]
        cd_ucb_instantaneus_regret[t_index] = clairvoyant_phases[i] -np.mean(rewards_per_experiment_cducb, axis=0)[t_index]
        sw_ucb_instantaneus_regret[t_index] = clairvoyant_phases[i] -np.mean(rewards_per_experiment_ucb, axis=0)[t_index]

    plt.figure(0)
    plt.plot(np.mean(rewards_per_experiment_cducb, axis=0), 'r')
    plt.plot(np.mean(rewards_per_experiment_ucb, axis=0), 'b')
    plt.plot(optimum_per_round, 'k--')
    plt.legend(['CD-UCB', 'SW-UCB', 'Clairvoyant'])
    plt.xlabel("t")
    plt.ylabel("Reward")

    plt.figure(1)
    plt.plot(np.cumsum(cd_ucb_instantaneus_regret), 'r')
    plt.plot(np.cumsum(sw_ucb_instantaneus_regret), 'b')
    plt.legend(['CD-UCB', 'SW-UCB'])
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.show()

