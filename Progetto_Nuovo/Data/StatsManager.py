from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain


def evaluate_mean_std_rewards(rewards):
  mean = np.mean(rewards, axis=0)
  std = np.std(rewards, axis=0)
  plt.figure(figsize=(9, 6))
  plt.plot(range(len(mean)), mean)
  plt.fill_between(range(len(mean)), (mean - std), (mean + std), color='b', alpha=.1)
  plt.show()
  return mean, std


def printRegret(rewardUCB, rewardTS, clairvoyant, tag):
  plt.figure(figsize=(9, 6))
  plt.ylabel("Regret")
  plt.xlabel("t")
  plt.plot(np.cumsum(np.mean(clairvoyant - rewardUCB, axis=0)), color='g', label='RegretUCB')
  plt.plot(np.cumsum(np.mean(clairvoyant - rewardTS, axis=0)), color='b', label='RegretTS')
  plt.legend(['UCB regret', 'TS regret'])
  plt.grid()
  plt.title(tag)
  plt.show()


def printReward(rewards, clairvoyant, tag):
  mean = np.mean(rewards, axis=0)
  std = np.std(rewards, axis=0)
  plt.figure(figsize=(9, 6))
  plt.xlabel("t")
  plt.ylabel("Reward")
  plt.axhline(y=clairvoyant, color='r', linestyle='-', label='Clairvoyant')
  plt.plot(mean, color='b', label='Reward')
  plt.fill_between(range(len(mean)), (mean - std), (mean + std), color='b', alpha=.1)
  plt.grid()
  plt.title(tag)
  plt.show()



def printTSBeta(beta_parameters, exp_reward):
  plt.figure(figsize=(16, 9))
  x = np.linspace(0, 1, 1000)
  colors = ['red', 'green', 'blue', 'purple', 'orange']
  for params, rew, color in zip(beta_parameters, exp_reward, colors):
    rv = beta(*params)
    plt.plot(x, rv.pdf(x), color=color)
    #plt.axvline(rew, linestyle='--', color=color)
  plt.grid()
  plt.show()

def printUCBBound(regrets, pseudo_regrets, T, n_repetitions, deltas):
  # Compute the cumulative sum
  cumu_regret = np.cumsum(regrets, axis=1)
  cumu_pseudo_regret = np.cumsum(pseudo_regrets, axis=1)

  # Take the average over different runs
  avg_cumu_regret = np.mean(cumu_regret, axis=0)
  avg_cumu_pseudo_regret = np.mean(cumu_pseudo_regret, axis=0)

  std_cumu_regret = np.std(cumu_regret, axis=0)
  std_cumu_pseudo_regret = np.std(cumu_pseudo_regret, axis=0)

  ucb1_upper_bound = np.array([8 * np.log(t) * sum(1 / deltas) + (1 + np.pi ** 2 / 3) * sum(deltas)
                               for t in range(1, T + 1)])

  plt.figure(figsize=(16, 9))
  plt.ylabel("Regret")
  plt.xlabel("t")
  plt.plot(avg_cumu_pseudo_regret, color='r', label='Pseudo-regret')
  plt.plot(avg_cumu_regret + 1.96 * std_cumu_regret / np.sqrt(n_repetitions), linestyle='--', color='g')
  plt.plot(avg_cumu_regret, color='g', label='Regret')
  plt.plot(avg_cumu_regret - 1.96 * std_cumu_regret / np.sqrt(n_repetitions), linestyle='--', color='g')
  plt.plot(ucb1_upper_bound, color='b', label='Upper bound')
  plt.legend()
  plt.grid()
  plt.show()


def printData(price_configurations, customers, prices, number_of_customer_classes):
  # print all the prices
  print('All the available prices are: \n{0}'.format(prices))
  # print all the price configurations
  print('\nAll the available configurations are: ')
  for config in price_configurations:
    print(config)
  print('\nAll the reservation prices are: ')
  for c in range(number_of_customer_classes):
    print(customers[c].reservation_prices)


def print_contextual_graphs(rewards_per_experiment, clairvoyant, tag):
    rewards = []
    for exp_rewards in rewards_per_experiment:
        rewards.append(list(chain(*exp_rewards)))
        print(len(list(chain(*exp_rewards))))
    for customer_class in range(3):
        printReward(rewards, clairvoyant[customer_class], tag)
        printRegret(np.asarray(rewards), clairvoyant[customer_class], tag)

