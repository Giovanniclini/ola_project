from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

def printTSBeta(beta_parameters, exp_reward):
  plt.figure(figsize=(16, 9))
  x = np.linspace(0,1,1000)
  colors = ['red', 'green', 'blue', 'purple', 'orange']
  for params, rew, color in zip(beta_parameters, exp_reward, colors):
    rv = beta(*params)
    plt.plot(x, rv.pdf(x), color=color)
    plt.axvline(rew, linestyle='--', color=color)
  plt.grid()
  plt.show()

def printRegret(regret, pseudo_regret):
  plt.figure(figsize=(16,9))
  plt.ylabel("Regret")
  plt.xlabel("t")
  plt.plot(np.cumsum(pseudo_regret), color='r', label='Pseudo-regret')
  plt.plot(np.cumsum(regret), color='g', label='Regret')
  plt.legend()
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