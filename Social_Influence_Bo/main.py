'''import numpy as np
from copy import copy

def simulate_episode(init_probability_matrix, n_steps_max):
    prob_matrix = init_probability_matrix
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.random.binomial(1, 0.1, size=(n_nodes))
    history = np.array(initial_active_nodes)
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t = 0
    while(t<n_steps_max and np.sum(newly_active_nodes)>0):
        p = (prob_matrix.T*active_nodes).T #ricavo le probabilità dei nodi attivi
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1]) #ricavo i valori degli archi attivati, se il valore è maggiore di quello [0-1] lo attivo
        prob_matrix = prob_matrix*((p != 0) == activated_edges) #rimuovo dalla matrice di probabilità i valori relativi ai nodi attivati precedentemente
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0)*(1-active_nodes) #
        active_nodes = np.array(active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history

def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob = np.ones(n_nodes)+1.0/(n_nodes-1)
    credits = np.zeros(n_nodes)
    occur_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] ==1 ).reshape(-1)
        if len(idx_w_active)>0 and idx_w_active>0:
            active_nodes_in_prev_step = episode[idx_w_active-1,:].reshape[-1]
            credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
        for v in range(0, n_nodes):
            if(v!=node_index):
                idx_v_active = np.argwhere(episode[:, v]==1).reshape(-1)
                if len(idx_v_active)>0 and (idx_v_active<idx_w_active or len(idx_w_active)==0):
                    occur_v_active[v] += 1
    estimated_prob = credits/occur_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob

n_nodes = 5
n_episodes = 1000
prob_matrix = np.random.uniform(0.0, 0.5, (n_nodes,n_nodes))
node_index = 4
dataset = []

for e in range (0, n_episodes):
    dataset.append(simulate_episode(init_probability_matrix=prob_matrix, n_steps_max=10))

estimated_prob = estimate_probabilities(dateset=dataset, node_index=node_index, n_nodes=n_nodes)

print("True P Matrix", prob_matrix[:,4])
print("Estimated P Matrix", estimated_prob)'''
import numpy as np
import matplotlib.pyplot as plt
from LinearMabEnvironment import LinearMabEnvirnoment
from LinearUCBLearner import LinUCBLearner
n_arms = 10
T = 1000
n_experiments = 100
lin_ucb_rewards_per_experiments = []

env = LinearMabEnvirnoment(n_arms=n_arms, dim=10)

for e in range(0, n_experiments):
    lin_ucb_learner = LinUCBLearner(arms_features=env.arms_features)
    for t in range(0, T):
        pulled_arm = lin_ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        lin_ucb_learner.update(pulled_arm, reward)
    lin_ucb_rewards_per_experiments.append(lin_ucb_learner.collected_rewards)


opt = env.opt()
plt.figure(0)
plt.plot(np.cumsum(np.mean(opt-lin_ucb_rewards_per_experiments, axis=0)),'r')
plt.show()
