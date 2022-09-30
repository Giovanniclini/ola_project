import numpy as np
import random


class SocialInfluence:
    def __init__(self, lambda_coeff, customer_class, price_configuration, n_prod):
        self.n_users = customer_class.number_of_customers
        self.reward = 0.
        self.dirchlet_probs = customer_class.alpha_probabilities
        self.lambda_coeff = lambda_coeff
        self.graph_probs = customer_class.social_influence_transition_probability_matrix
        self.customer_class = customer_class
        self.configuration = price_configuration
        # da mettere un grafo per lo step 5
        self.n_prod = n_prod
        self.units_sold = [0.] * n_prod
        self.global_history = []

    def simulation(self):
        for u in range(self.n_users):
            initial_active_nodes = np.zeros(self.n_prod)
            # usare multinomial sull'output di np.random.dirichlet per ottere il primo prodotto mostrato
            initial_active_nodes[np.random.choice(np.arange(0, 5), self.customer_class.alpha_probabilities[1:5])] = 1
            self.global_history.append(self.graph_search(initial_active_nodes))
        self.evaluate_reward()

    def evaluate_reward(self):
        # assign zero to the variable (init)
        self.reward = 0.
        # for each product
        for product in range(5):
            # evaluate the profit (margin) by multiplying the units purchased (of each product) by their average margin
            self.reward += self.units_sold[product] * self.configuration[product] * self.dirchlet_probs[product]
        self.reward = self.reward / self.n_users

    def graph_search(self, initial_active_nodes):
        # store_the number of products, i.e, the nodes of the graph
        prob_matrix = np.copy(self.graph_probs)
        # initialize the history
        history = []
        # assign the active nodes to the initial active one(s)
        active_nodes = initial_active_nodes
        # assign the new active nodes (nodes active at the end of each iteration) to the initial active one(s)
        newly_active_nodes = active_nodes
        # assign value to initial time step
        t = 0
        # assign value to n_steps_max,
        n_steps_max = 5
        # check if the first product displayed has a price higher than the reservation price; if yes, return
        if self.customer_class.reservation_prices[int(np.where(active_nodes == 1)[0])] < self.configuration[int(
                np.where(active_nodes == 1)[0])]:
            return history
        # assign first the history to the initial active nodes array
        history = np.array([initial_active_nodes])
        # update transition probability of the new active nodes to zero value so that it is not possible to reach
        # again the same node (product)
        for i in range(prob_matrix.shape[1]):
            if i in np.array(np.where(newly_active_nodes == 1)):
                # from all the nodes to the new active one
                for j in range(prob_matrix.shape[0]):
                    prob_matrix[j][i] = 0
        # while the number of max steps is not reached and there are still active nodes, continue simulation
        order_of_parallel_product = [active_nodes]
        while t < n_steps_max and len(order_of_parallel_product) != 0:
            # assign at random value the first secondary node
            first_secondary_node = np.random.randint(0, 5)
            # assign at random value the second secondary node
            second_secondary_node = np.random.randint(0, 5)
            # repeat assignment until the two products are different
            while second_secondary_node == first_secondary_node:
                second_secondary_node = np.random.randint(0, 5)
            # select from the probability matrix only the rows related to the active nodes
            transition_probabilities_from_the_active_node = (prob_matrix.T * order_of_parallel_product[0]).T
            # select from the probability rows just the ones related to the active node
            p_row = transition_probabilities_from_the_active_node[np.where(order_of_parallel_product[0] == 1)][0]
            if np.all((p_row == 0)):
                return history
            # update the value of the transition probability related to the second secondary product
            p_row[second_secondary_node] = p_row[second_secondary_node] * self.lambda_coeff
            # assign false to all the activated edges array to keep track of the one that will be selected (
            # clicked secondary product)
            activated_edges = [False for _ in range(5)]
            # one or two secondary products are chosen num_prod_clicked = np.random.randint(1, 3) random choice
            # of the index of the secondary product selected by the user (select one, but maybe could be two)
            num_prod_clicked = np.random.randint(1, 3)
            indx = random.choices(np.arange(0, 5), p_row, k=num_prod_clicked)
            # check if the probability related to the chosen index if > 0.0; if yes, activate the edge (set true)
            for z in range(num_prod_clicked):
                if (p_row[indx[z]]) > 0.0:
                    activated_edges[indx[z]] = True
            # if ((p != 0) == activated_edges) it is False, empty the matrix
            prob_matrix = prob_matrix * ((transition_probabilities_from_the_active_node != 0) == activated_edges)
            # assign 0 to the new active nodes (reset values)
            newly_active_nodes = np.zeros(5)
            # for each product, find the chosen one
            for i in range(5):
                # if the chosen secondary product is found, let the costumer actually buy it, by updated the
                # values related to the units sold and the revenue
                if activated_edges[i] and self.customer_class.reservation_prices[i] >= self.configuration[i]:
                    # assign a random amount of units of product purchased by the user
                    units_purchased = np.random.randint(1, 20)
                    # update the amount of unites of product purchased by the class of user
                    self.units_sold[i] += units_purchased
                    # assign 1 to the new active nodes
                    newly_active_nodes[i] = 1
            # update transition probability of the new active nodes to zero value so that it is not possible to
            # reach again the same node (product)
            order_of_parallel_product.pop(0)
            for i in range(prob_matrix.shape[1]):
                if i in np.array(np.where(newly_active_nodes == 1)):
                    # from all the nodes to the new active one
                    for j in range(prob_matrix.shape[0]):
                        prob_matrix[j][i] = 0
                    # update the active nodes
                    active_nodes = np.zeros(5)
                    active_nodes[i] = 1
                    # update the history
                    history = np.concatenate((history, [active_nodes]), axis=0)
                    # update time step
                    t = t + 1
                    # list of product for manage the parallel case
                    order_of_parallel_product.append(active_nodes)
        # return the history
        return history

    def run_social_influence_simulation(self, customer_class, aggregate_conversion):
        if customer_class == 2 and aggregate_conversion:
            for i in range(3):
                self.simulation()
        elif not aggregate_conversion:
            self.simulation()


