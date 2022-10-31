import numpy as np
import random


class SocialInfluence:
    def __init__(self, lambda_coeff, alpha_ratios, item_sold_mean, customer_class, price_configuration, n_prod,
                 configuration):
        self.n_users = customer_class.number_of_customers
        # reward of the simulation
        self.reward = 0.
        # lamdba decay coefficient
        self.lambda_coeff = lambda_coeff
        # customer class in the simulation
        self.customer_class = customer_class
        # configuration in the simulation
        self.configuration = price_configuration
        # number of products
        self.n_prod = n_prod
        # number of units sold per product in the simulation
        self.units_sold = np.zeros(5)
        # units sold per product in the simulation
        self.bought = np.zeros(5)
        # history of the simulation
        self.global_history = []
        # dirichlet distribution given alphas
        self.dirichlet_probs = alpha_ratios
        # users for every alpha
        self.actual_users = np.zeros(6)
        # mean of sold item
        self.item_sold_mean = item_sold_mean
        # configuration with price indexes
        self.configuration_indexes = configuration
        # initialize graph probabilities
        self.graph_probs = customer_class.graph_probabilities

    def simulation(self):
        # for each user, make a simulation
        for u in range(self.n_users):
            # assign initial product shown to the user, given the dirichlet distribution
            initial_products = np.random.multinomial(1, self.dirichlet_probs)
            self.actual_users[np.where(initial_products == 1)[0]] += 1
            if initial_products[0] == 0:
                initial_products = initial_products[1:]
                # make a simulation, append history
                self.global_history.append(self.graph_search(initial_products))
        # evaluate reward of the simulation
        self.evaluate_reward()

    def evaluate_reward(self):
        # assign zero to the variable (init)1
        self.reward = 0.
        # for each product
        for product in range(5):
            # evaluate the reward for
            self.reward += self.units_sold[product] * self.configuration[product]
        # average reward over all the simulations
        self.reward = self.reward / self.n_users

    def graph_search(self, initial_active_nodes):
        # store_the number of products, i.e, the nodes of the graph
        prob_matrix = np.copy(self.graph_probs)
        # init number of clicks array
        n_clicks = np.zeros(5)
        # initialize the history
        history = []
        # initialize the index of the activated node
        index = 0
        # initialize the history of the activated edges in the simulation
        activated_edges_simulation = [False for _ in range(5)]
        # assign the active nodes to the initial active one(s)
        active_nodes = initial_active_nodes
        # assign the new active nodes (nodes active at the end of each iteration) to the initial active one(s)
        newly_active_nodes = active_nodes
        # assign value to initial time step
        t = 0
        # assign value to n_steps_max,
        n_steps_max = 5
        # check if the first product displayed has a price higher than the reservation price; if yes, return
        if self.customer_class.reservation_prices[0][int(np.where(active_nodes == 1)[0])] < self.configuration[int(
                np.where(active_nodes == 1)[0])]:
            return history
        # assign first the history to the initial active nodes array
        history = np.array([initial_active_nodes])
        # while the number of max steps is not reached and there are still active nodes, continue simulation
        order_of_parallel_product = [active_nodes]
        while t < n_steps_max and len(order_of_parallel_product) != 0:
            index = np.where(order_of_parallel_product[0] == 1)[0][0]
            # assign at random value the first secondary node
            # first_secondary_node = np.random.randint(0, 5)
            first_secondary_node = random.choices(range(len(prob_matrix[index, :])), prob_matrix[index, :], k=1)[0]
            # assign at random value the second secondary node
            second_secondary_node = random.choices(range(len(prob_matrix[index, :])), prob_matrix[index, :], k=1)[0]
            # repeat assignment until the two products are different
            while second_secondary_node == first_secondary_node:
                second_secondary_node = random.choices(range(len(prob_matrix[index, :])), prob_matrix[index, :], k=1)[0]
            # select from the probability matrix only the rows related to the active nodes
            transition_probabilities_from_the_active_node = (prob_matrix.T * order_of_parallel_product[0]).T
            # select from the probability rows just the ones related to the active node
            p_row = transition_probabilities_from_the_active_node[index]
            if np.all(activated_edges_simulation):  # if all activated_edges_simulation == True
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
            # check if the clicked node is already activated in activated_edges_simulation
            for z in range(num_prod_clicked):
                if activated_edges_simulation[indx[z]] is False:  # if indx[z] is not in activated_edges_simulation
                    activated_edges[indx[z]] = True
                    activated_edges_simulation[indx[z]] = True
            # if ((p != 0) == activated_edges) it is False, empty the matrix
            # assign 0 to the new active nodes (reset values)
            newly_active_nodes = np.zeros(5)
            # for each product, find the chosen one
            for i in range(5):
                # if the chosen secondary product is found, let the costumer actually buy it, by updated the
                # values related to the units sold and the reven
                if activated_edges[i] and self.customer_class.reservation_prices[0][i] >= self.configuration[i]:
                    # assign a random amount of units of product purchased by the user
                    units_purchased = np.random.randint(1, max(2, (self.item_sold_mean[i][self.configuration_indexes[i]]*2)))
                    # update the amount of unites of product purchased by the class of user
                    self.units_sold[i] += units_purchased
                    self.bought[i] += 1
                    # assign 1 to the new active nodes
                    newly_active_nodes[i] = 1
            # update transition probability of the new active nodes to zero value so that it is not possible to
            # reach again the same node (product)
            order_of_parallel_product.pop(0)
            for i in range(prob_matrix.shape[1]):
                if i in np.array(np.where(newly_active_nodes == 1)):
                    index = i
                    # from all the nodes to the new active one
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

    def abrupt_simulation(self, phase):
        # for each user, make a simulation
        for u in range(self.n_users):
            # assign initial product shown to the user, given the dirichlet distribution
            initial_products = np.random.multinomial(1, self.dirichlet_probs)
            self.actual_users[np.where(initial_products == 1)[0]] += 1
            if initial_products[0] == 0:
                initial_products = initial_products[1:]
                # make a simulation, append history
                self.global_history.append(self.abrupt_changes_graph_search(initial_products, phase))
        # evaluate reward of the simulation
        self.evaluate_reward()

    def abrupt_changes_graph_search(self, initial_active_nodes, phase):
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
        if self.customer_class.reservation_prices[phase][int(np.where(active_nodes == 1)[0])] < self.configuration[int(
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
                if activated_edges[i] and self.customer_class.reservation_prices[phase][i] >= self.configuration[i]:
                    # assign a random amount of units of product purchased by the user
                    units_purchased = np.random.randint(1, (self.item_sold_mean[i][self.configuration_indexes[i]] * 2))
                    # update the amount of unites of product purchased by the class of user
                    self.units_sold[i] += units_purchased
                    self.bought[i] += 1
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



