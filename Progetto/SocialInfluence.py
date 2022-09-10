import numpy as np
import random
from CustomerClass import CustomerClass


class SocialInfluence:
    def __init__(self):
        self.a = 0

    def ecommerce_user_simulation(self, number_of_products, initial_active_nodes, price_configuration, price_campaign,
                                  customer_class):
        # assign a value for the lambda coefficient for the second secondary product
        lambda_coefficient = 0.5
        # store probability_matrix of the current customer class
        prob_matrix = 0
        prob_matrix = np.copy(CustomerClass(customer_class).social_influence_transition_probability_matrix)
        np.fill_diagonal(prob_matrix, 0)
        # store_the number of products, i.e, the nodes of the graph
        n_nodes = number_of_products
        # assign first the history to the initial active nodes array
        history = np.array([initial_active_nodes])
        # assign the active nodes to the initial active one(s)
        active_nodes = initial_active_nodes
        # assign the new active nodes (nodes active at the end of each iteration) to the initial active one(s)
        newly_active_nodes = active_nodes
        # assign value to initial time step
        t = 0
        # assign value to n_steps_max,
        n_steps_max = 5
        # check if the first product displayed has a price higher than the reservation price; if yes, return
        if CustomerClass(customer_class).reservation_prices[int(np.where(active_nodes == 1)[0])] < price_configuration[int(
                np.where(active_nodes == 1)[0])]:
            return history
        # update transition probability of the new active nodes to zero value so that it is not possible to reach
        # again the same node (product)
        for i in range(prob_matrix.shape[1]):
            if i == int(np.where(newly_active_nodes == 1)[0]):
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
            p_row[second_secondary_node] = p_row[second_secondary_node] * lambda_coefficient
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
                if (activated_edges[i] == True and CustomerClass(customer_class).reservation_prices[i] >=
                        price_configuration[i]):
                    # assign a random amount of units of product purchased by the user
                    units_purchased = np.random.randint(1, 20)
                    # update the amount of unites of product purchased by the class of user
                    CustomerClass(customer_class).units_purchased_for_each_product[i] += units_purchased
                    # if the purchase cap is not reached, then activate the node to continue the simulation
                    if (CustomerClass(customer_class).max_number_of_purchases >= sum(
                            CustomerClass(customer_class).units_purchased_for_each_product)):
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

    # a function to evaluate the conversion rate of each customer class, by computing the number of occurrences in
    # global history (n. of sales) over the number of customers belonging to the class

    def evaluate_conversion_rate(self, customer_class, price_campaign, price_configuration):
        # check if the global history is not null (just to be sure)
        if price_campaign.global_history[customer_class] is not None:
            # for each history (relative to one costumer) in the global history
            for history in price_campaign.global_history[customer_class]:
                # check if the current step in global history is not null (just to be sure)
                if history is not None:
                    # for each step in the history
                    for history_step in history:
                        # if contains a 1, i.e., a product was bought
                        if 1 in history_step:
                            # increase the global number of sales of the customer class involved
                            price_campaign.sales[customer_class] += 1
            # at the end of the loop, evaluate the number of no-sales, by subtracting the number of sales from the
            # total number of customers
            price_campaign.no_sales[customer_class] = CustomerClass(customer_class).number_of_customers - price_campaign.sales[customer_class]
            # the conversion rate is equal to the number of sales over the number of customer of the current class.
            # the conversion rate is relative to the whole price campaign (price configuration)
            conversion = price_campaign.sales[customer_class] / CustomerClass(customer_class).number_of_customers
            # store the value of the conversion rate
            price_campaign.conversion_rate[customer_class] = conversion
            return conversion

    def evaluate_aggregate_conversion_rate(self, price_campaign, price_configuration):
        total_sales = 0
        # check if the global history is not null (just to be sure)
        for customer_class in range(3):
            if price_campaign.global_history[customer_class] is not None:
                # for each history (relative to one costumer) in the global history
                for history in price_campaign.global_history[customer_class]:
                    # check if the current step in global history is not null (just to be sure)
                    if history is not None:
                        # for each step in the history
                        for history_step in history:
                            # if contains a 1, i.e., a product was bought
                            if 1 in history_step:
                                # increase the global number of sales of the customer class involved
                                price_campaign.sales[customer_class] += 1
                                total_sales += 1
            price_campaign.aggregate_sales = total_sales
            # at the end of the loop, evaluate the number of no-sales, by subtracting the number of sales from the
            # total number of customers
            total_conversion = total_sales / (CustomerClass(0).number_of_customers + CustomerClass(1).number_of_customers + CustomerClass(2).number_of_customers)
            # the conversion rate is equal to the number of sales over the number of customer of the current class.
            # the conversion rate is relative to the whole price campaing (price configuration) store the value of
            # the conversion rate
            price_campaign.aggregate_conversion_rate = total_conversion
            return total_conversion

    # a function to evaluate the actual profit relative to one price configuration
    def evaluate_profit(self, customer_class, price_campaign, price_configuration):
        # assign zero to the variable (init)
        total_purchase_revenue = 0
        # for each product
        for product in range(5):
            # evaluate the profit (margin) by multiplying the units purchased (of each product) by their average margin
            total_purchase_revenue += customer_class.units_purchased_for_each_product[product] * \
                                      price_campaign.average_margin_for_price_in_configuration[product]
            # reset value
            customer_class.units_purchased_for_each_product[product] = 0
        return total_purchase_revenue

    def simulation(self, number_of_products, price_configuration, customer_class, price_campaign, aggregate_conversion):
        n = int(CustomerClass(customer_class).alpha_probabilities[0][0] * CustomerClass(customer_class).number_of_customers)
        # for each user belonging to the current class
        for j in range(n):
            # set initial active node, i.e., the first product shown to the user at random
            initial_active_nodes = np.zeros(number_of_products)
            initial_active_nodes[
                np.random.choice(np.arange(0, 5), CustomerClass(customer_class).alpha_probabilities[1:5])] = 1
            # simulate the ecommerce navigation for a user and store the history
            result = self.ecommerce_user_simulation(number_of_products, initial_active_nodes, price_configuration,
                                                    price_campaign, customer_class)
            # append the history to the global history of the customer class
            price_campaign.global_history[customer_class].append(result)
        # evaluate the conversion rate for the current customer class and price configuration
        conversion_rate = self.evaluate_conversion_rate(customer_class, price_campaign, price_configuration)
        # evaluate the marginal profit of the current price campaign (price configuration) and customer class
        price_campaign.marginal_profit[customer_class] = self.evaluate_profit(CustomerClass(customer_class),
                                                                              price_campaign, price_configuration)
        if not aggregate_conversion:
            print(
                'For customer class {0} and configuration {1}, the conversion rate is {2} and the profit is {3}'.format(
                    customer_class, price_campaign.id, conversion_rate, price_campaign.marginal_profit[customer_class]))

    def run_social_influence_simulation(self, number_of_products, price_configuration, customer_class, price_campaign,
                                        aggregate_conversion):
        if customer_class == 3 and aggregate_conversion:
            for customer_class in range(3):
                self.simulation(number_of_products, price_configuration, customer_class, price_campaign,
                                aggregate_conversion)
            aggregate_conversion_rate = self.evaluate_aggregate_conversion_rate(price_campaign, price_configuration)
            print('For configuration {0}, the aggregate conversion rate is {1}.'.format(price_campaign.id,
                                                                                        aggregate_conversion_rate))
        else:
            self.simulation(number_of_products, price_configuration, customer_class, price_campaign,
                            aggregate_conversion)
