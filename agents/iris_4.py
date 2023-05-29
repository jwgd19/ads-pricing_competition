import random
import pickle
import os
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from random import randint

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.project_part = params[
            'project_part']  # useful to be able to use same competition code for each project part
        self.n_items = params["n_items"]
        self.round_counter = 0
        self.opponent_history0 = []
        self.price_history0 = []
        self.behavior_history0 = []
        self.opponent_history1 = []
        self.price_history1 = []
        self.behavior_history1 = []
        self.agent_history = []
        self.model0 = []
        self.model1 = []
        self.oppo_alphas0 = []
        self.oppo_alphas1 = []
        self.is_normal = True
        self.lb = 0.6
        self.count_lower = 0
        self.ub0 = math.inf
        self.ub1 = math.inf

        # Potentially useful for Part 2 --
        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.qcut = 1000
        self.step_coef = 2
        self.filename_0 = 'agents/glowing_guacamole/trained_model_0'
        self.model_0 = pickle.load(open(self.filename_0, 'rb'))
        self.filename_1 = 'agents/glowing_guacamole/trained_model_1'
        self.model_1 = pickle.load(open(self.filename_1, 'rb'))
        self.alpha = 1
        self.training_range = pd.read_csv('data/train_prices_decisions.csv')
        self.prices_to_predict_0 = np.linspace(min(self.training_range['price_item_0']), max(self.training_range['price_item_0']), self.qcut)
        self.prices_to_predict_1 = np.linspace(min(self.training_range['price_item_1']), max(self.training_range['price_item_1']), self.qcut)

        self.demand_item_0 = []
        self.demand_item_1 = []

        self.cov_history = []

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        if not math.isnan(my_last_prices[0]):
            if not did_customer_buy_from_me and not did_customer_buy_from_opponent:
                price0 = min(my_last_prices[0], opponent_last_prices[0])
                price1 = min(my_last_prices[1], opponent_last_prices[1])
                if price0 < self.ub0:
                    self.ub0 = price0
                if price1 < self.ub1:
                    self.ub1 = price1


    def get_predict_model0(self):
        oh = np.array(self.opponent_history0)
        bh = np.array(self.behavior_history0)
        ph = np.array(self.price_history0[0:len(self.price_history0) - 1])
        cv = self.cov_history[0:-1]
        cov1 = [i[0] for i in cv]
        cov2 = [i[1] for i in cv]
        cov3 = [i[2] for i in cv]
        # ah = np.array(self.agent_history)
        X_train = np.column_stack([bh, ph, cov1, cov2, cov3])
        Y_train = oh.reshape(-1, 1)
        # print(X_train.shape)
        # print(Y_train.shape)
        reg = LinearRegression().fit(X_train, Y_train)
        # print("bh:", bh)
        # print("oh:", oh)
        # print("ph:", ph)
        # print("ah:", ah)
        # print("train:", X_train)
        return reg

    def get_predict_model1(self):
        oh = np.array(self.opponent_history1)
        bh = np.array(self.behavior_history1)
        ph = np.array(self.price_history1[0:len(self.price_history1) - 1])
        cv = self.cov_history[0:-1]
        cov1 = [i[0] for i in cv]
        cov2 = [i[1] for i in cv]
        cov3 = [i[2] for i in cv]
        # ah = np.array(self.agent_history)
        X_train = np.column_stack([bh, ph, cov1, cov2, cov3])
        Y_train = oh.reshape(-1, 1)
        # print(X_train.shape)
        # print(Y_train.shape)
        reg = LinearRegression().fit(X_train, Y_train)
        # print("bh:", bh)
        # print("oh:", oh)
        # print("ph:", ph)
        # print("ah:", ah)
        # print("train:", X_train)
        return reg

    def get_optimal(self, covariates):
        max_price_0, max_price_1 = 0,0
        max_rev = 0
        for i in range(100):
            rand_item_0 = randint(0, self.qcut - 1)
            rand_item_1 = randint(0, self.qcut - 1)

            list_covs = [covariates + [self.prices_to_predict_0[rand_item_0], self.prices_to_predict_1[rand_item_1]]]

            predicted_purchase_0 = self.get_demand(self.model_0, list_covs)[0]
            predicted_purchase_1 = self.get_demand(self.model_1, list_covs)[0]

            cur_rev_0 = predicted_purchase_0 * self.prices_to_predict_0[rand_item_0]
            cur_rev_1 = predicted_purchase_1 * self.prices_to_predict_1[rand_item_1]

            if (cur_rev_0 + cur_rev_1 > max_rev):
                max_rev = cur_rev_0 + cur_rev_1
                max_price_0 = rand_item_0
                max_price_1 = rand_item_1

        return max_price_0, max_price_1, max_rev


    #function for refining the price
    def refine_price(self, p0, p1, covariates):
        #sample another 11*11 points around the current price
        # demand_0 = self.demand_item_0
        # demand_1 = self.demand_item_1
        
        num_qcut = self.qcut
        rev_max = 0
        max_price_0 = p0
        max_price_1 = p1
        dir_x = [1, -1, 0, 1, -1, 0, 1, -1]
        dir_y = [0, 0, 1, 1, 1, -1, -1, -1]
    
        for step in range(5):
            for idx in range(len(dir_x)):
                next_x = p0 + dir_x[idx] * step
                next_y = p1 + dir_y[idx] * step
                
                if (next_x >= 0 and next_x < num_qcut and next_y >= 0 and next_y < num_qcut):
                    list_covs = [covariates + [self.prices_to_predict_0[next_x], self.prices_to_predict_1[next_y]]]
                    predicted_purchase_0 = self.get_demand(self.model_0, list_covs)[0]
                    predicted_purchase_1 = self.get_demand(self.model_1, list_covs)[0]

                    cur_rev_0 = predicted_purchase_0 * self.prices_to_predict_0[next_x]
                    cur_rev_1 = predicted_purchase_1 * self.prices_to_predict_1[next_y]

                    if (cur_rev_0 + cur_rev_1 > rev_max):
                        rev_max = cur_rev_0 + cur_rev_1
                        max_price_0 = next_x
                        max_price_1 = next_y

        return max_price_0, max_price_1, rev_max

    #function for getting demand of a specific price
    def get_demand(self, fitted_model, list_cov):
        prediction = pd.DataFrame(fitted_model.predict_proba(list_cov), columns = ["refused", "accepted"])
        return [prediction.iloc[i]['accepted'] for i in range(len(prediction))]

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items, indicating prices this agent is posting for each item.
    def action(self, obs):

        # For Part 1, new_buyer_covariates will simply be a vector of length 1, containing a single numeric float indicating the valuation the user has for the (single) item
        # For Part 2, new_buyer_covariates will be a vector of length 3 that can be used to estimate demand from that user for each of the two items
        new_buyer_covariates, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)

        # Potentially useful for Part 1 --
        # Currently output is just a deterministic price for the item, but students are expected to use the valuation (inside new_buyer_covariates) and history of prices from each team to set a better price for the item
        if self.project_part == 1:
            return [3]

        # Potentially useful for Part 2 --
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to set prices for each item.
        if self.project_part == 2:
            self.round_counter += 1

            #get optimal price and refine the result
            new_buyer_covariates = new_buyer_covariates.tolist()
            self.cov_history.append(new_buyer_covariates)
            tolerance = 0.1
            max_steps = 10
            #return index
            max_price_0, max_price_1, cur_max = self.get_optimal(new_buyer_covariates)

            while(max_steps > 0):
                r_price_0, r_price_1, r_max = self.refine_price(max_price_0, max_price_1, new_buyer_covariates)
                if (abs(r_max - cur_max) <= tolerance or r_max < cur_max):
                    break
                max_price_0, max_price_1, cur_max = r_price_0, r_price_1, r_max
                max_steps -= 1

            #refining using fixed step size
            max_price_0 = max(self.prices_to_predict_0[max_price_0], 0)
            max_price_1 = max(self.prices_to_predict_1[max_price_1], 0)

            # upper bound
            # max_price_0 = min(max_price_0, self.ub0)
            # max_price_1 = min(max_price_1, self.ub1)

            #--------------------------------------------- dividing line -------------------------------------------
            oh_coef0 = 0
            oh_coef1 = 0
            self.price_history0.append(max_price_0)
            self.price_history1.append(max_price_1)
            last_opponent_price = last_sale[2][self.opponent_number]
            last_agent_price = last_sale[2][self.this_agent_number]
            # sanitize the opponent_price
            for i in range(len(last_opponent_price)):
                if last_opponent_price[i] <= 0:
                    last_opponent_price[i] = 0
                if not math.isnan(last_opponent_price[0]):
                    if i == 0:
                        self.opponent_history0.append(last_opponent_price[i])
                    else:
                        self.opponent_history1.append(last_opponent_price[i])
                    self.agent_history.append(last_agent_price)

            recent_idx0 = 0
            if (len(self.opponent_history0) >= 1):
                for i in range(len(self.price_history0) - 2, -1, -1):
                    normalized_price = math.log(self.price_history0[i], 102)
                    cov_coef = self.cov_history[i][0] * self.cov_history[i][1] * self.cov_history[i][2] * normalized_price
                    oh_coef0 += self.opponent_history0[i] * (0.5 ** recent_idx0) * cov_coef
                    recent_idx0 += 1
                    if (recent_idx0 == 10):
                        break
                self.behavior_history0.append(oh_coef0)

            recent_idx1 = 0
            if (len(self.opponent_history1) >= 1):
                for i in range(len(self.price_history1) - 2, -1, -1):
                    normalized_price = math.log(self.price_history1[i], 102)
                    cov_coef = self.cov_history[i][0] * self.cov_history[i][1] * self.cov_history[i][2] * normalized_price
                    oh_coef1 += self.opponent_history1[i] * (0.5 ** recent_idx1) * cov_coef
                    recent_idx1 += 1
                    if (recent_idx1 == 10):
                        break
                self.behavior_history1.append(oh_coef1)

            if (self.round_counter > 10):
                # print("===========================")
                self.model0 = self.get_predict_model0()
                self.model1 = self.get_predict_model1()
            else:
                # for the beginning rounds, use default strategy
                # for the beginning rounds, use default strategy
                random_coef = random.uniform(0.1, 0.5)
                print("===============price0==================")
                return [max_price_0 * random_coef, max_price_1 * random_coef]

            if (self.model0 != []):
                predicted_price0 = self.model0.predict(np.array([[oh_coef0, max_price_0, 
                                        new_buyer_covariates[0],new_buyer_covariates[1], new_buyer_covariates[2]]]))[0][0]
            if (self.model1 != []):
                predicted_price1 = self.model1.predict(np.array([[oh_coef1, max_price_1, 
                                        new_buyer_covariates[0],new_buyer_covariates[1], new_buyer_covariates[2]]]))[0][0]

            if len(self.price_history0) > 1:
                opponent_alpha0 = last_opponent_price[0] / self.price_history0[-2]
                opponent_alpha1 = last_opponent_price[1] / self.price_history1[-2]
                self.oppo_alphas0.append(opponent_alpha0)
                self.oppo_alphas1.append(opponent_alpha1)
                # if np.mean(self.oppo_history[-10:]) < self.lb_price and self.is_normal:
                if (np.mean(self.oppo_alphas0[-5:]) < self.lb and np.mean(self.oppo_alphas1[-5:]) < self.lb) and self.is_normal:
                    self.count_lower += 1
                    if self.count_lower > 100:
                        self.is_normal = False
                    print("===============price1==================")
                    return [max_price_0 * 0.99, max_price_1 * 0.99]
                elif np.mean(self.oppo_alphas0[-5:]) > self.lb and np.mean(self.oppo_alphas1[-5:]) > self.lb:
                    self.count_lower = 0
                    self.is_normal = True

                #find out which item they most likely to sell
                if np.mean(self.oppo_alphas0[-5:]) < self.lb and np.mean(self.oppo_alphas1[-5:]) > self.lb:
                    predicted_price0 = predicted_price0 * 0.95
                    predicted_price1 = max_price_1 - 0.001
                elif np.mean(self.oppo_alphas1[-5:]) < self.lb and np.mean(self.oppo_alphas0[-5:]) > self.lb:
                    predicted_price0 = max_price_0 - 0.001
                    predicted_price1 = predicted_price1 * 0.95
            # predicted_price0 = predicted_price0 * 0.95  # * (lose_coef ** consecutive_lose)
            # predicted_price1 = predicted_price1 * 0.95  # * (lose_coef ** consecutive_lose)

            if (predicted_price0 > max_price_0):
                predicted_price0 = max_price_0 - 0.001
            if (predicted_price1 > max_price_1):
                predicted_price1 = max_price_1 - 0.001
            if (predicted_price0 < 0):
                predicted_price0 = max_price_0 * 0.99
            if (predicted_price1 < 0):
                predicted_price1 = max_price_1 * 0.99

            print("===============price2==================")

            return [predicted_price0 * 0.95, predicted_price1 * 0.95]
            #return self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0] + random.random()
