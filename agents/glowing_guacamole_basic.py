import random
import pickle
import os
import numpy as np
import pandas as pd
from random import randint

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.project_part = params['project_part'] #useful to be able to use same competition code for each project part
        self.n_items = params["n_items"]
        

        # Potentially useful for Part 2 -- 
        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.qcut = 20
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
        if did_customer_buy_from_me:  # can increase prices
            self.alpha *= 1.1
        else:  # should decrease prices
            if self.alpha > 0.3:
                self.alpha *= 0.9

    def get_optimal (self):
        max_price_0, max_rev_0 = 0,0
        max_price_1, max_rev_1 = 0,0
        for i in range (200):
            rand_item_0 = randint(0,self.qcut-1)
            rand_item_1 = randint(0,self.qcut-1)
            prediction_0 = self.demand_item_0[rand_item_0 * self.qcut + rand_item_1]
            prediction_1 = self.demand_item_1[rand_item_0 * self.qcut + rand_item_1]

            cur_rev_0 = prediction_0 * self.prices_to_predict_0[rand_item_0]
            cur_rev_1 = prediction_1 * self.prices_to_predict_1[rand_item_1]

            if (cur_rev_0 > max_rev_0):
                max_rev_0 = cur_rev_0
                max_price_0 = self.prices_to_predict_0[rand_item_0]
            
            if (cur_rev_1 > max_rev_1):
                max_rev_1 = cur_rev_1
                max_price_1 = self.prices_to_predict_1[rand_item_1]
        
        return max_price_0, max_rev_0, max_price_1, max_rev_1
            
    
    #function for getting demand of a specific price
    def get_demand(self, fitted_model, list_cov):
        return pd.DataFrame(fitted_model.predict_proba(list_cov), columns = ["refused", "accepted"])

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
            list_covs = []
            for i in range(len(self.prices_to_predict_0)):
                price_0 = self.prices_to_predict_0[i]
                for j in range(len(self.prices_to_predict_1)):
                    price_1 = self.prices_to_predict_0[j]
                    list_covs.append(new_buyer_covariates.tolist() + [price_0, price_1])
                    
            predicted_purchase_0 = self.get_demand(self.model_0, list_covs)
            predicted_purchase_1 = self.get_demand(self.model_1, list_covs)

            self.demand_item_0 = [predicted_purchase_0.iloc[i]['accepted'] for i in range(len(predicted_purchase_0))]
            self.demand_item_1 = [predicted_purchase_1.iloc[i]['accepted'] for i in range(len(predicted_purchase_1))]
            
            max_price_0, max_rev_0, max_price_1, max_rev_1 = self.get_optimal()
            max_price_0 = max(max_price_0, 0)
            max_price_1 = max(max_price_1, 0)


            return [max_price_0 * self.alpha, max_price_1 * self.alpha]