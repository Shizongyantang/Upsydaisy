import random
import pickle
import os
import numpy as np
import time
import xgboost as xgb
import pandas as pd

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
        # self.trained_model = pickle.load(open(self.filename, 'rb'))
        self.model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)
        self.model.load_model('agents/UpsyDaisy/best_model.xgb')
        self.last_value0 = 0
        self.last_value1 = 0
        self.this_value0 = 0
        self.this_value1 = 0
        self.win_count = 0
        self.random_factor = 0.1
        self.opponent_last_alphaA = 0
        self.opponent_last_alphaB = 0
        
        self.alphaA = 0.8
        self.alphaB = 0.8
        
        self.delta_alpha1 = 0.05  # Increment for alpha when winning
        self.delta_alpha2 = -0.1 # Decrement for alpha when losing
        self.delta_alpha3 = 0.03
        self.delta_alpha4 = -0.1
        self.w1 = 0.6  # Weight for delta_alpha1
        self.w2 = 0.4  # Weight for delta_alpha2
        self.round_counter = 0
        self.round_counterA = 0  # Counter for rounds
        self.round_counterB = 0
        self.my_cumulative_profit = 0  # Cumulative profit for self
        self.opponent_cumulative_profit = 0  # Cumulative profit for opponent
        self.min_alpha = 0.8
        self.max_alpha = 1
        self.inc_countA = 0
        self.dec_countA = 0
        self.inc_countB = 0
        self.dec_countB = 0 

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]
        #my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]
        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        if self.round_counter >= 1:
            if which_item_customer_bought == 0:
                opponent_alpha = (opponent_last_prices[0] /self.last_value0)
                if self.round_counter >= 2:
                    if self.opponent_last_alphaA < opponent_alpha:
                        self.inc_countA += 1
                        self.dec_countA = 0
                    else:
                        self.dec_countA += 1
                        self.inc_countA = 0
                self.opponent_last_alphaA = opponent_alpha
            else:
                opponent_alpha = (opponent_last_prices[1] /self.last_value1)
                if self.round_counter >= 2:
                    if self.opponent_last_alphaB < opponent_alpha:
                        self.inc_countB += 1
                        self.dec_countB = 0
                    else:
                        self.dec_countB += 1
                        self.inc_countB = 0
                self.opponent_last_alphaB = opponent_alpha
        
        if which_item_customer_bought == 0:
            if self.inc_countA <= 10 and self.dec_countA <= 10:
            # Adjust alpha based on last sale outcome
                if did_customer_buy_from_me:
                    self.alphaA +=  self.delta_alpha1
                    self.win_count += 1
                elif did_customer_buy_from_opponent:
                    self.alphaA +=  self.delta_alpha2
                
                self.alphaA = max(self.min_alpha, self.alphaA)
                self.alphaA = min(self.max_alpha, self.alphaA)
        
            elif self.inc_countA > 10:
                if did_customer_buy_from_me:
                    self.alphaA +=  (self.delta_alpha1 * self.w1 + self.delta_alpha3 * self.w2)
                    self.win_count += 1
                elif did_customer_buy_from_opponent:
                    self.alphaA +=  (self.delta_alpha2 * self.w1 + self.delta_alpha3 * self.w2)
                
                self.alphaA = max(self.min_alpha, self.alphaA)
                self.alphaA = min(self.max_alpha, self.alphaA)
            
            else:
                if did_customer_buy_from_me:
                    self.alphaA +=  (self.delta_alpha1 * self.w1 + self.delta_alpha4 * self.w2)
                    self.win_count += 1

                elif did_customer_buy_from_opponent:
                    self.alphaA +=  (self.delta_alpha2 * self.w1 + self.delta_alpha4 * self.w2)
                
                self.alphaA = max(self.min_alpha, self.alphaA)
                self.alphaA = min(self.max_alpha, self.alphaA)
                
        else:
            if self.inc_countB <= 10 and self.dec_countB <= 10:
            # Adjust alpha based on last sale outcome
                if did_customer_buy_from_me:
                    self.alphaB +=  self.delta_alpha1
                    self.win_count += 1
                elif did_customer_buy_from_opponent:
                    self.alphaB +=  self.delta_alpha2
                
                self.alphaB = max(self.min_alpha, self.alphaB)
                self.alphaB = min(self.max_alpha, self.alphaB)
        
            elif self.inc_countB > 10:
                if did_customer_buy_from_me:
                    self.alphaB +=  (self.delta_alpha1 * self.w1 + self.delta_alpha3 * self.w2)
                    self.win_count += 1
                elif did_customer_buy_from_opponent:
                    self.alphaB +=  (self.delta_alpha2 * self.w1 + self.delta_alpha3 * self.w2)
                
                self.alphaB = max(self.min_alpha, self.alphaB)
                self.alphaB = min(self.max_alpha, self.alphaB)
            
            else:
                if did_customer_buy_from_me:
                    self.alphaB +=  (self.delta_alpha1 * self.w1 + self.delta_alpha4 * self.w2)
                    self.win_count += 1

                elif did_customer_buy_from_opponent:
                    self.alphaB +=  (self.delta_alpha2 * self.w1 + self.delta_alpha4 * self.w2)
                
                self.alphaB = max(self.min_alpha, self.alphaB)
                self.alphaB = min(self.max_alpha, self.alphaB)
        
        
        value_max = max(self.this_value0, self.this_value1)
        if value_max >= 84:
            self.alphaA = 0.85 * self.alphaA
            self.alphaB = 0.85 * self.alphaB

        elif value_max >= 67:
            self.alphaA = 0.9 * self.alphaA
            self.alphaB = 0.9 * self.alphaB

        elif value_max >= 50:
            self.alphaA = 0.95 * self.alphaA
            self.alphaB = 0.95 * self.alphaB

        else:
            if random.random() < self.random_factor:
                self.alphaA = random.random()
                self.alphaB = random.random()

        self.my_cumulative_profit += my_current_profit
        self.opponent_cumulative_profit += opponent_current_profit

            # Adjust strategy every 50 rounds
        self.round_counter += 1
        if self.round_counter % 50 == 0:
            if self.my_cumulative_profit > self.opponent_cumulative_profit:
                # Become more aggressive
                self.delta_alpha1 += 0.0001
                self.delta_alpha2 += 0.0005
                self.delta_alpha3 += 0.0005
                self.delta_alpha4 += 0.001
            else:
                    # Become more conservative
                self.delta_alpha1 -= 0.005
                self.delta_alpha2 -= 0.005
                self.delta_alpha3 -= 0.01
                self.delta_alpha4 -= 0.01
            
            self.delta_alpha1 = max(0, self.delta_alpha1)
            self.delta_alpha2 = min(0, self.delta_alpha2)
            self.delta_alpha3 = max(0, self.delta_alpha3)
            self.delta_alpha4 = min(0, self.delta_alpha4)
            self.my_cumulative_profit = 0 
            self.opponent_cumulative_profit = 0 
            if self.win_count < 20:
                self.min_alpha -= 0.05
                self.w1 += 0.05
                self.w2 -= 0.05
                self.w1 = min(1, self.w1)
                self.w2 = max(0, self.w2)
            if self.win_count > 30:
                self.min_alpha += 0.05
                self.w1 -= 0.05
                self.w2 += 0.05
                self.w2 = min(1, self.w1)
                self.w1 = max(0, self.w2)
            self.win_count = 0

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items, indicating prices this agent is posting for each item.
    def action(self, obs):

        # For Part 1, new_buyer_covariates will simply be a vector of length 1, containing a single numeric float indicating the valuation the user has for the (single) item
        # For Part 2, new_buyer_covariates will be a vector of length 3 that can be used to estimate demand from that user for each of the two items
        new_buyer_covariates, last_sale, profit_each_team = obs
        
        #item = pd.DataFrame([new_buyer_covariates], columns = ['Covariate1','Covariate2','Covariate3'])
        price_pair, _ = self.predict_optimal(item = list(new_buyer_covariates))
        value0 = price_pair[0]
        value1 = price_pair[1]
        self.this_value0 = value0
        self.this_value1 = value1
        self._process_last_sale(last_sale, profit_each_team)

        # Potentially useful for Part 1 --
        # Currently output is just a deterministic price for the item, but students are expected to use the valuation (inside new_buyer_covariates) and history of prices from each team to set a better price for the item
        if self.project_part == 1:
            prices = [3]

        # Potentially useful for Part 2 -- 
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to set prices for each item.
        if self.project_part == 2:
            prices = [self.alphaA * value0, self.alphaB * value1]
        
        self.last_value0 = value0
        self.last_value1 = value1
        return prices
    
    def predict_optimal(self, item, item0_max = 90, item0_min = 3, item1_max = 133, item1_min = 3):
        #index = item.index[0]
        revenue_diff = np.inf
        old_revenue = 0
        new_revenue = 0
        price_pair = []
        item0_range = np.linspace(item0_min, item0_max, 8)
        item1_range = np.linspace(item1_min, item1_max, 8)
        max_revenue = 0
        for i in item0_range:
            for j in item1_range:
                #item.loc[index, 'price_item_0'] = i
                #item.loc[index, 'price_item_1'] = j
                demand = self.model.predict_proba([item+[i,j]])
                demand0 = demand[0][0]
                demand1 = demand[0][1]
                revenue = i * demand0 + j * demand1
                if revenue > max_revenue:
                    max_revenue = revenue
                    price_pair = [i,j]
        
        # print('Optimal Price Pair', price_pair)
        # print('Optimal Revenue', max_revenue) 
        new_revenue = max_revenue
        revenue_diff = new_revenue

        while revenue_diff >= 5:
            max_revenue = 0
            interval1 = item0_range[1] - item0_range[0]
            interval2 = item1_range[1] - item1_range[0]
            item0_range = np.linspace(price_pair[0] - interval1/2, price_pair[0] + interval1/2, 5)
            item1_range = np.linspace(price_pair[1] - interval2/2, price_pair[1] + interval2/2, 5)
            price_pair = []

            for i in item0_range:
                for j in item1_range:
                    #item.loc[index, 'price_item_0'] = i
                    #item.loc[index, 'price_item_1'] = j
                    demand = self.model.predict_proba([item+[i,j]])
                    demand0 = demand[0][0]
                    demand1 = demand[0][1]
                    revenue = i*demand0 + j * demand1
                    if revenue > max_revenue:
                        max_revenue = revenue
                        price_pair = [i,j]
            old_revenue = new_revenue
            new_revenue = max_revenue
            revenue_diff = new_revenue - old_revenue
        # print('Running Time is', end_time-start_time)
        return price_pair, new_revenue
                