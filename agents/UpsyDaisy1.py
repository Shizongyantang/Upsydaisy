import random
import pickle
import os
import numpy as np
import scipy.stats as stats
import random


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
        self.alpha = 0.8 # Initial alpha
        self.delta_alpha1 = 0.05  # Increment for alpha when winning
        self.delta_alpha2 = -0.1 # Decrement for alpha when losing
        self.delta_alpha3 = 0.03
        self.delta_alpha4 = -0.1
        self.w1 = 0.4  # Weight for delta_alpha1
        self.w2 = 0.6  # Weight for delta_alpha2
        self.round_counter = 0  # Counter for rounds
        self.my_cumulative_profit = 0  # Cumulative profit for self
        self.opponent_cumulative_profit = 0  # Cumulative profit for opponent
        self.min_alpha = 0.8
        self.max_alpha = 1
        self.inc_count = 0
        self.dec_count = 0 
        # self.opponent_inc = False
        self.opponent_last_alpha = 0
        self.last_value = 0
        self.this_value = 0
        self.win_count = 0
        self.random_factor = 0.1
        
    def _process_last_sale(self, last_sale, profit_each_team):
            # Extract relevant information from the last sale
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]
        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]
        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        if self.round_counter >= 1:
            opponent_alpha = opponent_last_prices[0] /self.last_value
            if self.round_counter >= 2:
                if self.opponent_last_alpha < opponent_alpha:
                    self.inc_count += 1
                    self.dec_count = 0
                else:
                    self.dec_count += 1
                    self.inc_count = 0
            self.opponent_last_alpha = opponent_alpha
        
        
        if self.inc_count <= 10 and self.dec_count <= 10:
        # Adjust alpha based on last sale outcome
            if did_customer_buy_from_me:
                self.alpha +=  self.delta_alpha1
                self.win_count += 1
            elif did_customer_buy_from_opponent:
                self.alpha +=  self.delta_alpha2
            
            self.alpha = max(self.min_alpha, self.alpha)
            self.alpha = min(self.max_alpha, self.alpha)
        
        
        elif self.inc_count > 10:
            if did_customer_buy_from_me:
                self.alpha +=  (self.delta_alpha1 * self.w1 + self.delta_alpha3 * self.w2)
                self.win_count += 1
            elif did_customer_buy_from_opponent:
                self.alpha +=  (self.delta_alpha2 * self.w1 + self.delta_alpha3 * self.w2)
            
            self.alpha = max(self.min_alpha, self.alpha)
            self.alpha = min(self.max_alpha, self.alpha)
        
        else:
            if did_customer_buy_from_me:
                self.alpha +=  (self.delta_alpha1 * self.w1 + self.delta_alpha4 * self.w2)
                self.win_count += 1

            elif did_customer_buy_from_opponent:
                self.alpha +=  (self.delta_alpha2 * self.w1 + self.delta_alpha4 * self.w2)
            
            self.alpha = max(self.min_alpha, self.alpha)
            self.alpha = min(self.max_alpha, self.alpha)
        b = 2
        scale = 1
        pareto_dist = stats.pareto(b, scale=scale)
        cdf_value = pareto_dist.cdf(self.this_value/5)
        
        if cdf_value >= 0.95:
            self.alpha = 0.8 * self.alpha
        elif cdf_value >= 0.9:
            self.alpha = 0.9 * self.alpha
        elif cdf_value >= 0.8:
            self.alpha = 0.95 * self.alpha
        else:
            if random.random() < self.random_factor:
                self.alpha = random.random()
        



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


    def action(self, obs):
        new_buyer_covariates, last_sale, profit_each_team = obs
        self.this_value = new_buyer_covariates[0]
        self._process_last_sale(last_sale, profit_each_team)

        # Ensure that action method always returns a list of prices
        # prices = []  # Initialize an empty list for prices

        if self.project_part == 1:
            # For Part 1, return a list with a single price
            # Example: return [3]
            prices = [self.alpha * new_buyer_covariates[0]] # Replace 3 with your calculated price

        elif self.project_part == 2:
            # For Part 2, return a list with prices for each item
            # Example: return [price1, price2]
            predicted_prices = self.trained_model.predict(np.array(new_buyer_covariates).reshape(1, -1))[0]
            adjusted_prices = [price * self.alpha for price in predicted_prices]
            prices = adjusted_prices + random.random()  # Replace this with your calculated prices

        self.last_value = new_buyer_covariates[0]
        return prices  # Ensure that a list is always returned
        # Rest of the method...