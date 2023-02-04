import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''

        if dead:
            reward = -1.0
        elif points > self.points:
            reward = 1.0
        else:
            reward = -0.1
        
        #next state
        s_prime = self.generate_state(environment)
        #current state + action
        
        #find max Q value based on current STATE but different ACTIONS
        q_max = -100
        f_vals = [-1] * 4 #for choosing next optimal action
        if self.train:    
            if self.s is not None and self.a is not None:
                
                for i in range(4):
                    c_idx = s_prime + (i,) # given the current pair of (state s_prime, and action i)
                    if self.Q[c_idx] > q_max:
                        q_max = self.Q[c_idx]
                
                curr_st_act = self.s + (self.a,)
                self.N[curr_st_act] += 1
                # optimal action based on the index
                
                '''Part B: Update Q(s,a) based on Q(s',a') using Q update formula'''
                act_reward = reward #NOTE: Reward based future (current) environment
                lr = self.C / (self.C + self.N[curr_st_act])
                temp_diff  = act_reward + self.gamma * q_max - self.Q[curr_st_act]
                self.Q[curr_st_act] = self.Q[curr_st_act] + lr * temp_diff # NOTE: Q update
            
            for i in range(4):
                c_idx = s_prime + (i,)
                if self.N[c_idx] < self.Ne:
                    f_vals[i] = 1
                else:
                    f_vals[i] = self.Q[c_idx]
            
            
        #NOTE: IF THIS IS IN TESTING PHASE:
        else:
            #Only exploit
            for i in range(4):
                f_vals[i] = self.Q[s_prime][i]
                
        # for tie breaking by taking max from right if there are same values
        mx = -100
        opt_next_action=None
        
        
        for i in range(4):
            if f_vals[i] >= mx:
                mx = f_vals[i]
                opt_next_action = i
        #update self.s and self.a:
        if dead:
            self.reset()
        else:
            self.s = s_prime
            self.a = opt_next_action
            self.points = points
        
        return self.a
            

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        #also update points/ dead 
        head_x = environment[0]
        head_y = environment[1]
        body_pos = set(environment[2])
        food_x = environment[3]
        food_y = environment[4]
        
        food_dir_x = 3
        food_dir_y = 3
        adj_wall_x = 0
        adj_wall_y = 0
        adj_body_top = 0
        adj_body_bot = 0
        adj_body_left = 0
        adj_body_right = 0
        
        #check if next to body        
        for i, pos in enumerate(self.surrounding_pos(head_x, head_y)):
            if pos in body_pos:
                if i == 0: #right
                    adj_body_right = 1
                elif i == 1: #top
                    adj_body_top = 1
                elif i ==2: # left
                    adj_body_left = 1
                else: # bot
                    adj_body_bot = 1
                    
        #check food x dir
        if food_x == head_x:
            food_dir_x = 0
        elif food_x < head_x:
            food_dir_x = 1
        else:
            food_dir_x = 2
        #check food y dir
        if food_y == head_y:
            food_dir_y = 0
        elif food_y < head_y:
            food_dir_y = 1
        else:
            food_dir_y = 2
        #check adj_wall_x:
        if head_x == 1:
            adj_wall_x = 1
        elif head_x == utils.DISPLAY_WIDTH-2:
            adj_wall_x = 2
        #check adj_wall_y:
        if head_y == 1:
            adj_wall_y = 1
        elif head_y == utils.DISPLAY_HEIGHT-2:
            adj_wall_y = 2
            
        return (food_dir_x, food_dir_y, adj_wall_x, adj_wall_y, adj_body_top, adj_body_bot, adj_body_left, adj_body_right)
    
    def surrounding_pos(self, xpos, ypos):
        #Returns a list of <=4 (x,y) positions indicating the right, down, left, up of the current position
        return [(xpos+1, ypos), (xpos, ypos-1), (xpos-1, ypos), (xpos, ypos+1)]
    
    # def get_left_env(self, environment):
    #     new_body = environment[2].copy()
    #     new_body.pop()    
    #     new_body.insert(0, (environment[0], environment[1]))
    #     new_env = [environment[0]-1, environment[1], new_body, environment[3], environment[4]]
    #     return new_env
    # def get_right_env(self, environment):
    #     new_body = environment[2].copy()
    #     new_body.pop()
    #     new_body.insert(0, (environment[0], environment[1]))
    #     new_env = [environment[0]+1, environment[1], new_body, environment[3], environment[4]]
    #     return new_env
    # def get_bot_env(self, environment):
    #     new_body = environment[2].copy()
    #     new_body.pop()
    #     new_body.insert(0, (environment[0], environment[1]))
    #     new_env = [environment[0], environment[1]-1, new_body, environment[3], environment[4]]
    #     return new_env
    # def get_top_env(self, environment):
    #     new_body = environment[2].copy()
    #     new_body.pop()
    #     new_body.insert(0, (environment[0], environment[1]))
    #     new_env = [environment[0]-1, environment[1]+1, new_body, environment[3], environment[4]]
    #     return new_env
    
    # def reward(self, environment): # reward/ penalty at this position
    #     head_x = environment[0]
    #     head_y = environment[1]
    #     body_pos = set(environment[2])
    #     food_x = environment[3]
    #     food_y = environment[4]
        
    #     #if it hits obstacles or wall
    #     if head_x == 0 or head_x == utils.DISPLAY_WIDTH-1 or \
    #         head_y == 0 or head_y == utils.DISPLAY_HEIGHT-1 or \
    #         (head_x, head_y) in body_pos:
    #             return -1
        
    #     # if head hits food
    #     if head_x == food_x and head_y == food_y:
    #         self.points += 1
    #         return 1
        
    #     #otherwise
    #     return -0.1