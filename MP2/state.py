from utils import compute_mst_cost, is_english_word, levenshteinDistance
from abc import ABC, abstractmethod
from copy import deepcopy
# NOTE: using this global index means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0...
from itertools import count
global_index = count()

# TODO(III): You should read through this abstract class
#           your search implementation must work with this API,
#           namely your search will need to call is_goal() and get_neighbors()
class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f(state) = g(start, state) + h(state, goal)
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of AbstractState objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from each state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # The "less than" method ensures that states are comparable, meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # The "hash" method allow us to keep track of which states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass
    
# WordLadder ------------------------------------------------------------------------------------------------

# TODO(III): we've provided you most of WordLadderState, read through our comments and code below.
#           The only thing you must do is fill in the WordLadderState.__lt__(self, other) method
class WordLadderState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic):
        '''
        state: string of length n
        goal: string of length n
        dist_from_start: integer
        use_heuristic: boolean
        '''
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # Each word can have the following neighbors:
    #   Every letter in the word (self.state) can be replaced by every letter in the alphabet
    #   The resulting word must be a valid English word (i.e., in our dictionary)
    def get_neighbors(self):
        '''
        Return: a list of WordLadderState
        '''
        nbr_states = []
        for word_idx in range(len(self.state)):
            prefix = self.state[:word_idx]
            suffix = self.state[word_idx+1:]
            # 'a' = 97, 'z' = 97 + 25 = 122
            for c_idx in range(97, 97+26):
                c = chr(c_idx) # convert index to character
                # Replace the character at word_idx with c
                potential_nbr = prefix + c + suffix
                # If the resulting word is a valid english word, add it as a neighbor
                if is_english_word(potential_nbr):
                    # NOTE: the distance from start of a neighboring state is 1 more than the distance from current state
                    new_state = WordLadderState(potential_nbr, self.goal, 
                                                dist_from_start=self.dist_from_start + 1, use_heuristic=self.use_heuristic)
                    nbr_states.append(new_state)
        return nbr_states

    # Checks if we reached the goal word with a simple string equality check
    def is_goal(self):
        return self.state == self.goal
    
    # Strings are hashable, directly hash self.state
    def __hash__(self):
        return hash(self.state)
    def __eq__(self, other):
        return self.state == other.state
    
    # The heuristic we use is the edit distance (Levenshtein) between our current word and the goal word
    def compute_heuristic(self):
        return levenshteinDistance(self.state, self.goal)
    
    # TODO(III): implement this method
    def __lt__(self, other):    
        # You should return True if the current state has a lower g + h value than "other"
        # If they have the same value then you should use tiebreak_idx to decide which is smaller
        self_f = self.dist_from_start + self.h
        other_f = other.dist_from_start + other.h
        if self_f == other_f:
            return super().__lt__(other)
        else:
            return self_f < other_f
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return self.state
    def __repr__(self):
        return self.state

# EightPuzzle ------------------------------------------------------------------------------------------------

# TODO(IV): implement this method (also need it for parts V and VI)
# Manhattan distance between two points (a=(a1,a2), b=(b1,b2))
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class EightPuzzleState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, zero_loc):
        '''
        state: 3x3 array of integers 0-8
        goal: 3x3 goal array, default is np.arange(9).reshape(3,3).tolist()
        zero_loc: an additional helper argument indicating the 2d index of 0 in state, you do not have to use it
        '''
        # NOTE: AbstractState constructor does not take zero_loc
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.zero_loc = zero_loc
    
    # TODO(IV): implement this method
    def get_neighbors(self):
        '''
        Return: a list of EightPuzzleState
        '''
        nbr_states = []
        # NOTE: There are *up to 4* possible neighbors and the order you add them matters for tiebreaking
        #   Please add them in the following order: [below, left, above, right], where for example "below" 
        #   corresponds to moving the empty tile down (moving the tile below the empty tile up)
        curr_zero = self.zero_loc
        #print('zero is at: ', curr_zero)
        zatop = zadown = zaleft = zaright = False
        if curr_zero[0] == 0: # if at top edge
            zatop = True
        if curr_zero[0] == 2: # if at bottom edge
            zadown = True
        if curr_zero[1] == 0: # if at left edge
            zaleft = True
        if curr_zero[1] ==2: # if at right edge
            zaright = True
        #print(self.state)
        
        if not zadown:
            # make new state configuration, shifting the bottom ele of zero_loc up
            new_config = [[ele for ele in row] for row in self.state]
            new_config[self.zero_loc[0]][self.zero_loc[1]] = new_config[self.zero_loc[0]+1][self.zero_loc[1]]
            new_zero_loc = [self.zero_loc[0]+1, self.zero_loc[1]]
            new_config[self.zero_loc[0]+1][self.zero_loc[1]] = 0
            #print("og: ", self.state, ' new config: ', new_config)
            down_new_state = EightPuzzleState(new_config, self.goal, self.dist_from_start+1, self.use_heuristic, new_zero_loc)
            nbr_states.append(down_new_state)
        if not zaleft:
            # make new state configuration, shifting the top ele of zero_loc down
            new_config = [[ele for ele in row] for row in self.state]
            new_config[self.zero_loc[0]][self.zero_loc[1]] = new_config[self.zero_loc[0]][self.zero_loc[1]-1]
            new_zero_loc = [self.zero_loc[0], self.zero_loc[1]-1]
            new_config[self.zero_loc[0]][self.zero_loc[1]-1] = 0
            #print("og: ", self.state, ' new config: ', new_config)
            left_new_state = EightPuzzleState(new_config, self.goal, self.dist_from_start+1, self.use_heuristic, new_zero_loc)
            nbr_states.append(left_new_state)
        if not zatop:
            # make new state configuration, shifting the top ele of zero_loc down
            new_config = [[ele for ele in row] for row in self.state]
            new_config[self.zero_loc[0]][self.zero_loc[1]] = new_config[self.zero_loc[0]-1][self.zero_loc[1]]
            new_zero_loc = [self.zero_loc[0]-1, self.zero_loc[1]]
            new_config[self.zero_loc[0]-1][self.zero_loc[1]] = 0
            #print("og: ", self.state, ' new config: ', new_config)
            top_new_state = EightPuzzleState(new_config, self.goal, self.dist_from_start+1, self.use_heuristic, new_zero_loc)
            nbr_states.append(top_new_state)
        if not zaright:
            # make new state configuration, shifting the top ele of zero_loc down
            new_config = [[ele for ele in row] for row in self.state]
            new_config[self.zero_loc[0]][self.zero_loc[1]] = new_config[self.zero_loc[0]][self.zero_loc[1]+1]
            new_zero_loc = [self.zero_loc[0], self.zero_loc[1]+1]
            new_config[self.zero_loc[0]][self.zero_loc[1]+1] = 0
            #print("og: ", self.state, ' new config: ', new_config)
            right_new_state = EightPuzzleState(new_config, self.goal, self.dist_from_start+1, self.use_heuristic, new_zero_loc)
            nbr_states.append(right_new_state)
        return nbr_states

    # Checks if goal has been reached
    def is_goal(self):
        # In python "==" performs deep list equality checking, so this works as desired
        return self.state == self.goal
    
    # Can't hash a list, so first flatten the 2d array and then turn into tuple
    def __hash__(self):
        return hash(tuple([item for sublist in self.state for item in sublist]))
    def __eq__(self, other):
        return self.state == other.state
    
    # TODO(IV): implement this method
    def compute_heuristic(self):
        total = 0
        # NOTE: There is more than one possible heuristic, 
        #       please implement the Manhattan heuristic, as described in the MP instructions
        req_dic = {1:[0,1], 2:[0,2], 3:[1,0], 4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8:[2,2]}
        for num in range(1,9):
            c_pos = self.get_pos(num)
            total+=manhattan(req_dic[num],c_pos)
                        
        return total
    def get_pos(self, num):
        #print('to find:', num, ' states: ', self.state)
        for i in range(0,3):
            for j in range(0,3):
                if self.state[i][j] == num:
                    return [i,j]
        #print('SOMETHING IS FUCKING WRONG!')
        return [-1,-1]
    # TODO(IV): implement this method
    # Hint: it should be identical to what you wrote in WordLadder.__lt__(self, other)
    def __lt__(self, other):
        self_f = self.dist_from_start + self.h
        other_f = other.dist_from_start + other.h
        if self_f == other_f:
            return super().__lt__(other)
        else:
            return self_f < other_f
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return self.state
    def __repr__(self):
        return "\n---\n"+"\n".join([" ".join([str(r) for r in c]) for c in self.state])

# Grid ------------------------------------------------------------------------------------------------

class SingleGoalGridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of a single length 2 tuple location in the grid that needs to be reached, i.e., ((x,y),)
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        '''
        self.maze_neighbors = maze_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # TODO(V): implement this method
    def get_neighbors(self):
        nbr_states = []
        # We provide you with a method for getting a list of neighbors of a state,
        # you need to instantiate them as GridState objects
        neighboring_grid_locs = self.maze_neighbors(*self.state)
        for idx in neighboring_grid_locs:
            # the best_first_search function takes care of the repeated location visits
            new_state = SingleGoalGridState(idx, self.goal, self.dist_from_start+1, self.use_heuristic, self.maze_neighbors)
            nbr_states.append(new_state)
        return nbr_states

    # TODO(V): implement this method, check if the current state is the goal state
    def is_goal(self):
        return self.state == self.goal[0]
    
    def __hash__(self):
        return hash(self.state) + hash(self.goal)
    
    def __eq__(self, other):
        return self.state == other.state
    
    # TODO(V): implement this method
    # Compute the manhattan distance between self.state and self.goal 
    def compute_heuristic(self):
        return manhattan(self.state, self.goal[0])
    
    # TODO(V): implement this method... should be unchanged from before
    def __lt__(self, other):
        self_f = self.dist_from_start + self.h
        other_f = other.dist_from_start + other.h
        if self_f == other_f:
            return super().__lt__(other)
        else:
            return self_f < other_f

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goal=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goal=" + str(self.goal)

# def potNewMST(curr):
#         smallestMan = 1000000000000
#         nearest_goal = curr.goal[0]
#         for g in curr.goal:
#             curr_man = manhattan(curr.state, g)
#             if curr_man < smallestMan:
#                 smallestMan = curr_man
#                 nearest_goal = g
#         newgoalset = deepcopy(curr.goal)
#         newgoalset = tuple(g for g in newgoalset if g != nearest_goal)
#         print(newgoalset)
#         return compute_mst_cost(newgoalset, manhattan)
    
class GridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, mst_cache=None):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of length 2 tuples location in the grid that needs to be reached
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
        '''
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # TODO(VI): implement this method
    def get_neighbors(self):
        nbr_states = []
        # We provide you with a method for getting a list of neighbors of a state,
        # You need to instantiate them as GridState objects
        neighboring_locs = self.maze_neighbors(*self.state)
        for idx in neighboring_locs:
            goal_copy = deepcopy(self.goal)
            goal_copy = tuple(g for g in self.goal if g != idx)
            new_state = GridState(idx, goal_copy, self.dist_from_start+1, self.use_heuristic, self.maze_neighbors, self.mst_cache)
            
            nbr_states.append(new_state)
        return nbr_states

    # TODO(VI): implement this method
    def is_goal(self):
        return len(self.goal) == 0
        # for g in self.goal:
        #     if self.state == g:
        #         self.goal = tuple(g for g in self.goal if g != self.state)
        #         return True
        # return False
    
    # TODO(VI): implement these methods __hash__ AND __eq__
    def __hash__(self):
        return hash(self.state + self.goal)
    
    def __eq__(self, other):
        return self.state == other.state and self.goal == other.goal
    
    # TODO(VI): implement this method
    # Our heuristic is: manhattan(self.state, nearest_goal) + MST(self.goal)
    # If we've computed MST(self.goal) before we can simply query the cache, otherwise compute it and cache value
    # NOTE: if self.goal has only one goal then the MST value is simply zero, 
    #       and so the heuristic reduces to manhattan(self.state, self.goal[0])
    # You should use compute_mst_cost(self.goal, manhattan) which we imported from utils.py
    def compute_heuristic(self):
        if not self.goal: return 0
        # if self.mst_cache:
        #     curr_mst = self.mst_cache
        # else:
        #     curr_mst = compute_mst_cost(self.goal, manhattan)
        #     self.mst_cache = curr_mst
            #print(curr_mst)
        curr_mst = compute_mst_cost(self.goal, manhattan) # always compute new mst
        nearest_dist = manhattan(self.state, self.goal[0])
        for i in range(1, len(self.goal)):
            g = self.goal[i]
            curr_man = manhattan(self.state, g)
            if curr_man < nearest_dist:
                nearest_dist = curr_man
        #print('nearest dist: ', nearest_dist, ' mst: ', curr_mst)
        return nearest_dist + curr_mst
        
    # TODO(VI): implement this method... should be unchanged from before
    def __lt__(self, other):
        self_f = self.dist_from_start + self.h
        other_f = other.dist_from_start + other.h
        if self_f == other_f:
            return super().__lt__(other)
            # self_newMST = potNewMST(self)
            # other_newMST = potNewMST(other)
            # print(self_newMST, other_newMST)
            # return self_newMST < other_newMST
        else:
            return self_f < other_f
        
        
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)