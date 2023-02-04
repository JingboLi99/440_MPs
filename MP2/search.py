import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search 
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state)
    #   - keep track of the distance of each state from start
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    
    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    
    #Check if it is multipath:
    #
    #for all other questions
    #if not isMultiPath:
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    #we add to visited_states when we explore the neighbours, not when we actually pop the state in frontier
    while len(frontier) > 0:
        curr_state = heapq.heappop(frontier)
        
        if curr_state.is_goal():
            return backtrack(visited_states, curr_state)
                
        neighbors = curr_state.get_neighbors()
        for state in neighbors:
            if state not in visited_states or state.dist_from_start < visited_states[state][1]:
                heapq.heappush(frontier, state)
                visited_states[state] = (curr_state, state.dist_from_start)    
    # if you do not find the goal return an empty list
    return []
    # ------------------------------
    # # for multipath question
    # else:
    #     totalPath = []
    #     #we add to visited_states when we explore the neighbours, not when we actually pop the state in frontier
    #     while len(starting_state.goal) > 0:
    #         visited_states = {starting_state: (None, 0)}
    #         frontier = []
    #         heapq.heappush(frontier, starting_state)
            
    #         while len(frontier) > 0:
    #             #print("step")
    #             curr_state = heapq.heappop(frontier)
                
    #             if curr_state.is_goal():
    #                 #do some kind of extension
    #                 totalPath.extend(backtrack(visited_states, curr_state)[:-1]) #extend start to end (goal), without the end
    #                 starting_state = curr_state
    #                 break
                    
    #             neighbors = curr_state.get_neighbors()
    #             for state in neighbors:
                    
    #                 if state not in visited_states or state.dist_from_start + state.h < visited_states[state][1]:
    #                     #print("curr evaluated neighbour state: ", state.state)
    #                     heapq.heappush(frontier, state)
    #                     visited_states[state] = (curr_state, state.dist_from_start + state.h)    
    #     totalPath.append(starting_state)
    #     for i in totalPath:
    #         print(i.state, end="->")
    #     return totalPath

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    curr = goal_state
    while curr:
        path.append(curr)
        curr = visited_states[curr][0]
    return path[::-1]
    # ------------------------------