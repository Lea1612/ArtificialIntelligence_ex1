from functools import total_ordering
import heapq
import math

# The priorityQueue class countains our priorities to pop from the frontier. Who will go out first ?
# Who will get into the closed list, and all the functions we need to build the frontier and the closed list.
class PriorityQueue:
    # Initialization of the PriorityQueue class. The items are a heap and f
    def __init__(self, f=lambda x: x):
        self.heap = []
        self.f = f

    # Push the value item onto the heap, maintaining the heap invariant.
    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    # Adding the items
    def extend(self, items):
        for item in items:
            self.append(item)

    # Pop and return the smallest item from the heap, maintaining the heap invariant. 
    # Raise exception if the heap is empty
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    # Return the length of the heap      
    def __len__(self):
        return len(self.heap)

    # Return the elements of the heap
    def __contains__(self, key):
        return any([item == key for _, item in self.heap])

    # Return the value of a key of the heap
    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    # Delete an element from the heap
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    # System representation
    def __repr__(self):
      return str(self.heap)


# The GridProblem class countains our problem. The starting point, the goal point, the grid and the cost.
# We also tell him the actions it can do, and the forbidden ones.
class GridProblem:
    # Initialization of the grid problem. Its items are the starting point, the goal point and the grid
    def __init__(self, s_start, goal, G):
        self.s_start = s_start
        self.goal = goal
        self.G = G  # grid

    # The different action the algorithm can do and which way it can go. We also need to check the cliffs
    def actions(self, s_string):
        x, y = xy_string_to_axis(s_string)
        actions_to_do = []

        # First action = right
        # Check if good place and if it's not leading to a cliff (for every action)
        if self.xy_place_verification(x, y+1) and self.G[x][y+1] != -1:
            actions_to_do.append('R')

        # Diagonal RD and diagonal is forbidden when there is a cliff above or below the next move
        if self.xy_place_verification(x+1, y+1) and self.G[x+1][y+1] != -1 and self.G[x][y + 1] != -1 and self.G[x + 1][y] != -1:
            actions_to_do.append('RD')

        # Down
        if self.xy_place_verification(x+1, y) and self.G[x+1][y] != -1:
            actions_to_do.append('D')

        # Diagonal LD
        if self.xy_place_verification(x+1, y-1) and self.G[x+1][y-1] != -1 and self.G[x + 1][y] != -1 and self.G[x][y - 1] != -1:
            actions_to_do.append('LD')

        # Left
        if self.xy_place_verification(x, y-1) and self.G[x][y-1] != -1:
            actions_to_do.append('L')

        # Diagonal LU
        if self.xy_place_verification(x-1, y-1) and self.G[x-1][y-1] != -1 and self.G[x][y - 1] != -1 and self.G[x - 1][y] != -1:
            actions_to_do.append('LU')

        # Up
        if self.xy_place_verification(x-1, y) and self.G[x-1][y] != -1:
            actions_to_do.append('U')

        # Diagonal RU
        if self.xy_place_verification(x-1, y+1) and self.G[x-1][y+1] != -1 and self.G[x - 1][y] != -1 and self.G[x][y + 1] != -1:
            actions_to_do.append('RU')

        return actions_to_do

    # Check if x and y are in a proper place
    def xy_place_verification(self, x, y):
        if x >= 0 and x < len(self.G) and y >= 0 and y < len(self.G[0]):
            return True
        else:
            return False

    # Returns the action letters
    def xy_from_action(self, s, a):
        x, y = xy_string_to_axis(s)
        if a == 'R':
            return (x, y + 1)
        elif a == 'RD':
            return (x + 1, y + 1)
        elif a == 'D':
            return (x + 1, y)
        elif a == 'LD':
            return (x + 1, y - 1)
        elif a == 'L':
            return (x, y - 1)
        elif a == 'LU':
            return (x - 1, y - 1)
        elif a == 'U':
            return (x - 1, y)
        elif a == 'RU':
            return (x - 1, y + 1)

    # Succ returns a new state from an action. 
    def succ(self, s, a):
        x, y = self.xy_from_action(s, a)
        return xy_axis_to_string(x, y)

    # Check if we got to the goal
    def is_goal(self, s):
        return s == self.goal

    # Check the cost of a step by a given action
    def step_cost(self, s, a):
        x, y = self.xy_from_action(s, a)
        return self.G[x][y]

    # Return the state so it's readable
    def state_str(self, s):
        return s

    # System representation
    def __repr__(self):
        return {'s_start': self.s_start, 'goal': self.goal, 'graph': self.G}


# In the node class we represent a current state from the algorithm and the problem.
# We represent the current node with items like its state, its parent, the action to do ...
@total_ordering
class Node(object):
    # Initialization for the current state
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.identifier_iteration = 0
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    # The possible next steps the current state can do 
    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    # Returns the next node (object) from an action according to the problem
    def child_node(self, problem, action):
        next_state = problem.succ(self.state, action)
        return Node(next_state, self, action, self.path_cost+problem.step_cost(self.state, action))

    # Returns the solution (what actions it did) 
    def solution(self):
      return [node.action for node in self.path()[1:]]

    # Iteration from the current node to the start node and returns a list of nodes from the path.
    def path(self):
      node, path_back = self, []

      while node:

          path_back.append(node)
          node = node.parent
      return list(reversed(path_back))

    # The actions priority function returns number from an action. 
    # It sorts our priority (R-RD-D-LD-L-LU-RU) to have the clock order
    def actions_priority(self):
        if self.action == 'R':
            return 1
        elif self.action == 'RD':
            return 2
        elif self.action == 'D':
            return 3
        elif self.action == 'LD':
            return 4
        elif self.action == 'L':
            return 5
        elif self.action == 'LU':
            return 6
        elif self.action == 'U':
            return 7
        elif self.action == 'RU':
            return 8

    # System representation
    def __repr__(self):
      return f"<{self.state}>"

    # We'll use this function to know which way to go on UCS and IDA*. We want first minimum path cost,
    # then, we check the node that has been discovered first, and then the priority direction (R-RD...)
    def __lt__(self, node):
         return self.path_cost <= node.path_cost and self.identifier_iteration <= node.identifier_iteration and self.actions_priority() < node.actions_priority()

    # Equals to (between two states)
    def __eq__(self, other):
      return isinstance(other, Node) and self.state == other.state

    # Not equal to 
    def __ne__(self, other):
      return not (self == other)

    # Hash implementation
    def __hash__(self):
        return hash(self.state)

# The NodesCounter class is a helper class. It's just an easy way for me to count the number 
# of developed nodes in every algorithms 
# initialization of the new_limit variable in DFS_f algorithm
# initialization of the finish_counter variable that we use to calculate the depth in IDASTAR. We want it to be < 21.
# I want those variables to be reference type because I use them on several functions.
# And we add to it 1 when it's necessary in the algorithms' functions.
class NodesCounter:
    def __init__(self):
        self.counter = 0
        self.new_limit = 0
        self.finish_counter = 0

# This function reads the input file and represents it on a list of strings 
def get_input(input):

    input_list = []

    with open(input, 'r') as f:

        for line in f:
            input_list.append(line.rstrip('\n'))

    return input_list

# The last elements of the input list represent the table (grid).
# We separate it, create it into a matrix in the grid_creation function.
def grid_creation(input_list):

    grid = []

    for element in input_list[4:]:
        element_split = element.split(',')
        vector_element = []

        for char in element_split:
           vector_element.append(int(char))

        grid.append(vector_element)

    return grid

# (x, y) : the location of the state is given from the input as a string 'x,y'. 
# This function returns x and y as numbers (coordinate on 'axis' of the grid)
def xy_string_to_axis(xy_string):
    s = xy_string.split(',')
    x = int(s[0])
    y = int(s[1])
    return x, y

# This function does the opposit. It takes as input x and y as numbers and put it back to a string 'x,y'
def xy_axis_to_string(x, y):
    xy_st = str(x) + ',' + str(y)
    return xy_st

# The function g returns the path cost of a node (The cost from the starting node to the current node). 
# We use it in UCS, A* and IDA*
def g(node):
    return node.path_cost

# Heuristic function: Chebychev distance. It's an optimal heuristic. We use it in A* and IDA*.
def heuristic(node, problem):
    goal_point = problem.goal
    current_point = node.state
    x_cur, y_cur = xy_string_to_axis(current_point)
    x_goal, y_goal = xy_string_to_axis(goal_point)
    return max(abs(x_cur - x_goal), abs(y_cur - y_goal))


###### IDS ######

# DLS algorithm with a limit, that we need to implement the IDS
def depth_limited_search(problem, limit, nodes_counter):
    frontier = [(Node(problem.s_start))]  # Stack

    while frontier:
        node = frontier.pop()
        
        if problem.is_goal(node.state):
            return node.solution(), node.path_cost, nodes_counter.counter

        nodes_counter.counter += 1
        if node.depth < limit:
            expanded_list = node.expand(problem)
            frontier.extend(expanded_list[::-1])

    return None, node.path_cost, nodes_counter.counter


# IDS calculates the maximum depth (We fix it to 20), uses DLS algo on depth of the iteration
# In every loop, we increase the depth.
def iterative_deepening_search(problem):
    max_depth = 21
    nodes_counter = NodesCounter()

    for depth in range(0, max_depth):
        result, path_cost, nodes_count = depth_limited_search(problem, depth, nodes_counter)

        if result:
            return result, path_cost, nodes_count

    return None, None, None


##### UCS #####

# BFGS algorithm implementation. We need it to implement UCS
def best_first_graph_search(nodes_counter, problem, f):
    node = Node(problem.s_start)
    frontier = PriorityQueue(f)
    frontier.append(node)
    closed_list = set()
    iteration_update = 0

    while frontier:

        node = frontier.pop()

        if problem.is_goal(node.state):
            return node.solution(), node.path_cost, nodes_counter.counter

        closed_list.add(node.state)
        nodes_counter.counter += 1

        for child in node.expand(problem):
            iteration_update += 1
            child.identifier_iteration = iteration_update 

            if child.state not in closed_list and child not in frontier:
                frontier.append(child)

            elif child in frontier and f(child) < frontier[child]:
                del frontier[child]
                frontier.append(child)

    return None, None, None

# UCS algorithm with f = g:
def uniform_cost_search(problem):
    nodes_counter = NodesCounter()
    return best_first_graph_search(nodes_counter, problem, f=g)


#### A* ####

# Implementation of the A* algorithm. It uses the BFGS algorithm on f and g
def A_star(problem):
    nodes_counter = NodesCounter()
    return best_first_graph_search(nodes_counter, problem, f=lambda n: g(n)+heuristic(n, problem))


#### IDA* ####

# Implementation of DFS algorithm with limit. To implement IDA*, we need this function.
def DFS_f(node, g, cost_limit, problem, nodes_counter):
    new_cost = g + heuristic(node, problem)
    # nodes_counter.counter += 1
    if new_cost > cost_limit:
        nodes_counter.new_limit = min(nodes_counter.new_limit, new_cost)
        return None, None, None
    
    if problem.is_goal(node.state):
        return node.solution(), node.path_cost, nodes_counter.counter
    
    nodes_counter.counter += 1

    for child in node.expand(problem):
        x_child, y_child = xy_string_to_axis(child.state)
        child_cost = problem.G[x_child][y_child]

        nodes_counter.finish_counter += 1

        if nodes_counter.finish_counter > 20:
            return None, None, None

        solution, path_cost, nodes_count = DFS_f(child, g + child_cost, cost_limit, problem, nodes_counter)
        nodes_counter.finish_counter -= 1
        
        if solution:
            return solution, path_cost, nodes_count

    return None, None, None


# Implementation of IDA* algorithm, which calculates the maximum depth (we fix it to 20) 
# In each iteration, it increases the depth and on every depth, uses algo DFS-f
def IDA_star(problem): 
    start_node = Node(problem.s_start)
    count_nodes = NodesCounter()
    count_nodes.new_limit = heuristic(start_node, problem)

    while count_nodes.finish_counter < 21 :
        cost_limit = count_nodes.new_limit
        count_nodes.new_limit = math.inf

        solution, path_cost, nodes_counter = DFS_f(start_node, 0, cost_limit, problem, count_nodes)

        if solution:
            return solution, path_cost, nodes_counter

    return None, None, None

# Return the solution from the chosen algorithm (from the input file and according to the grid problem)
def run_algorithm(input_list, problem):
    solution, path_cost, nodes_counter = None, None, None

    if input_list[0] == 'IDS':
        solution, path_cost, nodes_counter = iterative_deepening_search(problem)

    elif input_list[0] == 'UCS':
        solution, path_cost, nodes_counter = uniform_cost_search(problem)

    elif input_list[0] == 'ASTAR':
        solution, path_cost, nodes_counter = A_star(problem)

    elif input_list[0] == 'IDASTAR':
        solution, path_cost, nodes_counter = IDA_star(problem)

    return solution, path_cost, nodes_counter


# The get_output function write the output on a text file. It prints the path, the path cost
# and the number of developped nodes. And it prints 'no path' if it takes too much time 
# or if there is no way to get to the goal.
def get_output(solution, path_cost, nodes_counter):
    output_string = ""

    if solution == None:
        output_string += 'no path'

    else:
        solution = '-'.join(solution)
        output_string = f'{solution} {path_cost} {nodes_counter}'
    print(output_string)

    with open('output.txt', 'w') as f:
        f.write(output_string)


# The main function countains everything we need to compute.
# It reads the input file, creates the grid, fix the starting point and the goal point
# Gets the problem, choose the adequat algorithm and prints the solution on the output file.
def main():
    input_list = get_input('./input.txt')

    grid = grid_creation(input_list)

    start_point = input_list[1]
    goal_point = input_list[2]

    grid_problem = GridProblem(start_point, goal_point, grid)
    solution, path_cost, nodes_counter = run_algorithm(input_list, grid_problem)
    get_output(solution, path_cost, nodes_counter)


if __name__ == "__main__":
    main()