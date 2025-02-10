import argparse
from queue import PriorityQueue
import time
import os
import psutil

# Goal state
goal = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

# Action set
actions = {
    'U': -4,
    'D': 4,
    'L': -1,
    'R': 1
}

# Node object
class Node:
    def __init__(self, state, move="", depth=0, cost=0, parent=None):
        self.state = state      # Current state
        self.move = move        # Move taken to move from parent to current state
        self.depth = depth      # Node depth
        self.cost = cost        # Current cost considering the heuristic
        self.parent = parent    # Parent node for traceback
    
    def __lt__(self, other):    # Used to let priority queue sort nodes by cost  
        return self.cost < other.cost


# Count the number of misplaced tiles in the 'state' configuration compared to the goal state
def MisplacedHeuristic(state):
    result = 0

    for i in range(len(state)):
        if state[i] != goal[i]:
            result += 1
    
    return result


# Sum the Manhattan distance of each tile in the 'state' configuration compared to the goal state
def ManhattanHeuristic(state):
    result = 0

    for i in range(1, len(state)):
        # Position of tile i in the goal
        correct_x = (i - 1) % 4
        correct_y = (i - 1) // 4

        # Position of tile i in the 'state' configuration
        curr_x = state.index(i) % 4
        curr_y = state.index(i) // 4

        # Manhattan distance
        result += abs(curr_x - correct_x) + abs(curr_y - correct_y)
    
    return result


# A* algorithm
def a_star(initial_state, heuristic_type=""):
    # Setting up variables
    frontier = PriorityQueue()  # Priority queue to search all the states
    expanded_nodes = 0
    reached = {}                # Lookup table to store the visited states and their associated best cost

    # Choose the heuristic function
    if heuristic_type == "Manhattan":
        heuristic = ManhattanHeuristic
    else: # Default heuristic
        heuristic = MisplacedHeuristic
    
    # Frontier initialization
    h = heuristic(initial_state)
    initial_node = Node(initial_state, cost=h)
    frontier.put((h, initial_node))
    reached[initial_state] = initial_node

    while not frontier.empty(): # Process all nodes in the frontier
        node = frontier.get()[1] # Extract the node
        expanded_nodes += 1

        if node.state == goal:  # Check if the goal is reached
            return node, expanded_nodes

        for action, distance in actions.items():    # Analyze all possible moves (EXPAND() function)
            # Checking the valid moves
            new_index = node.state.index(0) + distance
            if action == 'L' and node.state.index(0) % 4 == 0:
                continue
            if action == 'R' and (node.state.index(0) + 1) % 4 == 0:
                continue
            if 0 <= new_index < 16: # Check U and D actions validity
                state_list = list(node.state)
                state_list[node.state.index(0)], state_list[new_index] = state_list[new_index], state_list[node.state.index(0)]
                new_state = tuple(state_list)
                new_cost = node.depth + 1 + heuristic(new_state)

                if new_state not in reached or new_cost < reached[new_state].cost:   # Expand node only if new or already visited but with a higher cost
                    new_node = Node(new_state, action, node.depth + 1, new_cost, node)
                    frontier.put((new_cost, new_node))
                    reached[new_state] = new_node

    return "No solution", expanded_nodes


# Traceback to get the set of moves
def traceback(starting_node):
    moves = ""
    node = starting_node

    if node != None:
        while node.parent != None:
            moves = node.move + moves
            node = node.parent

    return moves


def main():
    # Setting up variables for memory and time measurements
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Getting the initial state from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('board', type=int, nargs=16)
    parser.add_argument('--h', type=str, default="Misplaced", nargs=1) # Optional argument to select the type of heuristic
    args = parser.parse_args()
    initial_state = tuple(args.board)   # Represent the board as tuple so that it's hashable and it can be added to the hashset
    heuristic = args.h[0]

    # Checking if the initial state is valid
    if sorted(initial_state) != list(range(16)):
        print("Invalid board")
    else:
        # Proceeding with the search
        node, nodes_visited = a_star(initial_state, heuristic_type=heuristic)
        moves = traceback(node)

        # Computing time and memory information
        time_taken = time.time() - start_time
        memory_used = process.memory_info().rss - initial_memory

        # Printing the results
        print(f"Moves: {moves}")
        print(f"Number of Nodes expanded: {nodes_visited}")
        print(f"Time Taken: {time_taken:.4f} seconds")

        # Printing memory usage in the proper unit measure
        if memory_used < 1024:
            print(f"Memory Used: {memory_used:.3f} B")
        elif memory_used < 1024 ** 2:
            memory_used = memory_used / 1024
            print(f"Memory Used: {memory_used:.3f} KB")
        else:
            memory_used = memory_used / (1024 ** 2)
            print(f"Memory Used: {memory_used:.3f} MB")

if __name__ == "__main__":
    main()
