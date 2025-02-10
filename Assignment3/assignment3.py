import argparse
from collections import deque
import time
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
    def __init__(self, state, move="", depth=0, parent=None):
        self.state = state      # Current state
        self.move = move        # Move taken to move from parent to current state
        self.depth = depth      # Node depth
        self.cost = depth       # Current cost (in this case same as depth)
        self.parent = parent    # Parent node for traceback


def bfs(initial_state):
    # Setting up variables
    process = psutil.Process()
    start_time = time.time()
    reached = set()                     # Hashset for fast lookup to implement graph search
    frontier = deque()                  # FIFO queue to breadth-first seach all the states
    
    initial_node = Node(initial_state)
    frontier.append(initial_node)       # Frontier initialization
    reached.add(initial_state)
    expanded_nodes = 0

    while frontier:                     # Process all nodes in the frontier
        node = frontier.popleft()
        expanded_nodes += 1

        if node.state == goal:          # Check if we reached the goal
            return node, expanded_nodes, time.time() - start_time, process.memory_info().rss

        for action, distance in actions.items():    # Analyze all possible moves (EXPAND() function)
            # Checking the valid moves
            new_index = node.state.index(0) + distance
            if action == 'L' and node.state.index(0) % 4 == 0:
                continue
            if action == 'R' and (node.state.index(0) + 1) % 4 == 0:
                continue
            if 0 <= new_index < 16:     # Check U and D actions validity
                state_list = list(node.state)
                state_list[node.state.index(0)], state_list[new_index] = state_list[new_index], state_list[node.state.index(0)]
                new_state = tuple(state_list)
                
                if new_state not in reached:        # Checking if the child node has already been visited
                    reached.add(new_state)
                    new_node = Node(new_state, action, node.depth + 1, node)
                    frontier.append(new_node)

    return None, expanded_nodes, time.time() - start_time, process.memory_info().rss


def main():
    # Getting the initial state from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('board', type=int, nargs=16)
    args = parser.parse_args()
    initial_state = tuple(args.board)   # Represent the board as tuple so that it's hashable and it can be added to the hashset

    # Checking if the initial state is valid
    if sorted(initial_state) != list(range(16)):
        print("Invalid board")
    else:
        # Proceeding with the search
        node, nodes_visited, time_taken, memory_used = bfs(initial_state)

        # Traceback to get the set of moves
        moves = ""
        if node != None:
            while node.parent != None:
                moves = node.move + moves
                node = node.parent

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
