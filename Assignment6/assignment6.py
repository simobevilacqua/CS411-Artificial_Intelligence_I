import json
import argparse
import random
from collections import OrderedDict

class MDP:
    def __init__(self, T, R, gamma, epsilon, terminals_pos, terminals_neg, rows, cols):
        self.T = T                                      # Transition model T(s'| s, a)
        self.R = R                                      # Rewards function R(s, a, s')
        self.gamma = gamma                              # Discount factor
        self.epsilon = epsilon
        self.actions = ['up', 'down', 'left', 'right']  # Set of possible actions

        # Set of possible states (does not include obstacles)
        unique_states = OrderedDict()
        for key in T.keys():
            unique_states[key[0]] = None
        self.states = list(unique_states.keys())

        # Terminal states (with positive and negative reward)
        self.terminals_pos = [tuple(t) for t in terminals_pos]
        self.terminals_neg = [tuple(t) for t in terminals_neg]

        # Set of non terminal states
        self.non_terminal_states = [s for s in self.states if s not in self.terminals_neg and s not in self.terminals_pos]

        # We want the utilities to be correctly computed, so we need to propagate the utility values to all
        # other cells considering the largest possible distance between starting point and terminal states
        self.modified_policy_evaluation_limit = rows + cols

    # Bellman equation
    def q_value(self, s, a, U):
        q_value = 0
        for s_prime in self.states:
            q_value += self.T.get((s, a, s_prime), 0) * (self.R.get((s, a, s_prime), 0) + self.gamma * U[s_prime])
        
        return q_value

    # Selection of the best policy given the current utilities for each state
    def policy_selection(self, U):
        policy = {}
        for s in self.non_terminal_states:
            best_action = None
            best_value = float('-inf')

            for a in self.actions:
                value = self.q_value(s, a, U)
                if value > best_value:
                    best_value = value
                    best_action = a

            policy[s] = best_action
        
        return policy

    # Value iteration algorithm
    def value_iteration(self):
        print("=== START ===")
        iterations = 0
        U_prime = {s: 0 for s in self.states}
        epsilon_term = self.epsilon * (1 - self.gamma) / self.gamma

        while True:
            iterations += 1
            U = U_prime.copy()
            delta = 0

            for s in self.non_terminal_states:
                U_prime[s] = max(self.q_value(s, a, U) for a in self.actions)
                if abs(U_prime[s] - U[s]) > delta:
                    delta = abs(U_prime[s] - U[s])

            print(f"Iteration: {iterations}")
            for u in U_prime.keys():
                print(f"{u}: {U[u]:.10f}")

            if delta <= epsilon_term:
                break
        
        policy = self.policy_selection(U_prime)
        print("=== END ===")

        return U_prime, policy

    # Deriving the utilities for all states given a policy
    def policy_evaluation(self, policy, U):
        U_prime = {s: 0 for s in self.states}

        for _ in range(self.modified_policy_evaluation_limit):
            for s in self.non_terminal_states:
                U_prime[s] = self.q_value(s, policy[s], U)
            U = U_prime.copy()
            
        return U_prime

    # Get the best action given a state and the utilites for each state
    def get_best_action(self, s, U):
        best_action = None
        best_value = float('-inf')
        for a in self.actions:
            value = self.q_value(s, a, U)
            if value > best_value:
                best_value = value
                best_action = a
        
        return best_action

    # Policy iteration algorithm
    def policy_iteration(self):
        print("=== START ===")
        iterations = 0
        U = {s: 0 for s in self.states}
        policy = {s: random.choice(self.actions) for s in self.non_terminal_states}

        unchanged = False
        while not unchanged:
            iterations += 1
            U = self.policy_evaluation(policy, U)
            unchanged = True

            for s in self.non_terminal_states:
                a_star = self.get_best_action(s, U)
                if self.q_value(s, a_star, U) > self.q_value(s, policy[s], U):
                    policy[s] = a_star
                    unchanged = False
            
            print(f"Iteration: {iterations}")
            for u in U.keys():
                print(f"{u}: {U[u]:.10f}")
        
        print("=== END ===")
        
        return U, policy

def main():
    # Getting the input file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs=1)
    args = parser.parse_args()
    inputFile = args.input[0]

    with open(inputFile, 'r') as file:
        # Loading the information from the JSON file
        data = json.load(file)
        T = {eval(k): v for k, v in data['T'].items()}
        R = {eval(k): v for k, v in data['R'].items()}
        gamma = data['gamma']
        epsilon = data['epsilon']
        terminals_pos = data['terminals_pos']
        terminals_neg = data['terminals_neg']
        rows = data["rows"]
        cols = data["cols"]
        mdp = MDP(T, R, gamma, epsilon, terminals_pos, terminals_neg, rows, cols)
    
    # Value iteration
    print("Value iteration")
    U, policy = mdp.value_iteration()
    print("Value iteration results:")
    for u in U.keys():
        if u in policy.keys():
            print(f"{u}: {U[u]:.4f} -> {policy[u]}")
        else:
            print(f"{u}: {U[u]:.4f}")
    print("\n")

    # Policy iteration
    print("Policy iteration")
    U, policy = mdp.policy_iteration()
    print("Policy iteration results:")
    for u in U.keys():
        if u in policy.keys():
            print(f"{u}: {U[u]:.4f} -> {policy[u]}")
        else:
            print(f"{u}: {U[u]:.4f}")

if __name__ == "__main__":
    main()
