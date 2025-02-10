import json

# Grid description
rows = 3
cols = 4
obstacles = [(2, 2)]
terminals_pos = [(4, 3)]
terminals_neg = [(4, 2)]
prob_move = 0.8
prob_left = 0.1
prob_right = 0.1
prob_back = 0
reward_pos = 1
reward_neg = -1
reward_def = -0.04
gamma = 1
epsilon = 0.001

# Non-deterministic behavior for each move
# For each move it's possible to specify a deviated move to the left, right, back with respect to the current move
moves = {
    'up': ('left', 'right', 'down'),
    'down': ('right', 'left', 'up'),
    'left': ('down', 'up', 'right'),
    'right': ('up', 'down', 'left')
}

# Get the correct next state considering the grid boundaries and the obstacles
def next_state(x, y, move):
    if (x, y) in terminals_neg or (x, y) in terminals_pos:
        return (x, y)
    if move == 'up':
        return (x, y + 1) if y + 1 <= rows and (x, y + 1) not in obstacles else (x, y)
    elif move == 'down':
        return (x, y - 1) if y - 1 > 0 and (x, y - 1) not in obstacles else (x, y)
    elif move == 'left':
        return (x - 1, y) if x - 1 > 0 and (x - 1, y) not in obstacles else (x, y)
    elif move == 'right':
        return (x + 1, y) if x + 1 <= cols and (x + 1, y) not in obstacles else (x, y)

def main():
    # Transitions and Rewards
    T = {}
    R = {}
    for x in range(1, cols + 1):
        for y in range(1, rows + 1):
            if (x, y) in obstacles:
                continue

            for move, (left, right, back) in moves.items():
                next_move = next_state(x, y, move)      # Correct move
                next_left = next_state(x, y, left)      # Move slipped to the left
                next_right = next_state(x, y, right)    # Move slipped to the right
                next_back = next_state(x, y, back)      # Move slipped to the back

                T[f"(({x}, {y}), '{move}', {next_move})"] = prob_move
                if prob_left > 0:
                    if next_left == next_move:
                        T[f"(({x}, {y}), '{move}', {next_left})"] += prob_left
                    else:
                        T[f"(({x}, {y}), '{move}', {next_left})"] = prob_left

                if prob_right > 0:
                    if next_right == next_left or next_right == next_move:
                        T[f"(({x}, {y}), '{move}', {next_right})"] += prob_right
                    else:
                        T[f"(({x}, {y}), '{move}', {next_right})"] = prob_right

                if prob_back > 0:
                    if next_back == next_move or next_back == next_left or next_back == next_right:
                        T[f"(({x}, {y}), '{move}', {next_back})"] += prob_back
                    else:
                        T[f"(({x}, {y}), '{move}', {next_back})"] = prob_back

                if (x, y) not in terminals_neg and (x, y) not in terminals_pos:
                    R[f"(({x}, {y}), '{move}', {next_move})"] = reward_pos if next_move in terminals_pos else (reward_neg if next_move in terminals_neg else reward_def)
                    if prob_left > 0:
                        R[f"(({x}, {y}), '{move}', {next_left})"] = reward_pos if next_left in terminals_pos else (reward_neg if next_left in terminals_neg else reward_def)
                    if prob_right > 0:
                        R[f"(({x}, {y}), '{move}', {next_right})"] = reward_pos if next_right in terminals_pos else (reward_neg if next_right in terminals_neg else reward_def)
                    if prob_back > 0:
                        R[f"(({x}, {y}), '{move}', {next_back})"] = reward_pos if next_back in terminals_pos else (reward_neg if next_back in terminals_neg else reward_def)

    # Create JSON file
    json_data = {
        "T": T,
        "R": R,
        "gamma": gamma,
        "epsilon": epsilon,
        "terminals_pos": terminals_pos,
        "terminals_neg": terminals_neg,
        "rows": rows,
        "cols": cols
    }

    # Write JSON file
    file_path = 'input.json'
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    main()
