import pandas as pd
import numpy as np
import random
from scipy.stats import chi2_contingency

class DecisionTree:
    def __init__(self, attribute=None, branches=None, label=None):
        self.attribute = attribute                      # Attribute associated to the current node
        self.branches = branches if branches else {}    # Branches for each attribute value
        self.label = label                              # Label for the leaf nodes

# Decision tree learning algorithm
def learn_decision_tree(examples, attributes, default=None):
    if examples.empty:
        return DecisionTree(label=plurality_value(default))
    elif len(set(examples['WillWait'])) == 1:
        return DecisionTree(label=examples['WillWait'].iloc[0])
    elif not attributes:
        return DecisionTree(label=plurality_value(examples))
    else:
        best_attr = choose_best_attribute(attributes, examples)
        tree = DecisionTree(attribute=best_attr)
        remaining_attributes = [a for a in attributes if a != best_attr]
        
        for value in np.unique(examples[best_attr]):
            exs = examples[examples[best_attr] == value]
            subtree = learn_decision_tree(exs, remaining_attributes, examples)
            tree.branches[value] = subtree
        return tree

# Selecting the most common output value
def plurality_value(examples):
    class_counts = examples['WillWait'].value_counts()
    most_common_class = None
    max_count = 0

    for class_label, count in class_counts.items():
        if count > max_count:
            most_common_class = class_label
            max_count = count
        elif count == max_count:
            # Break ties randomly
            most_common_class = random.choice([most_common_class, class_label])
    
    return most_common_class

# Calculating entropy of the WillWait attribute
def entropy(examples):
    total_count = len(examples)
    class_counts = examples['WillWait'].value_counts()
    class_probabilities = class_counts / total_count
    
    entropy_value = 0
    for prob in class_probabilities:
        entropy_value -= prob * np.log2(prob)
    
    return entropy_value

# Calculating information gain for attribute
def information_gain(examples, attribute):
    total_entropy = entropy(examples)
    total_count = len(examples)

    remainder = 0
    for value in examples[attribute].unique():
        subset = examples[examples[attribute] == value]
        subset_entropy = entropy(subset)
        remainder += (len(subset) / total_count) * subset_entropy
    
    info_gain = total_entropy - remainder
    return info_gain

# Choosing the attribute with the highest information gain
def choose_best_attribute(attributes, examples):
    best_attribute = None
    max_gain = float('-inf')
    
    for attribute in attributes:
        gain = information_gain(examples, attribute)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    
    return best_attribute

# Printing the decision tree
def print_tree(node, indent=""):
    if node.label is not None:
        print(indent + "Label:", node.label)
    else:
        print(indent + "Attribute:", node.attribute)
        for value, subtree in node.branches.items():
            print(indent + f"{node.attribute} = {value}")
            print_tree(subtree, indent + "    ")

# Determining whether to prune a node using chi-square test
def chi_square_test(node, examples, alpha=0.05):
    observed = []
    class_labels = examples['WillWait'].unique()
    
    for value, _ in node.branches.items():
        subset = examples[examples[node.attribute] == value]
        if len(subset) == 0: # Skipping branches with no examples
            continue
        observed.append([sum(subset['WillWait'] == label) for label in class_labels])
    
    _, p, _, _ = chi2_contingency(observed)

    return p > alpha

# Pruning the decision tree
def prune_tree(node, examples):
    if node.label is not None: # Leaf node
        return
    
    for value, subtree in list(node.branches.items()):
        subset = examples[examples[node.attribute] == value]
        prune_tree(subtree, subset)
    
    # Checking statistical significance
    if chi_square_test(node, examples):
        # Replace the subtree with a leaf node
        node.label = plurality_value(examples)
        node.attribute = None
        node.branches = {}

def main():
    # Defining feature names that are not present in the dataset
    column_names = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Estimate", "WillWait"]
    
    # Data retrieval
    data = pd.read_csv("https://raw.githubusercontent.com/aimacode/aima-data/refs/heads/master/restaurant.csv", header=None, names=column_names)
    attributes = list(data.columns)
    attributes.remove('WillWait')
    
    # Stripping whitespaces from the data to avoid issues
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()

    # Training the decision tree
    decision_tree = learn_decision_tree(data, attributes)
    print("Decision Tree:")
    print_tree(decision_tree)

    # Pruning the decision tree
    prune_tree(decision_tree, data)
    print("\nPruned Decision Tree:")
    print_tree(decision_tree)

if __name__ == "__main__":
    main()
