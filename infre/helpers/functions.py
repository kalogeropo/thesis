import numpy as np

# Define a function to generate a single random walk
def generate_random_walk(graph, start_node, walk_length, transition_probs):
    """
    Generates a single random walk of a given length from a starting node,
    using transition probabilities defined by the node2vec parameters.

    Parameters:
    - graph: a NetworkX graph object
    - start_node: the node to start the random walk from
    - walk_length: the length of the random walk
    - transition_probs: a dictionary of transition probabilities for each node in the graph

    Returns:
    - walk: a list of nodes visited in the random walk
    """
    walk = [start_node]
    for i in range(walk_length-1):
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))
        if len(neighbors) == 0:
            break
        if len(walk) == 1:
            # On the first step, only consider the neighbors that have edges to the starting node
            weights = [transition_probs[current_node][neighbor] for neighbor in neighbors
                       if graph.has_edge(start_node, neighbor)]
        else:
            # On subsequent steps, use the transition probabilities based on p and q
            weights = [transition_probs[current_node][neighbor] for neighbor in neighbors]
        weights = np.array(weights)
        weights /= np.sum(weights)
        next_node = np.random.choice(neighbors, size=1, p=weights)[0]
        walk.append(next_node)
    return walk


# Define a function to generate multiple random walks
def generate_random_walks(graph, walk_length, num_walks, p, q):
    """
    Generates multiple random walks of a given length from each node in a graph,
    using transition probabilities defined by the node2vec parameters.

    Parameters:
    - graph: a NetworkX graph object
    - walk_length: the length of the random walks
    - num_walks: the number of random walks to generate for each node
    - p: the return parameter of node2vec
    - q: the in-out parameter of node2vec

    Returns:
    - walks: a list of all generated random walks
    """
    # Compute transition probabilities based on p and q
    transition_probs = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        """
        weights = [1.0 / graph.degree(neighbor) if neighbor == node else 1.0 / (q * graph.degree(neighbor))
                   if graph.has_edge(neighbor, node) else 1.0 / (p * graph.degree(neighbor))
                   for neighbor in neighbors]
        """
        weights = []
        for neighbor in neighbors:
            if neighbor == node:
                weights.append(1.0 / graph.degree(neighbor))
            elif graph.has_edge(neighbor, node):
                weights.append(1.0 / (q * graph.degree(neighbor)))
            else:
                weights.append(1.0 / (p * graph.degree(neighbor)))

        weights = np.array(weights)
        weights /= np.sum(weights)
        transition_probs[node] = dict(zip(neighbors, weights))
    # Generate random walks starting from all nodes
    walks = []
    for i in range(num_walks):
        for node in graph.nodes():
            walk = generate_random_walk(graph, node, walk_length, transition_probs)
            walks.append(walk)
    return walks


import random

def generate_colors(n):
    colors = []
    for _ in range(n):
        color_code = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color_code)
    return colors