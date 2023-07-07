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
    for _ in range(walk_length-1):
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


def prune_graph(graph, collection, n_clstrs=100, condition={}):

    # import needed functions
    from networkx import to_numpy_array
    from infre.tools import SpectralClustering
    from infre.metrics import cosine_similarity

    # Cluster the nodes using spectral clustering
    sc = SpectralClustering(n_clusters=n_clstrs, affinity='precomputed', assign_labels='kmeans')
    
    adj_matrix = to_numpy_array(graph)
    labels, _embeddings = sc.fit_predict(adj_matrix)

    # Remove edges between nodes in different clusters
    for u, v in graph.edges():
        c, w = collection.inverted_index[u]['id'], collection.inverted_index[v]['id']

        try:
            cond = list(condition.keys())[0]
            theshold = list(condition.values())[0]
            
            if cond == 'edge':
                edge_weight = graph.get_edge_data(u, v)['weight']
                flag = edge_weight <= theshold
            elif cond == 'sim':
                flag = cosine_similarity(_embeddings[c, :],  _embeddings[w, :]) <= theshold

        except IndexError:
            flag = 0

        if labels[c] != labels[w] or flag:
            graph.remove_edge(u, v)


    # assign node clusters
    for node in graph.nodes():
        idx = collection.inverted_index[node]['id']
        graph.add_node(node, cluster=labels[idx])

    # convert emebeddings 2D array to a labeled df for future visual exploitation
    from pandas import DataFrame
    embeddings = DataFrame(_embeddings)
    embeddings['labels'] = labels

    return graph, embeddings


def draw_clusters(graph):
    from networkx import draw_networkx
    import matplotlib as plt
    # Assign a random color to each node based on its cluster
    n_clusters = len(set([v["cluster"] for _, v in graph.nodes(data=True)]))
    colors = generate_colors(n_clusters)

    color_map = {v["cluster"]: colors[i] for i, (_, v) in enumerate(graph.nodes(data=True))}

    # Draw the graph with nodes colored by their clusters
    draw_networkx(graph, with_labels=False, node_color=[colors[v["cluster"]] for _, v in graph.nodes(data=True)])
    plt.show()


def plot_scatter_pca(df, c_name, cmap_set="plasma"):
    """
    Visualizes the values of the component columns of the DataFrame according to its column
    that includes the labels.

    Args:
        df: The DataFrame that contains the transformed data after the PCA procedure.
        c_name: The name of the column that includes the labels.
        cmap_set: The format of the plot.

    Returns:
    """

    import matplotlib.pyplot as plt

    if len(df.columns) == 3:
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(16, 8))
        scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.legend(*scatter.legend_elements(), title=c_name)

    elif len(df.columns) == 4:
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        ax.legend(*scatter.legend_elements(), title=c_name)