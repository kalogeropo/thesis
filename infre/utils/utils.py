def prune_graph(graph, collection, n_clstrs=0):

    from networkx import to_numpy_array
    from infre.tools import SpectralClustering

    # Cluster the nodes using spectral clustering
    sc = SpectralClustering(n_clusters=150, affinity='precomputed', assign_labels='kmeans')
    
    adj_matrix = to_numpy_array(graph)
    labels, _embeddings = sc.fit_predict(adj_matrix)
    
    # Remove edges between nodes in different clusters
    for u, v in graph.edges():
        c, w = collection.inverted_index[u]['id'], collection.inverted_index[v]['id']

        if labels[c] != labels[w]:
            # union.add_node(u, cluster=labels[c])
            # union.add_node(v, cluster=labels[w])
            graph.remove_edge(u, v)

    # assign node clusters
    for node in graph.nodes():
        idx = collection.inverted_index[node]['id']
        graph.add_node(node, cluster=labels[idx])

    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # # dim reduction with SVD
    # from sklearn.decomposition  import PCA
    # embSvd = PCA(2).fit_transform(_embeddings)

    # for i in np.unique(labels):
    #     plt.scatter(embSvd[labels == i, 0], embSvd[labels == i, 1], label=i)
    # plt.show()
    # print("Dellta average")
    # print(sum(value for _, value in union.degree()) / union.number_of_nodes())

    return graph, _embeddings