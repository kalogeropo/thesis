# TODO: find avg weight of edges and use it as condition in edge removal
# TODO: cluster size variance 
# TODO: Incorporate cluster graph in window gsb
# TODO: CONCEPTS ALSO IN WINDOWED GSB

# TODO: sentiment analysis - by product
# 1. set-based, 2. gsb 3. pgsb 4. cgsb 5. cwgsb 6. 

from infre.models import GSB
from numpy import zeros
from infre.utils import prune_graph

class ConGSB(GSB):
    def __init__(self, collection):
        super().__init__(collection)

        self.model = self._model()

        self.graph, self.embeddings = prune_graph(self.graph, collection)

        self._cnwk()
        # This model introduces a conceputalize aspect
        # each term have each own concept
        # TODO: probably insert into inverted index
        self.concepts = []

    def _model(self): return __class__.__name__

    """
    def union_graph(self, kcore=[], kcore_bool=False):
        union = super().union_graph(kcore, kcore_bool)

        from networkx import to_numpy_array
        # # from sklearn.cluster import SpectralClustering
        from infre.tools import SpectralClustering

        # Cluster the nodes using spectral clustering
        sc = SpectralClustering(n_clusters=50, affinity='precomputed', assign_labels='discretize')
        
        adj_matrix = to_numpy_array(union)
        labels, _embeddings = sc.fit_predict(adj_matrix)
        
        # Remove edges between nodes in different clusters
        for u, v in union.edges():
            c, w = self.collection.inverted_index[u]['id'], self.collection.inverted_index[v]['id']

            if labels[c] != labels[w]:
                # union.add_node(u, cluster=labels[c])
                # union.add_node(v, cluster=labels[w])
                union.remove_edge(u, v)

        # assign node clusters
        for node in union.nodes():
            idx = self.collection.inverted_index[node]['id']
            union.add_node(node, cluster=labels[idx])

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

        return union
    """
    

    def _cnwk(self):

        # inv_index = self.collection.inverted_index
        # cnwk = zeros(len(inv_index), dtype=float)

        for term in self.collection.inverted_index:
            # try:
            #     self.collection.inverted_index[term]['cnwk']
            #     flag = True
            # except KeyError:
            #     flag = False

            # if not flag:
                _cnwk = 0
                cluster_size = 0
                neighbors = []
                # find it's cluster and it's neigbhors in that space
                cluster = self.graph.nodes[term]['cluster']
                # TODO: CHECK ONLY IT'S NEIGBHORS
                for node, attrs in self.graph.nodes(data=True):
                    if attrs['cluster'] == cluster:
                        cluster_size += 1
                        _cnwk += self.collection.inverted_index[node]['nwk']
                        # neighbors.append(node)

                # cnwk[i] = _cnwk / cluster_size
                self.collection.inverted_index[term]['cnwk'] = round(_cnwk / cluster_size, 3)
                # update neighbors in same cluster
                # for ngb in neighbors: 
                #    self.collection.inverted_index[ngb]['cnwk'] = round(_cnwk / cluster_size, 3)

        # print(self.collection.inverted_index)
        return 


    def _model_func(self, termsets): 

        inv_index = self.collection.inverted_index
        tns = zeros(len(termsets), dtype=float)

        for i, termset in enumerate(termsets):
            tw = 1
            for term in termset:
                if term in inv_index:
                    # get the nw weight of term k and mulitply it to total
                    tw *= inv_index[term]['cnwk']
            tns[i] = tw

        return tns 
    
    
    # tsf * idf for set based
    def _vectorizer(self, tsf_ij, idf, *args):

        tns, *_ = args
        ########## each column corresponds to a document #########
        return tsf_ij * (idf * tns).reshape(-1, 1)