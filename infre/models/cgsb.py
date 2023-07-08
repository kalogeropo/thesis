# TODO: cluster size variance 

# TODO: sentiment analysis - by product
# 1. set-based, 2.gsb 3.gsbw 4.cgsb 5.cwgsb 6. 

from infre.models import GSB
from numpy import zeros
from infre.helpers.functions import prune_graph

class ConGSB(GSB):
    def __init__(self, collection, clusters, cond={}):
        
        super().__init__(collection)

        self.model = self._model()

        self.graph, self.embeddings = prune_graph(self.graph, collection, n_clstrs=clusters, condition=cond)

        self._cnwk()


    def _model(self): return __class__.__name__
    

    def _cnwk(self):
        # Dictionary to store computed _cnwk values for each cluster
        cluster_cnwk = {}
        
        for term in self.collection.inverted_index:
            # Get the cluster of the current term
            cluster = self.graph.nodes[term]['cluster']
            
            # Check if _cnwk value for the cluster has been computed before
            if cluster not in cluster_cnwk:
                _cnwk = 0
                cluster_size = 0
                
                # Iterate over nodes and compute _cnwk for the cluster
                for node, attrs in self.graph.nodes(data=True):
                    if attrs['cluster'] == cluster:
                        cluster_size += 1
                        _cnwk += self.collection.inverted_index[node]['nwk']
                
                # Compute average _cnwk value for the cluster
                cluster_cnwk[cluster] = round(_cnwk / cluster_size, 3)
            
            # Assign the computed _cnwk value to the current term
            self.collection.inverted_index[term]['cnwk'] = cluster_cnwk[cluster]

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