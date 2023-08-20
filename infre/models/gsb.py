from networkx import Graph, set_node_attributes, get_node_attributes, is_empty
from numpy import array, dot, fill_diagonal, zeros
from networkx.readwrite import json_graph
from math import log2, log
from json import dumps, load
from pickle import load, dump
from os.path import join, exists
from os import makedirs, getcwd
from bz2 import BZ2File
from infre.models import BaseIRModel

class GSB(BaseIRModel):
    """
    Graphicl Set-Based (GSB) Information Retrieval Model based on kalogeropoulos et. al.
    """
    def __init__(self, collection):
        """
        Initializes the GSB model with a given collection.

        Parameters:
            collection (object): The collection of documents to be processed.
        """
        super().__init__(collection)
        
        # model name
        self.model = self._model()

        # Create Union Graph
        self.graph = self.union_graph()
        
        # NW Weight of GSB
        self._nwk()
        
        # average wout edge weight
        self.avg_wout = sum(self._wout().values()) / (2 * self.graph.number_of_edges())
        # print(f"Average Wout edge weight on {self._model()} = {self.avg_wout}")


    def _model(self): return __class__.__name__

    
    def _model_func(self, termsets): 
        """
        Calculates the weight for each termset based on the product of the NW weights of its terms (kalogeropoulos et. al).

        Parameters:
            termsets (list): List of termsets.

        Returns:
            numpy.array: Array of weights for each termset.
        """
        tns = zeros(len(termsets), dtype=float)
        inv_index = self.collection.inverted_index

        for i, termset in enumerate(termsets):
            tw = 1
            for term in termset:
                if term in inv_index:
                    # get the nw weight of term k and mulitply it to total
                    tw *= inv_index[term]['nwk']
            tns[i] = round(tw, 3)

        return tns 
    
    
    def _vectorizer(self, tsf_ij, idf, *args):
        """
        Computes the vector representation of terms based on termset frequency (TSF) 
        and inverse document frequency (IDF).

        Parameters:
            tsf_ij (numpy.array): Term set frequencies.
            idf (numpy.array): Inverse document frequencies.
            *args: Variable length argument list. In this model, argument is the 
                    array of weights for each termset returned from _model_func.

        Returns:
            numpy.array: Vectorized representation.
        """

        tns, *_ = args
        ########## each column corresponds to a document #########
        return tsf_ij * (idf * tns).reshape(-1, 1)


    def doc2adj(self, document):
        """
        Computes the adjacency matrix for the terms in a document based on their 
        term frequencies using Makris algorithm.

        Parameters:
            document (object): Document with term frequencies.

        Returns:
            numpy.array: Adjacency matrix.
        """

        # get list of term frequencies
        rows = array(list(document.tf.values())).reshape((-1, 1))

        # create adjecency matrix by dot product
        adj_matrix = dot(rows, rows.T)
        
        f = lambda w: (w * (w + 1) * 0.5)

        # calculate Win weights (diagonal terms)
        win = [f(w) for w in rows]

        # assign weights of each nodes
        fill_diagonal(adj_matrix, win)

        return adj_matrix


    def union_graph(self, kcore=[], kcore_bool=False):
        """
        Constructs the union graph of the collection using adjacency matrices 
        of individual documents.

        Parameters:
            kcore (list, optional): List of core terms for importance weighting.
            kcore_bool (bool, optional): Whether to use importance weighting.

        Returns:
            NetworkX Graph: Union graph of the collection.
        """
        
        union = Graph() # empty graph 

        # for every graph document object
        for doc in self.collection.docs:

            # get terms of each document
            terms = list(doc.tf.keys())

            # create it's adjacency matrix based on Makris algorithm
            adj_matrix = self.doc2adj(doc)
                
            for i in range(adj_matrix.shape[0]):
                # gain value of importance
                h = 0.06 if terms[i] in kcore and kcore_bool else 1

                for j in range(adj_matrix.shape[1]):
                    # iterate through lower triangular matrix
                    if i >= j:
                        if union.has_edge(terms[i], terms[j]):
                            union[terms[i]][terms[j]]['weight'] += (adj_matrix[i][j] * h)  # += Wout
                        else:
                            # if term frequency not 0
                            if adj_matrix[i][j] > 0:
                                union.add_edge(terms[i], terms[j], weight=adj_matrix[i][j] * h)

        # in-wards edge weights represent Win
        w_in = {n: union.get_edge_data(n, n)['weight'] for n in union.nodes()}

        # set them as node attr
        set_node_attributes(union, w_in, 'weight')

        # remove self edges
        for n in union.nodes(): union.remove_edge(n, n)
        
        return union
        
    
    def _win(self):
        """
        Retrieves the inward edge weights (Win) for each node in the graph.

        Returns:
            dict: Dictionary of inward edge weights indexed by node.
        """
        return get_node_attributes(self.graph, 'weight')


    def _wout(self):
        """
        Computes the outward edge weights (Wout) for each node in the graph.

        Returns:
            dict: Dictionary of outward edge weights indexed by node.
        """
        return {node: val for (node, val) in self.graph.degree(weight='weight')}


    def _number_of_nbrs(self):
        """
        Computes the number of neighbors for each node in the graph.

        Returns:
            dict: Dictionary of number of neighbors indexed by node.
        """
        return {node: val for (node, val) in self.graph.degree()}


    def _nwk(self, a=1, b=10):
        """
        Computes the node weights (NWk) for each term in the collection using 
        inward and outward edge weights and updates the inverted index of the 
        collection with these weights, based on kalogeropulos et. al equation.

        Parameters:
            a (float, optional): Weighting parameter. Default is 1.
            b (float, optional): Weighting parameter. Default is 10.

        Returns:
            None
        """

        if is_empty(self.graph): 
            raise("Union Graph must be constructed first.")
        
        Win = self._win()
        Wout = self._wout()
        ngb = self._number_of_nbrs()
        a, b = a, b

        for k in list(Win.keys()):
            f = a * Wout[k] / ((Win[k] + 1) * (ngb[k] + 1))
            s = b / (ngb[k] + 1)
            self.collection.inverted_index[k]['nwk'] = round(log2(1 + f) * log2(1 + s), 3)
            # print(f'log(1 + ({a} * {Wout[k]} / (({Win[k]} + 1) * ({ngb[k]} + 1)) ) ) * log(1 + ({b} / ({ngb[k]} + 1))) = {nwk[k]}')

        return
    

    # picke model
    def save_model(self, path, name='config.model'):
        
        # define indexes path
        dir = join(getcwd(), path, self.model)
   
        if not exists(dir):
            makedirs(dir)

        path = join(dir, name)

        try:
            with BZ2File(path, 'wb') as config_model:
                dump(self, config_model)

        except FileNotFoundError:
                FileNotFoundError


    # un-picke model
    def load_model(self, **kwargs):

        # define indexes path
        try:
            path = join(getcwd(), kwargs['dir'], self.model, kwargs['name'])
        except KeyError:
            raise KeyError

        try:
            with BZ2File(path, 'rb') as config_model:
                return load(config_model)

        except FileNotFoundError:
                raise FileNotFoundError
        

    def save_graph_index(self, name='graph_index.json'):
        
        # check if union is created, otherwise auto-create
        if is_empty(self.graph): self.graph = self.union_graph()

        # define path to store index
        path = join(self.path['index_path'], name)

        # format data to store
        graph_index = json_graph.adjacency_data(self.graph)

        try:
            # store via the help of json dump
            with open(path, "w") as gf:
                gf.write(dumps(graph_index))
        
        # if directory does not exist
        except FileNotFoundError:
                # create directory 
                self.create_model_directory()

                # call method recursively to complete the job
                self.save_graph_index()
        finally: # if fails again, reteurn object
            return self



    def load_graph(self, name='graph_index.json'):

        # path to find stored graph index
        path = join(self.path['index_path'], name)

        try:
            # open file and read as dict
            with open(path) as gf: js_graph = load(gf)
        
        except FileNotFoundError:
            raise('There is no such file to load collection.')

        self.graph = json_graph.adjacency_graph(js_graph)

        return self.graph