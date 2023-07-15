from networkx import Graph, set_node_attributes, get_node_attributes, to_numpy_array, is_empty
from numpy import array, dot, fill_diagonal, zeros
from networkx.readwrite import json_graph
from math import log2, log
from json import dumps, load
from pickle import load, dump
from os.path import join, exists
from os import makedirs, getcwd
from bz2 import BZ2File

from infre.models import BaseIRModel
from time import time

class GSB(BaseIRModel):
    def __init__(self, collection):
        super().__init__(collection)
        
        # model name
        self.model = self._model()

        # empty graph to be filled by union
        self.graph = self.union_graph()
        
        # NW Weight of GSB
        self._nwk()
        
        # average wout edge weight
        self.avg_wout = sum(self._wout().values()) / (2 * self.graph.number_of_edges())
        # print(f"Average Wout edge weight on {self._model()} = {self.avg_wout}")


    def _model(self): return __class__.__name__

    
    def _model_func(self, termsets): 
        
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
    
    
    # tsf * idf for set based
    def _vectorizer(self, tsf_ij, idf, *args):

        tns, *_ = args
        ########## each column corresponds to a document #########
        return tsf_ij * (idf * tns).reshape(-1, 1)


    ##############################################
    ## Creating a complete graph TFi*TFj = Wout ##
    ##############################################
    def doc2adj(self, document):

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
        return get_node_attributes(self.graph, 'weight')


    def _wout(self):
        return {node: val for (node, val) in self.graph.degree(weight='weight')}


    def _number_of_nbrs(self):
        return {node: val for (node, val) in self.graph.degree()}


    def _nwk(self, a=1, b=10):

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