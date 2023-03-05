from networkx import Graph, set_node_attributes, get_node_attributes, to_numpy_array, is_empty
from numpy import array, dot, fill_diagonal, zeros
from networkx.readwrite import json_graph
from math import log2
from json import dumps, load
from os.path import join
from pickle import load
import infre.helpers.utilities as utl
from infre.tools.apriori import apriori

from infre.models import SetBased

class GSB(SetBased):
    def __init__(self, collection):
        super().__init__(collection)

        # tensor holds the adjecency matrix of each document
        self.adj_tensor = self.create_adj_tensor()
        
        # empty graph to be filled by union
        self.graph = self.union_graph()

        # NW Weight of GSB
        self.nwk = self._nwk()


    def class_name(self):
        return __class__.__name__


    # create tensor for adj matrices
    def create_adj_tensor(self):

        tensor = []

        for document in self.collection.docs:
            adj_matrix = self.create_adj_matrix(document)
            tensor += [adj_matrix]

        return tensor


    ##############################################
    ## Creating a complete graph TFi*TFj = Wout ##
    ##############################################
    def create_adj_matrix(self, document):

        # get list of term frequencies
        rows = array(list(document.tf.values()))

        # reshape list to column and row vector
        row = rows.reshape(1, rows.shape[0]).T
        col = rows.reshape(rows.shape[0], 1).T

        # create adjecency matrix by dot product
        adj_matrix = dot(row, col)

        # calculate Win weights (diagonal terms)
        win = [(w * (w + 1) * 0.5) for w in rows]

        # assign weights of each nodes
        fill_diagonal(adj_matrix, win)

        return adj_matrix


    def union_graph(self, kcore=[], kcore_bool=False):

        union = Graph()

        # for every graph document object
        for doc, adj_matrix in zip(self.collection.docs, self.adj_tensor):
            terms = list(doc.tf.keys())

            # iterate through lower triangular matrix
            for i in range(adj_matrix.shape[0]):
                # gain value of importance
                h = 0.06 if terms[i] in kcore and kcore_bool else 1

                for j in range(adj_matrix.shape[1]):
                    if i >= j:
                        if union.has_edge(terms[i], terms[j]):
                            union[terms[i]][terms[j]]['weight'] += (adj_matrix[i][j] * h)
                        else:
                            union.add_edge(terms[i], terms[j], weight=adj_matrix[i][j] * h)

        # in-wards edge weights represent Win
        w_in = {n: union.get_edge_data(n, n)['weight'] for n in union.nodes()}
 
        # set them as node attr
        set_node_attributes(union, w_in, 'weight')

        # remove in-wards edges
        for n in union.nodes(): union.remove_edge(n, n)

        return union
        
    
    def win(self):
        return get_node_attributes(self.graph, 'weight')


    def wout(self):
        return {node: val for (node, val) in self.graph.degree(weight='weight')}


    def number_of_nbrs(self):
        return {node: val for (node, val) in self.graph.degree()}


    def _nwk(self, a=1, b=10):

        if is_empty(self.graph): 
            self.graph = self.union_graph()
  
        nwk = {}
        Win = self.win()
        Wout = self.wout()
        ngb = self.number_of_nbrs()
        a, b = a, b

        for k in list(Win.keys()):
            f = a * Wout[k] / ((Win[k] + 1) * (ngb[k] + 1))
            s = b / (ngb[k] + 1)
            nwk[k] = round(log2(1 + f) * log2(1 + s), 3)
            # print(f'log(1 + ({a} * {Wout[k]} / (({Win[k]} + 1) * ({ngb[k]} + 1)) ) ) * log(1 + ({b} / ({ngb[k]} + 1))) = {nwk[k]}')

        return nwk


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
                gf.write(dumps(graph_index, cls=utl.NpEncoder))
        
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