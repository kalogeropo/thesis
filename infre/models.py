from networkx import Graph, set_node_attributes, get_node_attributes, to_numpy_array, is_empty
from numpy import array, dot, fill_diagonal, zeros
from networkx.readwrite import json_graph
from json import dumps, load, loads
from os.path import join
from os import makedirs

import infre.helpers.utilities as utl
from infre.tools.apriori import apriori
from infre.retrieval import *


class SetBased():
    def __init__(self, collection=None):

        # model name
        self.model = self.get_class_name()

        # collection handles all needed parsed data
        self.collection = collection

        # vector queries holds a list of idf/vector query of each query
        self.vector_queries = []

        # docs tensor holds documents representation in vector space of each query
        self.docs_tensor = []


    def get_class_name(self):
        return __class__.__name__


    def compile(self):
        pass


    def fit(self, queries, mf=1):

        # N = self.collection.number
        N = 1239
        inv_index = self.collection.inverted_index

        for i, query in enumerate(queries, start=1):
        
            print(f"=> Query {i} of {len(queries)}")
    
            freq_termsets = apriori(query, inv_index, min_freq=mf)
            
            print(f"Query length: {len(query)} | Frequent Termsets: {len(freq_termsets)}")
       
            idf = calculate_ts_idf(freq_termsets, N)

            tf_ij = calculate_tsf(freq_termsets, inv_index, N)

            # clucnky solution for polymorphism
            try:
                tnw = calculate_tnw(freq_termsets, self.nwk)
            except AttributeError:
                tnw = 1
            
            # represent documents in vector space
            doc_vectors = calculate_doc_vectors(tf_ij, idf, tnw=tnw)

            # keep local copy for dtm of every document
            self.docs_tensor += [doc_vectors]

            # keep local copy for every query vector
            self.vector_queries += [idf]

            # print(f'{(time() - start):.2f} secs.\n') TODO: maybe loadbar

        return self


    def evaluate(self, relevant):
        avg_pre, avg_rec = [], []

        for i, (q, doc_vectors, rel) in enumerate(zip(self.vector_queries, self.docs_tensor, relevant)):

            document_similarities = evaluate_sim(q, doc_vectors)

            pre, rec = calc_precision_recall(document_similarities.keys(), rel)

            print(f"=> Query {i+1} of {len(self.vector_queries)}")
            print(f'Precision: {pre:.3f} | Recall: {rec:.3f}\n')

            avg_pre.append(round(pre, 3))
            avg_rec.append(round(rec, 3))

        return array(pre), array(rec)


    def fit_evaluate(self, queries, relevant, mf=1):
        pass

    
    def save_model(self, name=f'config.model'):

        # import pick method
        from pickle import dump

        # define indexes path
        path = join(self.model, name)
       
        try:
            with open(path, 'wb') as config_model:
                dump(self, config_model)

        except FileNotFoundError:
                # create directories
                makedirs(self.model)

                # call method recursively to complete the job
                self.save_model(name)

        finally: # if fails again, reteurn object
            return self


    def load_model(self, name='config.model'):
        
        # import pick method
        from pickle import load

        # define indexes path
        path = join(self.model, name)

        try:
            with open(path, 'rb') as config_model:

                return load(config_model)

        except FileNotFoundError:
                raise FileNotFoundError


class GSB(SetBased):
    def __init__(self, collection):
        super().__init__(collection)

        # tensor holds the adjecency matrix of each document
        self.adj_tensor = self.create_adj_tensor()
        
        # empty graph to be filled by union
        self.graph = self.union_graph()

        # NW Weight of GSB
        self.nwk = self.calculate_nwk()


    def get_class_name(self):
        return __class__.__name__

    # create tensor for adj matrices
    def create_adj_tensor(self):
        matrices = []
        for document in self.collection.docs:
            adj_matrix = self.create_adj_matrix(document)
            matrices += [adj_matrix]

        return matrices


    ##############################################
    ## Creating a complete graph TFi*TFj = Wout ##
    ##############################################
    def create_adj_matrix(self, document):
        if document.tf is not None:

            # get list of term frequencies
            rows = array(list(document.tf.values()))

            # reshape list to column and row vector
            row = rows.reshape(1, rows.shape[0]).T
            col = rows.reshape(rows.shape[0], 1).T

            # create adjecency matrix by dot product
            adj_matrix = dot(row, col)

            # calculate Win weights (diagonal terms)
            win = [(w * (w + 1) * 0.5) for w in rows]
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
        
    
    def calculate_win(self):
        return get_node_attributes(self.graph, 'weight')


    def calculate_wout(self):
        return {node: val for (node, val) in self.graph.degree(weight='weight')}


    def number_of_nbrs(self):
        return {node: val for (node, val) in self.graph.degree()}


    def calculate_nwk(self, a=1, b=10):

        if is_empty(self.graph): 
            self.graph = self.union_graph()
  
        nwk = {}
        Win = self.calculate_win()
        Wout = self.calculate_wout()
        ngb = self.number_of_nbrs()
        a, b = a, b

        for k in list(Win.keys()):
            f = a * Wout[k] / ((Win[k] + 1) * (ngb[k] + 1))
            s = b / (ngb[k] + 1)
            nwk[k] = round(log(1 + f) * log(1 + s), 3)
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


class GSBWindow(GSB):
    def __init__(self, collection, window=8):
        if isinstance(window, int):
            self.window = window
        elif isinstance(window, float):
            num_of_words = len(self.collection.inverted_index)
            self.window = int(num_of_words * window) + 1
        super().__init__(collection)


    def get_class_name(self):
        return __class__.__name__


    # overide basic method with the windowed one
    def create_adj_matrix(self, document):

        windows_size = self.window

        # create windowed document
        windowed_doc = document.split_document(windows_size)

        adj_matrix = zeros(shape=(len(document.tf), len(document.tf)), dtype=int)
        for segment in windowed_doc:
            w_tf = calculate_tf(segment)

            for i, term_i in enumerate(document.tf):
                for j, term_j in enumerate(document.tf):
                    if term_i in w_tf.keys() and term_j in w_tf.keys():
                        if i == j:
                            adj_matrix[i][j] += w_tf[term_i] * (w_tf[term_i] + 1) / 2
                        else:
                            adj_matrix[i][j] += w_tf[term_i] * w_tf[term_j]
        return adj_matrix

