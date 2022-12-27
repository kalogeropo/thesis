from networkx import Graph, set_node_attributes, get_node_attributes, to_numpy_array, is_empty
from numpy import array, dot, fill_diagonal, zeros
from networkx.readwrite import json_graph
from math import log2
from json import dumps, load, loads
from os.path import join, exists
from os import makedirs, getcwd
from pickle import dump, load
from bz2 import BZ2File
from pandas import DataFrame

import infre.helpers.utilities as utl
from infre.tools.apriori import apriori
from infre import retrieval


class SetBased():
    def __init__(self, collection=None):

        # model name
        self.model = self.class_name()

        # collection handles all needed parsed data
        self.collection = collection

        # vector queries holds a list of idf/vector query of each query
        self.vector_queries = []

        # docs tensor holds documents representation in vector space of each query
        self.dtm_tensor = []
        
        # each index position corresponds to the mean precision of each query
        self.precision = []

        # each index position corresponds to the mean recall of each query
        self.recall = []


    def class_name(self):
        return __class__.__name__


    def fit(self, queries, mf=1):

        N = self.collection.number
     
        inv_index = self.collection.inverted_index

        for i, query in enumerate(queries, start=1):
        
            print(f"=> Query {i} of {len(queries)}")
    
            freq_termsets = apriori(query, inv_index, min_freq=mf)
            
            print(f"Query length: {len(query)} | Frequent Termsets: {len(freq_termsets)}")
       
            idf = retrieval.ts_idf(freq_termsets, N)

            sf_ij = retrieval.tsf(freq_termsets, inv_index, N)

            # clucnky solution for polymorphism
            try:
                tnw = retrieval.tnw(freq_termsets, self.nwk)
            except AttributeError:
                tnw = 1
            
            # represent documents in vector space
            dtm = retrieval.doc_vectorizer(sf_ij, idf, tnw=tnw)
    
            # keep local copy for dtm of every document
            self.dtm_tensor += [dtm]

            # keep local copy for every query vector
            self.vector_queries += [idf]

            # print(f'{(time() - start):.2f} secs.\n') TODO: maybe loadbar
        print('\n')

        return self


    def evaluate(self, relevant):

        # for each query and (dtm, relevant) pair
        for i, (q, dtm, rel) in enumerate(zip(self.vector_queries, self.dtm_tensor, relevant)):
            
            # cosine similarity between query and every document
            qd_sims = retrieval.qd_similarities(q, dtm)

            # rank them in desc order
            retrieved_docs = retrieval.rank_documents(qd_sims)

            # precision | recall of ranking
            pre, rec = retrieval.precision_recall(retrieved_docs.keys(), rel)

            print(f"=> Query {i+1} of {len(self.vector_queries)}")
            print(f'Precision: {pre:.3f} | Recall: {rec:.3f}')

            self.precision.append(round(pre, 3))
            self.recall.append(round(rec, 3))

        return array(self.precision), array(self.recall)


    def fit_evaluate(self, queries, relevant, mf=1):
        pass

    
    def save_results(self, *args):

        # pre, rec = args if args else self.precision, self.recall
        if args:
            pre, rec = args
        else:
            pre, rec = self.precision, self.recall
     
        df = DataFrame(list(zip(pre, rec)), columns=["precision", "recall"])
      
        path = join(getcwd(), 'saved_models', self.model, 'results')

        if not exists(path): makedirs(path)

        df.to_excel(join(path, f'{self.model.lower()}.xlsx'))

        return self


    def save_model(self, path, name='config.model'):
        
        # define indexes path
        dir = join(getcwd(), path, self.model)
   
        if not exists(dir): makedirs(dir)

        path = join(dir, name)

        try:
            with BZ2File(path, 'wb') as config_model:
                dump(self, config_model)

        except FileNotFoundError:
                FileNotFoundError


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


class GSB(SetBased):
    def __init__(self, collection):
        super().__init__(collection)

        # tensor holds the adjecency matrix of each document
        self.adj_tensor = self.create_adj_tensor()
        
        # empty graph to be filled by union
        self.graph = self.union_graph()

        # NW Weight of GSB
        self.nwk = self.calculate_nwk()


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


class GSBWindow(GSB):
    def __init__(self, collection, window=8):
        if isinstance(window, int):
            self.window = window
        elif isinstance(window, float):
            num_of_words = len(self.collection.inverted_index)
            self.window = int(num_of_words * window) + 1
        super().__init__(collection)


    def class_name(self):
        return __class__.__name__


    # overide basic method with the windowed one
    def create_adj_matrix(self, document):

        windows_size = self.window

        # create windowed document
        windowed_doc = document.split_document(windows_size)

        adj_matrix = zeros(shape=(len(document.tf), len(document.tf)), dtype=int)
        for segment in windowed_doc:
            w_tf = retrieval.tf(segment)

            for i, term_i in enumerate(document.tf):
                for j, term_j in enumerate(document.tf):
                    if term_i in w_tf.keys() and term_j in w_tf.keys():
                        if i == j:
                            adj_matrix[i][j] += w_tf[term_i] * (w_tf[term_i] + 1) / 2
                        else:
                            adj_matrix[i][j] += w_tf[term_i] * w_tf[term_j]
        return adj_matrix


class VSM():
    def __init__(self, collection=None):

        # model name
        self.model = self.class_name()

        # collection handles all needed parsed data
        self.collection = collection

        # idf vector 
        self.idf_ = [round(log2(self.collection.number / len(self.collection.inverted_index[term]['posting_list'])), 3) 
                            for term in self.collection.inverted_index]

        # tf matrix
        self.tf_idf_ = self.tf_idf()
        print(self.tf_idf_)
        # each index position corresponds to the mean precision of each query
        self.precision = []

        # each index position corresponds to the mean recall of each query
        self.recall = []
    

    def query_vectorizer(self, query):
        qv = []
        inv_index = self.collection.inverted_index 
        for term in inv_index:
            i = inv_index[term]['id']
            if term in query:
                qv += [self.idf_[i]]
            else:
                qv += [0.]

        return array(qv)


    def fit_evaluate(self, queries, relevant):

        N = self.collection.number

        for query, rel in zip(queries, relevant):
       
            q = self.query_vectorizer(query)
                
            # cosine similarity between query and every document
            qd_sims = retrieval.qd_similarities(q, self.tf_idf_)
    
            # rank them in desc order
            retrieved_docs = retrieval.rank_documents(qd_sims)
            
            # precision | recall of ranking
            pre, rec = retrieval.precision_recall(retrieved_docs.keys(), rel)

            print(f'Precision: {pre:.3f} | Recall: {rec:.3f}')

            self.precision.append(round(pre, 3))
            self.recall.append(round(rec, 3))

        return array(self.precision), array(self.recall)


    def class_name(self):
        return __class__.__name__


    def tf_idf(self):

        # inverted index of the collection
        inv_index = self.collection.inverted_index

        # Document objects
        docs = self.collection.docs

        # dtm
        tf_idf_matrix = zeros(shape=(len(inv_index), len(docs)))

        for j, doc in enumerate(docs):
            for term, tf in doc.tf.items():
                i = inv_index[term]['id']
                tf_idf_matrix[i][j] += round((1 + (tf)) * self.idf_[i], 3)

        return tf_idf_matrix