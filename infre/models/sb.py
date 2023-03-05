from numpy import array
from math import log2
from json import load
from os.path import join, exists
from os import makedirs, getcwd
from pickle import dump, load
from bz2 import BZ2File
from pandas import DataFrame
from infre.tools.apriori import apriori
from infre import retrieval

from infre.models import BaseIRModel
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
    

    def query2vec(self, termsets, N):

        # number of docs
        N = self.collection.num_docs
        
        # len(value) => in how many documents each termset appears
        return array([round(log2(1 + (N / len(value))), 3) for value in termsets.values()])
    

    def fit(self, queries, mf=1):
        
        # number of documents
        N = self.collection.num_docs

        # inverted index of collection documents
        inv_index = self.collection.inverted_index

        # for each query
        for i, query in enumerate(queries, start=1):
        
            print(f"=> Query {i} of {len(queries)}")

            # apply apriori to find frequent termsets
            freq_termsets = apriori(query, inv_index, min_freq=mf)
            
            print(f"Query length: {len(query)} | Frequent Termsets: {len(freq_termsets)}")
           
            idf = retrieval.query_vectorizer(freq_termsets, N)
            
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