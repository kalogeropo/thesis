from numpy import mean, array, zeros
from math import log2

from infre.retrieval import cosine_similarity

class BaseIRModel:
    def __init__(self, collection):

        # collection object to handle documents and inverted index
        self.collection = collection

        # array used as a 3D tensor to hold all the 
        # ermset per document frequency generated for each query
        self._docs2vec = []

        # array used as matrix to hold the query vector 
        # representation generated by each query
        self._q2vec = []

        # each index position corresponds to the mean precision of each query
        self.precision = []

        # each index position corresponds to the mean recall of each query
        self.recall = []


    def query2vec(self, termsets):

        # number of docs
        N = self.collection.num_docs
        
        # len(value) => in how many documents each termset appears
        return array([round(log2(1 + (N / len(value))), 3) for value in termsets.values()])
    

    # termset frequency
    def termsets2vec(self, termsets):
        #    d1  d2  d3  . . .  di
        # S1 f11 f12 f13 . . . f1i
        # S2     f22            .
        # S3         f33        .s
        # .               .     .
        # .                  .  .
        # Sj fj1 fj2 fj3 . . . fij

        # number of documents/columns
        N = self.collection.num_docs

        # get inv index
        inv_index = self.collection.inverted_index

        # initialize zero matrix with the appropriate dims
        tsf_ij = zeros((len(termsets), N))

        # for each termset
        for i, (termset, docs) in enumerate(termsets.items()):
            # e.x. termset = fronzenset{'t1', 't2', 't3'}
            terms = list(termset) # ['t1', 't2', 't3']
            temp = {}
            # for each term in the termset
            for term in terms:
                post_list = inv_index[term]['posting_list']
                # for term's id, tf pair
                for id, tf in post_list: 
                    # if belongs to the intersection of the termset
                    if id in docs:
                        # create a dict to hold frequencies for each term of termset
                        # by taking the min f, we get the termset frequency
                        if id in temp: temp[id] += [tf]
                        else: temp[id] = [tf]

            # assign raw termset frequencies
            for id, tfs in temp.items():
                tsf_ij[i, id-1] = round((1 + log2(min(tfs))), 3)
        
        return array(tsf_ij)


    # get cos similarity for each document-query pair
    def qd_similarities(self, query, dtsm):
        return {id: cosine_similarity(query, dv) for id, dv in enumerate(dtsm.T, start=1)}


    def rank_documents(self, qd_sims):
        return {id: sim for id, sim in sorted(qd_sims.items(), key=lambda item: item[1], reverse=True)}

