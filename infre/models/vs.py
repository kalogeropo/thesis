from math import log2
from numpy import array,zeros

from infre import retrieval


class VectorSpace():
    def __init__(self, collection=None):

        # model name
        self.model = self.class_name()

        self.inv_index = collection.inverted_index
        N = self.collection.number

        # idf vector 
        self.idf_ = [round(log2(N / len(self.inv_index[term]['posting_list'])), 3) for term in self.inv_index]

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