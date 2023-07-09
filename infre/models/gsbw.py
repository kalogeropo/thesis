from numpy import zeros
from infre import retrieval

from infre.models import GSB
class GSBWindow(GSB):
    def __init__(self, collection, window=7):
        self.window = window
        super().__init__(collection)


    def class_name(self): return __class__.__name__


    # overide basic method with the windowed one
    def doc2adj(self, document):

        window_size = - 1
        if isinstance(self.window, int):
            window_size = self.window
        elif isinstance(self.window, float):
            window_size = int(self.window * len(document.terms))

        # create windowed document
        windowed_doc = document.split_document(window_size)

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