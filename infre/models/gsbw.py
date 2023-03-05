from numpy import zeros
from infre import retrieval

from infre.models import GSB
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