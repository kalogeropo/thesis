from os import listdir
from os.path import expanduser, join

from networkx import from_numpy_matrix

from graph_docs import GraphDoc
from document import Corpus

def main():

    # define path
    home = expanduser('~')
    path = f'{home}/Desktop/thesis/data/test_docs'
            
    # list files
    filenames = [join(path, f) for f in listdir(path)]

    inverted_index = {}
    for filename in filenames:
        graph_doc = GraphDoc(filename)

        for key, value in graph_doc.tf.items():
            if key in inverted_index:
                inverted_index[key] += [([graph_doc.doc_id, value])]
            else:
                inverted_index[key] = [[graph_doc.doc_id, value]]

    print("\n\n")
    print(inverted_index)
  

main()