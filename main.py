from os import listdir
from os.path import expanduser, join

from networkx import from_numpy_matrix, draw
from matplotlib.pyplot import show

from graph_docs import GraphDoc, GraphUnion
from document import Corpus

def main():

    # define path
    home = expanduser('~')
    path = f'{home}/Desktop/thesis/data/test_docs'
            
    # list files
    filenames = [join(path, f) for f in listdir(path)]
    graph_documents = []
    inverted_index = {}
    i = 0
    for filename in filenames:
        graph_doc = GraphDoc(filename)
        graph_documents += [graph_doc]
        # print(graph_doc.tf)
        """
        for key, value in graph_doc.tf.items():
            if key in inverted_index:
                inverted_index[key] += [[graph_doc.doc_id, value]]
            else:
                inverted_index[key] = [[graph_doc.doc_id, value]]
        """


    # takes as input list of graph document objects
    ug = GraphUnion(graph_documents)
    union_graph = ug.union_graph()
    print(union_graph)
    labels = {n: union_graph.nodes[n]['weight'] for n in union_graph.nodes}
    # labels = {n: (n, union_graph.nodes[n]['weight']) for n in union_graph.nodes()}
    colors = [union_graph.nodes[n]['weight'] for n in union_graph.nodes]
    draw(union_graph, with_labels=True, labels=labels, node_color=colors)
    show()
  

main()