from infre.models import SetBased, GSB, ConGSB
from numpy import mean
from networkx import to_numpy_array
from infre.preprocess import Collection
from infre.helpers.functions import generate_colors
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # queries = [['a', 'b'], ['a', 'b', 'd', 'n'], ['b', 'h', 'g', 'l', 'm']]
    # rels = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5]]

    path = 'collections/CF/docs'
    ######### example of laod #############
    # sb_model = SetBased(collection=None).load_model(dir='saved models')

    # load queries, relevant documents
    queries, rels = Collection.load_qd('collections/CF')

    # create collection object
    # most needed attrs: inverted index and size of collection
    col = Collection(path).create()
    print("Collection Done!")
    """
    ########## SET BASED ####################
    sb_model = SetBased(col).fit(queries)
    pre, rec = sb_model.evaluate(rels)
    print(f'SetBased: {mean(pre):.3f}, {mean(rec):.3f}')
    
    # sb_model.save_results(pre, rec)
    # sb_model.save_model('saved_models')
   """
    ########## GRAPHICAL SET BASED ####################
    gsb_model = GSB(col).fit(queries)
    pre, rec = gsb_model.evaluate(rels)
    
    print(f'GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_model.save_results(pre, rec)
    # gsb_model.save_model('saved_models')
    
    ########## GRAPHICAL SET BASED WITH WIWNDOW ####################
    con_gsb_model = ConGSB(col).fit(queries)
    pre, rec = con_gsb_model.evaluate(rels)

    print(f'Conceptualized GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    print(con_gsb_model.graph.number_of_nodes(), con_gsb_model.graph.number_of_edges())
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')

    # Assign a random color to each node based on its cluster
    # n_clusters = len(set([v["cluster"] for _, v in gsb_model.graph.nodes(data=True)]))
    # colors = generate_colors(n_clusters)
    # color_map = {v["cluster"]: colors[i] for i, (_, v) in enumerate(gsb_model.graph.nodes(data=True))}

    # Draw the graph with nodes colored by their clusters
    # nx.draw_networkx(gsb_model.graph, with_labels=False, node_color=[colors[v["cluster"]] for _, v in gsb_model.graph.nodes(data=True)])
    # plt.show()

    ########## GRAPHICAL SET BASED WITH WIWNDOW ####################
    # gsb_window_model = GSBWindow(col, window=10).fit(queries)
    # pre, rec = gsb_window_model.evaluate(rels)
    # print(f'GSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')
    """
    from node2vec import Node2Vec
    
    node2vec = Node2Vec(gsb_model.graph, dimensions=64, workers=4)
    
    WINDOW = 10 # Node2Vec fit window
    MIN_COUNT = 2 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words

    model = node2vec.fit(
        vector_size = 16,
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    input_node = 'CF'
    for s in model.wv.most_similar(input_node, topn=10):
        print(s)


    gsb_model.load_model()
    """
   
# TODO: testing framework, logging result handling
# TODO: fix set based calculation weights and test it with the summing one
# TODO: implement vazirgiannis window and ranking (github: gowpy)
