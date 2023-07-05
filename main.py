from infre.models import SetBased, GSB, GSBWindow, ConGSB
from numpy import mean
from networkx import to_numpy_array
from infre.preprocess import Collection
from infre.helpers.functions import generate_colors
import networkx as nx
import matplotlib.pyplot as plt
from infre.utils import prune_graph

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
    print(f'SetBased: {mean(pre):.3f}, {mean(rec):.3f}') # 0.166 (raw queries)
    
    # sb_model.save_results(pre, rec)
    # sb_model.save_model('saved_models')
    
    
    ########## GRAPHICAL SET BASED #################### .188 (raw)
    gsb_model = GSB(col).fit(queries)
    pre, rec = gsb_model.evaluate(rels)
    
    # nghrs = gsb_model._number_of_nbrs()
    # print(max(nghrs.values()))
    # print(len(nghrs))
    
    print(f'GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    # # gsb_model.save_results(pre, rec)
    # # gsb_model.save_model('saved_models')
    # print(gsb_model.graph.number_of_nodes(), gsb_model.graph.number_of_edges())
    
    ########## GRAPHICAL SET BASED WITH PRUNING #################### .207 (raw)
    graph, _ = prune_graph(gsb_model.graph, col)

    gsb_pruned_model = GSB(col, graph).fit(queries)
    pre, rec = gsb_pruned_model.evaluate(rels)
    print(f'GSB Pruned: {mean(pre):.3f}, {mean(rec):.3f}')
    """

    ########## GRAPHICAL SET BASED WITH WIWNDOW ####################
    gsb_window_model = GSBWindow(col, window=10).fit(queries)
    pre, rec = gsb_window_model.evaluate(rels)
    print(f'GSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')


    ########## GRAPHICAL SET BASED WITH WIWNDOW AND PRUNING ####################
    graph, _ = prune_graph(gsb_window_model.graph, col)

    gsbw_pruned_model = GSBWindow(col, 10, graph).fit(queries)
    pre, rec = gsbw_pruned_model.evaluate(rels)
    print(f'GSB Window Pruned: {mean(pre):.3f}, {mean(rec):.3f}')


    """
    ########## CONCEPTUALIZED GRAPHICAL SET BASED ####################  .226 (raw queries)
    con_gsb_model = ConGSB(col).fit(queries)
    pre, rec = con_gsb_model.evaluate(rels)

    print(f'Conceptualized GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    print(con_gsb_model.graph.number_of_nodes(), con_gsb_model.graph.number_of_edges())
    """
    
    """
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')

    # Assign a random color to each node based on its cluster
    # n_clusters = len(set([v["cluster"] for _, v in gsb_model.graph.nodes(data=True)]))
    # colors = generate_colors(n_clusters)
    # color_map = {v["cluster"]: colors[i] for i, (_, v) in enumerate(gsb_model.graph.nodes(data=True))}

    # Draw the graph with nodes colored by their clusters
    # nx.draw_networkx(gsb_model.graph, with_labels=False, node_color=[colors[v["cluster"]] for _, v in gsb_model.graph.nodes(data=True)])
    # plt.show()
 


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
