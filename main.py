from infre.models import SetBased, GSB, GSBWindow, PGSB, PGSBW, ConGSB, ConGSBWindow
from numpy import mean
from networkx import to_numpy_array
from infre.preprocess import Collection
from infre.helpers.functions import generate_colors
import networkx as nx
import matplotlib.pyplot as plt
from infre.helpers.functions import prune_graph

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
    col = Collection(path).create(first=-1)
    
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
    
    print(f'GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_model.save_results(pre, rec)
    # gsb_model.save_model('saved_models')
    print(gsb_model.graph.number_of_nodes(), gsb_model.graph.number_of_edges())
    
    ########## GRAPHICAL SET BASED WITH PRUNING #################### .207 (raw)
    # graph, _ = prune_graph(gsb_model.graph, col, n_clstrs=100)

    # gsb_pruned_model = GSB(col, graph).fit(queries)
    # pre, rec = gsb_pruned_model.evaluate(rels)
    # print(f'GSB Pruned: {mean(pre):.3f}, {mean(rec):.3f}')
    
    pgsb_model = PGSB(col, clusters=50).fit(queries)
    pre, rec = pgsb_model.evaluate(rels)
    
    print(f'PGSB: {mean(pre):.3f}, {mean(rec):.3f}')

    """
    ########## GRAPHICAL SET BASED WITH WIWNDOW ####################
    gsb_window_model = GSBWindow(col, window=.15).fit(queries)
    print(gsb_window_model.graph)
    
    pre, rec = gsb_window_model.evaluate(rels)
    print(f'GSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')
    
    ########## GRAPHICAL SET BASED WIWNDOW WITH PRUNING ####################
    # graph, _ = prune_graph(gsb_window_model.graph, col, n_clstrs=100)

    # gsbw_pruned_model = GSBWindow(col, 10, graph).fit(queries)
    # pre, rec = gsbw_pruned_model.evaluate(rels)
    # print(f'GSB Window Pruned: {mean(pre):.3f}, {mean(rec):.3f}')
    

    # pgsbw_model = PGSBW(col, window=7, clusters=50).fit(queries)
    # pre, rec = pgsbw_model.evaluate(rels)
    
    # print(f'PGSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    """
    ########## CONCEPTUALIZED GRAPHICAL SET BASED ####################  .226 (raw queries)
    con_gsb_model = ConGSB(col, clusters=50, cond={'sim': 0.3}).fit(queries)
    pre, rec = con_gsb_model.evaluate(rels)

    print(f'CGSB: {mean(pre):.3f}, {mean(rec):.3f}')
    print(con_gsb_model.graph.number_of_nodes(), con_gsb_model.graph.number_of_edges())
    
    """
    ######### CONCEPTUALIZED GRAPHICAL SET BASED Window ####################  
    con_gsbw_model = ConGSBWindow(col, window=.15, clusters=50, cond={'sim':.2}).fit(queries)
    pre, rec = con_gsbw_model.evaluate(rels)

    print(f'CGSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    print(con_gsbw_model.graph.number_of_nodes(), con_gsbw_model.graph.number_of_edges())
    
    """
    import numpy as np

    # dim reduction with SVD
    from sklearn.decomposition  import PCA
    embSvd = PCA(2).fit_transform(_embeddings)

    for i in np.unique(labels):
        plt.scatter(embSvd[labels == i, 0], embSvd[labels == i, 1], label=i)
    plt.show()
    print("Dellta average")
    print(sum(value for _, value in union.degree()) / union.number_of_nodes())
    """

    from time import time
    from node2vec import Node2Vec

    start = time()

    # Define the embedding dimensions
    dimensions = 64

    # Create a Node2Vec instance with specified dimensions
    node2vec = Node2Vec(con_gsbw_model.graph, dimensions=dimensions, workers=4)

    # Train the model and generate the walks
    model = node2vec.fit(window=10, min_count=2, batch_words=4)

    # Use the trained model to find most similar nodes
    input_node = 'pseudomonas'
    similar_nodes = model.wv.most_similar(input_node, topn=10)

    # Print the most similar nodes
    for node, similarity in similar_nodes:
        print(node, similarity)


    print(time()-start)
    # gsb_model.load_model()