from infre.models import SetBased, GSB, GSBWindow
from numpy import mean
from networkx import to_numpy_array
from infre.tools.collection import Collection


def main():

    # queries = [['a', 'b'], ['a', 'b', 'd', 'n'], ['b', 'h', 'g', 'l', 'm']]
    # rel = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5]]

    path = 'collections/CF/docs'
    ######### example of laod #############
    # sb_model = SetBased(collection=None).load_model(dir='saved models')
    # print(sb_model.collection.inverted_index)

    # create directories, documents and inverted index
    col = Collection(path).create_collection()
    queries, rel = col.load_qd()

    ########## from scratch creation ###########
    ########## SET BASED ####################
    sb_model = SetBased(col).fit(queries)
    pre, rec = sb_model.evaluate(rel)
    print(f'SetBased: {mean(pre):.3f}, {mean(rec):.3f}')
    sb_model.save_results(pre, rec)
    # sb_model.save_model('saved_models')
   
    ########## GRAPHICAL SET BASED ####################
    gsb_model = GSB(col).fit(queries)
    pre, rec = gsb_model.evaluate(rel)
    print(f'GSB: {mean(pre):.3f}, {mean(rec):.3f}')
    gsb_model.save_results(pre, rec)
    # gsb_model.save_model('saved_models')

    ########## GRAPHICAL SET BASED WITH WIWNDOW ####################
    # gsb_window_model = GSBWindow(col, window=10).fit(queries)
    # pre, rec = gsb_window_model.evaluate(rel)
    # print(f'GSBW: {mean(pre):.3f}, {mean(rec):.3f}')
    # gsb_window_model.save_results(pre, rec)
    # gsb_window_model.save_model('saved_models')

    """
    g_emb = Node2Vec(G, dimensions=64, workers=4)

    WINDOW = 10 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words

    model = g_emb.fit(
        vector_size = 16,
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    input_node = 'infection'
    for s in model.wv.most_similar(input_node, topn=10):
        print(s)
"""



# TODO: testing framework, logging result handling
# TODO: fix set based calculation weights and test it with the summing one
# TODO: implement vazirgiannis window and ranking (github: gowpy)


main()
