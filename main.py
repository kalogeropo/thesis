from infre.models import SetBased, GSB, GSBWindow
from numpy import mean
from networkx import to_numpy_array
from infre.tools.collection import Collection


def main():

    # queries = [['a', 'b', 'd', 'n'], ['b', 'h', 'g', 'l']]
    # rel = [[1, 2, 3, 4], [2, 3, 4, 5]]

    path = 'collections/CF'

    # create directories, documents and inverted index
    col = Collection(path).create_collection()
    queries, rel = col.load_qd()

    # sb_model = SetBased(collection=None).load_model()
    
    sb_model = SetBased(col).fit(queries)
    pre, rec = sb_model.evaluate(rel)
    print(f'SetBased: {mean(pre), mean(rec)}')
    sb_model.save_model()

    gsb_model = GSB(col).fit(queries)
    pre, rec = gsb_model.evaluate(rel)
    print(f'GSB: {mean(pre), mean(rec)}')
    gsb_model.save_model()

    gsb_window_model = GSBWindow(col, window=7).fit(queries)
    pre, rec = gsb_window_model.evaluate(rel)
    print(f'GSBW: {mean(pre), mean(rec)}')
    gsb_window_model.save_model()


        
# df = DataFrame(list(zip(avg_pre, avg_rec)), columns=["A_pre", "A_rec"])
# test_writer = excelwriter()
# stest_writer.write_results('', df)


# TODO: testing framework, logging result handling
# TODO: fix set based calculation weights and test it with the summing one
# TODO: implement vazirgiannis window and ranking (github: gowpy)


main()
