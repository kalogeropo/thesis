from time import time
from numpy import mean
from infre.models.cgsb import ConGSB
from infre.preprocess.collection import Collection

# Constants
#
COLLECTION_PATH =  "collections/CF/docs"

def main():
    # Load queries and relevant documents
    queries, rels = Collection.load_qd("collections/CF")
    col = Collection(COLLECTION_PATH).create(first=-1)

    # Iterate over similarity values and evaluate the model
    for sim in range (2, 10):
        print(f"Iteration {sim} with Similarity: {sim/10}")
        start_time = time()
        cgsb_model = ConGSB(col, clusters=50, cond={'sim': sim/10},cluster_optimization="eigen_gap")
        pre, rec = cgsb_model.fit_evaluate(queries, rels)
        print(f'CGSB: {mean(pre):.3f}, {mean(rec):.3f}')
        print(cgsb_model.graph.number_of_nodes(), cgsb_model.graph.number_of_edges())
        cgsb_model.save_results(pre, rec)
        print(f"Time: {time()-start_time:.2f}s")

# Entry point
if __name__ == "__main__":
    main()
