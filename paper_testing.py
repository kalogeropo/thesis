from time import time
from numpy import array, mean
from sklearn.cluster import KMeans
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
    # sim = 3 # manuly selection of similarity value !time issues!
        print(f"Iteration {sim} with Similarity: {sim/10}")
        start_time = time()
        cgsb_model = ConGSB(col, clusters=50, cond={'sim': sim/10},cluster_optimization="silhouette")
        pre, rec = cgsb_model.fit_evaluate(queries, rels)
        print(f'CGSB: {mean(pre):.3f}, {mean(rec):.3f}')
        print(cgsb_model.graph.number_of_nodes(), cgsb_model.graph.number_of_edges())
        cgsb_model.save_results(f"sil_{cgsb_model.model}",pre, rec)
        print(f"Time: {time()-start_time:.2f}s")

# Entry point
if __name__ == "__main__":
    #main()
    queries, rels = Collection.load_qd("collections/CF")
    col = Collection(COLLECTION_PATH).create(first=-1)
    test = []
    for i in range(50,200,20):
        cgsb_model = ConGSB(col, clusters=i, cond={'sim': 5/10})
        x = cgsb_model.embeddings.drop(columns=["labels"])
        kmeans = KMeans(n_clusters=int(i), random_state=0).fit(x)
        print(kmeans.n_iter_)
        test.append(kmeans.n_iter_)
    print(test)

