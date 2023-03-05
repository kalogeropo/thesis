from numpy import array, zeros, round, dot, mean
from numpy.linalg import norm
from math import log2, log


def tf(terms):
    tf = {}
    for term in terms:
        if term not in tf:
            tf[term] = 1
        elif term in tf:
            tf[term] += 1
    return tf


def precision_recall(retrieved_docs, relevant):
    
    total, retrieved = 0, 1
    precision, recall = [], []

    # for each document model retrieved
    for doc in retrieved_docs:

        # if it's considered as relevant
        if doc in relevant:
            # count it
            total += 1
            # calculate current precision | recall
            p = total / retrieved
            r = total / len(relevant)

            precision += [p]
            recall += [r]

        # count total retrieved
        retrieved += 1

    # return mean precision and recall for given query
    return mean(precision), mean(recall)


def doc_vectorizer(tf_ij, idf, tnw):
    ########## each column corresponds to a document #########
    if isinstance(tnw, int):
        return round((tf_ij * idf.reshape(-1, 1)), 3)
    else:
        return round((tf_ij * (idf * tnw).reshape(-1, 1)), 3)