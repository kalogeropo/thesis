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


def ts_idf(termsets, N):
    # len(value) => in how many documents each termset appears
    return array([round(log2(1 + (N / len(value))), 3) for value in termsets.values()])


# termset frequency
def tsf(termsets, inv_index, N):
    #    d1  d2  d3  . . .  di
    # S1 f11 f12 f13 . . . f1i
    # S2     f22            .
    # S3         f33        .
    # .               .     .
    # .                  .  .
    # Sj fj1 fj2 fj3 . . . fij

    tf_ij = zeros((len(termsets), N))
    # for each termset
    for i, (termset, docs) in enumerate(termsets.items()):
        # e.x. termset = fronzenset{'t1', 't2', 't3'}
        terms = list(termset) # ['t1', 't2', 't3']
        temp = {}
        # for each term in the termset
        for term in terms:
            post_list = inv_index[term]['posting_list']
            # for term's id, tf pair
            for id, tf in post_list: 
                # if belongs to the intersection of the termset
                if id in docs:
                    # create a dict to hold frequencies for each term of termset
                    # by taking the min f, we get the termset frequency
                    if id in temp: temp[id] += [tf]
                    else: temp[id] = [tf]

        # assign raw termset frequencies
        for id, tfs in temp.items():
            tf_ij[i, id-1] = round((1 + log2(min(tfs))), 3)

    return array(tf_ij)


def tnw(termsets, nwk):
    termset_weight = []
    for termset in termsets:
        tnw = 1
        for term in termset:
            if term in nwk:
                tnw *= nwk[term]
        termset_weight += [round(tnw, 3)]

    return array(termset_weight)


def doc_vectorizer(tf_ij, idf, tnw):
    ########## each column corresponds to a document #########
    if isinstance(tnw, int):
        return round((tf_ij.T * idf).T, 3)
    else:
        return round((tf_ij.T * (idf * tnw)).T, 3)


def cosine_similarity(u, v):
    if (u == 0).all() | (v == 0).all():
        return 0.
    else:
        return dot(u,v) / (norm(u)*norm(v))


def qd_similarities(query, dtm):
    doc_sim = {}

    for id, doc_vec in enumerate(dtm.T, start=1):
        doc_sim[id] = cosine_similarity(query, doc_vec)

    return doc_sim


def rank_documents(doc_sim):
    return {id: sim for id, sim in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)}


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


    
