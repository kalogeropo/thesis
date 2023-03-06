from numpy import round


def tf(terms):
    tf = {}
    for term in terms:
        if term not in tf:
            tf[term] = 1
        elif term in tf:
            tf[term] += 1
    return tf


def doc_vectorizer(tf_ij, idf, tnw):
    ########## each column corresponds to a document #########
    if isinstance(tnw, int):
        return round((tf_ij * idf.reshape(-1, 1)), 3)
    else:
        return round((tf_ij * (idf * tnw).reshape(-1, 1)), 3)