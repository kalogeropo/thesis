from numpy import dot
from numpy.linalg import norm

def cosine_similarity(u, v):
    if (u == 0).all() | (v == 0).all():
        return 0.
    else:
        return dot(u,v) / (norm(u)*norm(v))