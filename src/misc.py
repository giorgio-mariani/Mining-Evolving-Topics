import networkx as nx
import numpy as np
import numpy.linalg as ln

def dictionary_to_numpy(d:dict):
    n = len(d)
    nparray = np.zeros([n])
    for ui in d:
        nparray[ui] = d[ui]
    return nparray

def low_rank_approximation(A:np.ndarray, k) -> np.ndarray:
    u, s, vh = ln.svd(
        A, full_matrices=True, compute_uv=True) # get singular values and associated matrices
    u = u[:, 0:k-1]
    s = s[0:k-1]
    vh = vh[0:k-1,:]
    return np.matmul(np.matmul(u, np.diag(s)), vh)

def edge_vertix_matrix(g:nx.Graph): #TODO refactor for networkx
    n = g.vcount()
    m = g.ecount()
    M=np.zeros([n,m],dtype=float)
    for e in g.es:
        e:ig.Edge
        M[e.source,e.index] = 1
        M[e.target, e.index]= 1
    return M

def local_maxima(g:nx.Graph, f:np.ndarray) -> np.ndarray: # TODO refactor for networkx
    m = np.zeros(g.vcount(), dtype=int)
    for v in g.vs:
        v:ig.Vertex
        p = f[v.index]
        localmax = True
        for n in v.neighbors():
            if p < f[n.index]:
                localmax = False
        if localmax:
            m[v.index] = 1
    return m


# statistics -------------------------------------------------------------------------
def compute_entropy(p:np.ndarray)->float:
    sumentropy = 0
    for pv in p:
        sumentropy += pv*np.log2(pv) if pv != 0 else 0
    return -sumentropy 
    #return -np.sum(p*np.log2(p, where=p!=0)) BUG: 'where' with log2 is not working as intended

def kullback_leiber_divergence(p:np.ndarray, q:np.ndarray)->float:
    p = np.ma.masked_equal(p, 0, copy=False)
    q = np.ma.masked_equal(q, 0, copy=False)
    tmp:np.ma.MaskedArray = p*np.log2(p/q) # using the division may be less stable than log subtraction, but it is more optimazed
    divergence = tmp.filled(0).sum()
    return divergence 

def jensen_shannon_divergence(p:np.ndarray,q:np.ndarray) -> float:
    m = (q+p)/2.0
    divergence = (kullback_leiber_divergence(p,m) + kullback_leiber_divergence(q,m))/2
    return divergence

def cosine_similarity(p:np.ndarray, q:np.ndarray, normalize=True) -> float:
    if normalize:
        pnorm = ln.norm(p)
        qnorm = ln.norm(q)
        p = p/pnorm if pnorm != 0 else p
        q = q/qnorm if qnorm != 0 else q
    return np.dot(p, q)

def euclidean_distance(p:np.ndarray, q:np.ndarray) -> float:
    return ln.norm(p-q)

