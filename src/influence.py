import collections as cl
import networkx as nx
import numpy as np

def normalize_weights(
    g:nx.DiGraph, 
    weight:str="weight", 
    mode:str='mixed'):
    weight_sums = np.array([d for u,d in g.in_degree(weight=weight)])
    max_sum = weight_sums.max()

    if mode=="local":
        scale = 1/weight_sums
    elif mode == "global":
        scale = np.full(weight_sums.shape, 1.0/max_sum)
    elif mode == "mixed":
        tmp = np.log(1 + weight_sums) #NOTE: assuming min != 0
        tmp = tmp/tmp.max() # normalize max indegree sum to one
        scale = tmp/weight_sums # create scale factor

    for vi in g:
        for ui in g.predecessors(vi):
            g[ui][vi][weight] = g[ui][vi][weight] * scale[vi]

def linear_threshold(
    g:nx.DiGraph, 
    seeds:set, 
    theta:np.ndarray, 
    weight:str="weight") -> np.ndarray:

    vertex_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    V = np.zeros([vertex_count])
    for u in seeds:
        V[u] = 1.0
    F = cl.deque(seeds)
    while F:
        vi:int = F.popleft()
        for ui in g.successors(vi):
            if V[ui] < theta[ui]:
                V[ui] += g[vi][ui][weight]
                if V[ui] >= theta[ui]:
                    F.append(ui)
    return V >= theta

def linear_threshold_mean(
    g:nx.DiGraph, 
    seeds:set, 
    samples=5, 
    weight:str="weight") -> np.ndarray:
    
    vertex_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    counter = np.zeros(vertex_count)
    for _ in range(samples):
        theta = np.random.rand(vertex_count)
        activenodes = linear_threshold(
            g, seeds, theta,
            weight=weight)
        counter += activenodes
    return counter/samples

def influence_maximization(g:nx.DiGraph, k:int):
    seed = set()
    rank = np.array(g.pagerank(directed=False))
    for _ in range(k):
        maxcoverage = 0
        newseed = 0
        for v in range(g.number_of_nodes()):
            infected_nodes = linear_threshold_mean(g, seed.union({v}) , samples=2)
            coverage = infected_nodes.sum()
            if maxcoverage<coverage or (maxcoverage==coverage and rank[v]>=rank[newseed]):
                maxcoverage = coverage
                newseed = v
        seed.add(newseed)
    return seed
