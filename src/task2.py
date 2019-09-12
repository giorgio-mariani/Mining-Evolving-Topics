import networkx as nx
import numpy as np


def merge_topics(g1:nx.DiGraph, g2:nx.DiGraph, t1:np.ndarray, t2:np.ndarray):
    similarity = np.zeros([g1.number_of_nodes(), g2.number_of_nodes()])
    for ui in g1:
        for vi in g2:
            similarity[ui,vi] = keyword_similarity(ui["name"],vi["name"])
    

def keyword_similarity(keyword1:str, keyword2:str) -> float:
    # TODO: implement
    return 0