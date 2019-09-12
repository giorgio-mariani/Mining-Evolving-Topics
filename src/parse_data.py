import numpy as np
import networkx as nx
import matplotlib

DATASET_LOC="../data/"
DATA_COKEYWORDS="ds-1.tsv"
DATA_COAUTHORS="ds-2.tsv"

def parseKeyword():
    V = dict()
    E = dict()
    
    def getAuthors(s:str)->dict:
        authors = dict()
        s = s.replace(" ", "")
        s = s[1:-2] #must remove brackets AND newline
        apairs = s.split(",")
        for apair in apairs:
            a, n =  apair.split(":")
            a = a[1:-1] #remove trailing quotes
            authors[a] = int(n)
        return authors

    with open(DATASET_LOC+DATA_COKEYWORDS,'r',encoding="utf8") as f:
        for line in f:
            tokens = line.split("\t")
            year = int(tokens[0])
            k1 = tokens[1]
            k2 = tokens[2]
            authors = getAuthors(tokens[3])
            w = sum(authors.values()) #NOTE: sum(authors) is atleast 2 if authors is not empty

            v = V.setdefault(year, set())
            e = E.setdefault(year, list())

            if not k1 in v:
                v.add(k1)
            if not k2 in v:
                v.add(k2)
            e.append((k1, k2, w))
            e.append((k2, k1, w))
    
    G = dict()
    # add vertices to the graphs
    for year, vertices in V.items():
        graph:nx.DiGraph = G.setdefault(year, nx.DiGraph())
        vertices = list(vertices)
        vertices.sort()
        graph.add_nodes_from(vertices)
        graph.graph["year"] = year

    # add edges to the graphs
    for year, edges in E.items():
        graph = G[year]
        graph.add_weighted_edges_from(edges)
        G[year] = nx.convert_node_labels_to_integers(
            graph,
            ordering='sorted',
            label_attribute='name')
    return G

def parseAuthor():
    V = dict()
    E = dict()
    with open(DATASET_LOC+DATA_COAUTHORS,'r',encoding="utf8") as f:
        for line in f:
            tokens = line.split("\t")
            
            # get attributes
            year = int(tokens[0])
            a1 = tokens[1]
            a2 = tokens[2]
            w = tokens[3]

            v = V.setdefault(year, set())
            e = E.setdefault(year, list())

            if not a1 in v:
                v.add(a1)
            if not a2 in v:
                v.add(a2)
            e.append((a1,a2,w))
    
    G = dict()
    for y,v in V.items():
        g:nx.DiGraph = G.setdefault(y, nx.DiGraph())
        g.add_nodes_from(list(v))

    for y,e in E.items():
        g = G[year]
        g.add_weighted_edges_from(e)
    return G

def _analysis(G:dict):
    years = list(G.keys())
    years.sort()
    for year in years:
        graph:nx.Graph = G[year]
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        c = 2 if graph.is_directed() else 1
        d = c*m/(n*n)
        print("year "+str(year)+":"+
            "\t|V|="+str(n)+"\t|E|="+str(m)+"\td="+str(d))

