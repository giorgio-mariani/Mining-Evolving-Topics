import os

import numpy as np
import networkx as nx
import matplotlib
import tqdm

from typing import Dict
from typing import Tuple

DATASET_LOC=os.path.join( os.path.dirname(os.path.abspath(__file__)),"../data/")
DATA_COKEYWORDS="ds-1.tsv"
DATA_COAUTHORS="ds-2.tsv"


def parseAuthor(start:int, end:int) -> Dict[int, nx.DiGraph]:
    """
    Parse the input file containing the co-authorship information.

    Parameters
    ----------
    start : int
        first year from which start to parse the graph data
    end : int 
        last year in which parse the data

    Returns
    -------
    Dict[int, DiGraph]
        a dictionary mapping years in [start, end] to the respective co-authorship graph.
    """
    VerticesPerYear = dict()
    EdgesPerYear = dict()
    with open(DATASET_LOC+DATA_COAUTHORS,'r',encoding="utf8") as f:
        for line in f:
            tokens = line.split("\t")
            
            # get attributes
            year = int(tokens[0])
            a1 = tokens[1]
            a2 = tokens[2]
            w = int(tokens[3])

            if start<=year<=end: 
                v = VerticesPerYear.setdefault(year, set())
                e = EdgesPerYear.setdefault(year, list())

                if not a1 in v:
                    v.add(a1)
                if not a2 in v:
                    v.add(a2)
                e.append((a1,a2,w))
    
    graphsPerYear = dict()
    for year,vertices in VerticesPerYear.items():
        g:nx.DiGraph = graphsPerYear.setdefault(year, nx.DiGraph())
        g.add_nodes_from(list(vertices))

    for year,edges in EdgesPerYear.items():
        g = graphsPerYear[year]
        g.add_weighted_edges_from(edges)
    return graphsPerYear

def parseKeyword(start:int, end:int, usepagerank=True) -> Dict[int, nx.DiGraph]:
    """
    Parse the input file containing the keywords co-occurrence information.

    Extended Summary
    ----------------
    This function generates a dictionary mpping years to keywords graphs, the edges
    are computed by summing the co-occurrence between two keywords. If usepagerank=True, 
    then this is a weighted sum, using as weight the author's pagerank for the year.

    Parameters
    ----------
    start : int
        first year from which start to parse the graph data
    end : int 
        last year in which parse the data
    usepagerank : bool
        whether the edge-weights should also take into consideration the co-authorship graphs pagerank.

    Returns
    -------
    Dict[int, DiGraph]
        a dictionary mapping years in [start, end] to the respective keyword co-occurrence graph.
    """
    if usepagerank:
         # get the author graphs and compute their pageranks
        authorsGraphs = parseAuthor(start=start, end=end)
        print("Computing PageRank ...")
        authorsPageranks = dict()
        for year in tqdm.trange(start,end+1):
            authorsPageranks[year] = nx.pagerank(authorsGraphs[year])

    VerticesPerYear = dict()
    EdgesPerYear = dict()
    def _getAuthors(s:str)->Tuple[list,list]:
        authors = list()
        count = list()
        s = s.replace(" ", "")
        s = s[1:-2] #must remove brackets AND newline
        apairs = s.split(",")
        for apair in apairs:
            a, n =  apair.split(":")
            a = a[1:-1] #remove trailing quotes
            authors.append(a)
            count.append(int(n))
        return authors, count
    
    def _getRank(author:str): # return pagerank scaled such that the average value is 1
        for year in reversed(range(start, end+1)):
            if author in authorsPageranks[year]:
                return authorsPageranks[year][author] * len(authorsPageranks[year])
        print("Warning: the author "+author+" was not found in the author-graphs.")
        return 1

    with open(DATASET_LOC+DATA_COKEYWORDS,'r',encoding="utf8") as f:
        for line in f:
            tokens = line.split("\t")
            year:int = int(tokens[0])
            keyword1:str = tokens[1]
            keyword2:str = tokens[2]

            if start <= year <= end: # check that it is in the correct range
                # compute the weight value
                weight = 0
                authors, count = _getAuthors(tokens[3])
                if usepagerank:
                    rank = [_getRank(a) for a in authors]
                    for ai in range(len(authors)):
                        weight += rank[ai]*count[ai]
                else:
                    weight = sum(count)

                vertices = VerticesPerYear.setdefault(year, set())
                edges = EdgesPerYear.setdefault(year, list())

                if not keyword1 in vertices:
                    vertices.add(keyword1)
                if not keyword2 in vertices:
                    vertices.add(keyword2)
                edges.append((keyword1, keyword2, weight))
                edges.append((keyword2, keyword1, weight))
    
    GraphsPerYear = dict()
    # add vertices to the graphs
    for year, vertices in VerticesPerYear.items():
        graph:nx.DiGraph = GraphsPerYear.setdefault(year, nx.DiGraph())
        vertices = list(vertices)
        vertices.sort()
        graph.add_nodes_from(vertices)
        graph.graph["year"] = year

    # add edges to the graphs
    for year, edges in EdgesPerYear.items():
        graph = GraphsPerYear[year]
        graph.add_weighted_edges_from(edges)
        GraphsPerYear[year] = nx.convert_node_labels_to_integers(
            graph,
            ordering='sorted',
            label_attribute='name')
    return GraphsPerYear