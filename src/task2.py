import os
from typing import Tuple
from typing import Dict
from typing import List


import networkx as nx
import numpy as np
import tqdm

import visualization

def load_topics(parent_directory:str="topics", topic_directory:str="last") -> Tuple[
                                                                                Dict[int, nx.DiGraph], 
                                                                                Dict[int, np.ndarray]]:
    """
    This function loads into memory the graphs and topics generated during task1.
    """
    graphs_per_year:Dict[int, nx.DiGraph] = dict()
    topics_per_year:Dict[int, np.ndarray]  = dict()

    if parent_directory is None:
        topic_directory_absolute = os.path.abspath(parent_directory)
    else:
        parent_directory_absolute = os.path.abspath(parent_directory)
        if topic_directory=="last":
            topic_directory=max(os.listdir(parent_directory_absolute))
        topic_directory_absolute = os.path.join(parent_directory_absolute, topic_directory)

    print("loading topics ...")
    for name in tqdm.tqdm(os.listdir(topic_directory_absolute)):
        year_directory_abs = os.path.join(topic_directory_absolute, name)
        if os.path.isdir(year_directory_abs):
            year = int(name)
            graph_filename = os.path.join(year_directory_abs, "graph.pickle")
            topic_filename = os.path.join(year_directory_abs, "topics.npy")

            graphs_per_year[year] = nx.read_gpickle(graph_filename)
            topics_per_year[year] = np.load(topic_filename)
    print("topics loaded.")
    return graphs_per_year, topics_per_year

def keyword_similarity(keyword1:str, keyword2:str) -> float:
    """
    This function computes a simple word similarity between keywords.
    """
    wordbag1 = set(keyword1.split())
    wordbag2 = set(keyword2.split())
    jaccard = len(wordbag1&wordbag2)/len(wordbag1|wordbag2)
    return jaccard

    '''
    keyword1_words = keyword1.split()
    keyword2_words = keyword2.split()
    n1:int = len(keyword1_words)
    n2:int = len(keyword2_words)
    word_similarity = np.zeros([n1,n2])
    for i1 in range(n1):
        w1 = keyword1_words[i1]
        for i2 in range(n2):
            w2 = keyword2_words[i2]
            word_similarity[i1,i2] = float(w1==w2)
    sim1 = word_similarity.max(axis=1).sum()/n1
    sim2 = word_similarity.max(axis=0).sum()/n2
    return np.maximum(sim1,sim2)
    '''

def create_topic_mapping(g1:nx.DiGraph, g2:nx.DiGraph, T1:np.ndarray, T2:np.ndarray) -> list:
    # compute word similarity for each pair
    S = np.zeros([g1.number_of_nodes(), g2.number_of_nodes()])
    for ui in g1:
        w1 = g1.nodes[ui]["name"]
        for vi in g2:
            w2 = g2.nodes[vi]["name"]
            S[ui,vi] =  keyword_similarity(w1,w2)

    #visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=S)

    # create mapping between topics from g1 to g2
    M = np.transpose(T1)@S@T2     #M[t1,t2] = sum_{u,v} T1[u,t1]*T2[v,t2]*S[u,v]
    g1_to_g2 = M.argmax(axis=1) # array containing mapping between topics in g1 to topics in g2
    g2_to_g1 = M.argmax(axis=0) # array containing mapping between topics in g2 to topics in g1
    mapping = []
    for t1, t2 in enumerate(g1_to_g2):
        mapping.append( (t1, t2, M[t1,t2]) )

    for t2, t1 in enumerate(g2_to_g1):
        mapping.append( (t1, t2, M[t1,t2]) )
    return mapping

def create_chain(
    paths:dict, #TODO add precise type
    tdag: nx.DiGraph,
    keyword_graphs:Dict[int,nx.DiGraph],
    keyword_topics:Dict[int,np.ndarray],
    directory:str,
    k=3):

    #create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create loop variable
    path_weights = dict()
    for u, path in paths.items():
        weight, pathlen = 0, len(path)
        for i in range(pathlen-1):
            weight += tdag[path[i]][path[i+1]]["map_strength"]
        path_weights[u] = weight

    # get heaviest paths
    value_key_list = [(pweight, u) for u, pweight in path_weights.items()]
    value_key_list.sort()
    hpaths_targets = value_key_list[-min(k, len(value_key_list)):]
    hpaths = [paths[u] for _, u in hpaths_targets]

    # store paths
    for pi in reversed(range(len(hpaths))): # decreasing order
        path = hpaths[pi]
        pdirectory = os.path.join(directory, "path_"+str(pi)+".png")
        if not os.path.exists(pdirectory):
            os.makedirs(pdirectory)
        for u in path:
            y, ti = u
            g = keyword_graphs[y]
            t = keyword_topics[y][:,ti]
            f = os.path.join(pdirectory, str(y)+".png")
            visualization.create_topic_picture_gradient(g=g, topic=t, file=f)

def create_macro_topics():
    #TODO implement
    return

def task2(parent_directory:str="topics", topic_directory:str="last"):
    gperyear, tperyear = load_topics(parent_directory, topic_directory)
    years:List[int] = [year for year in tperyear.keys()]
    topic_DAG = nx.DiGraph()

    def _get_topic_keywords(g:nx.Graph, topic:np.ndarray) -> Dict[str,float]:
        return {g.nodes[ui]["name"]:topic[ui] for ui in g.nodes if topic[ui] > 0}

    def _create_DAG_nodes(g:nx.Graph, topics:np.ndarray, year:int) -> list:
        nodes_DAG = []
        topic_count = topics.shape[1]
        for ti in range(topic_count):
            nodeid = (year, ti) # node identified by topicindex AND year
            attr = {"topic":_get_topic_keywords(g, topics[:,ti])} # attribute to the topic: a dictionary 
                                                                  # describing how much a keyword is
                                                                  # inherent to topic 'ti'
            nodes_DAG.append( (nodeid, attr) )
        return nodes_DAG

    def _create_DAG_edges(mapping:list, source_year:int, target_year:int ) -> list:
        edges = []
        for sourcetopic, targettopic, mapstrength in mapping:
            source_node = (source_year, sourcetopic)
            target_node = (target_year, targettopic)
            attr = {"map_strength":mapstrength}
            edges.append( (source_node, target_node, attr) )
        return edges

    def _create_chains(tdag:nx.DiGraph, startyear:int):
        S = [(y,t) for y,t in tdag if y==startyear]
        return nx.multi_source_dijkstra_path(
            tdag, sources=S, weight="map_strength")

    # add nodes in the first year
    firstyear = years[0]
    firstgraph = gperyear[firstyear]
    firsttopic = tperyear[firstyear]
    topic_DAG.add_nodes_from(_create_DAG_nodes(firstgraph, firsttopic, firstyear))

    # add leyers to the DAG
    for i in tqdm.trange(len(years)-1):
        # get years
        year1 = years[i]
        year2 = years[i+1]
        # load graphs
        g1 = gperyear[year1]
        g2 = gperyear[year2]
        # load topics
        T1 = tperyear[year1]
        T2 = tperyear[year2]
        mapping = create_topic_mapping(g1,g2,T1,T2)
        # add nodes and edges to the DAG
        topic_DAG.add_nodes_from(_create_DAG_nodes(g2, T2, year2))
        topic_DAG.add_edges_from(_create_DAG_edges(mapping, year1, year2))
    


    # compute chains
    paths = _create_chains(tdag=topic_DAG, startyear=firstyear)

    # store heaviest chains
    create_chain(
        tdag=topic_DAG,
        paths=paths,
        keyword_graphs=gperyear,
        keyword_topics=tperyear,
        directory="chains")

    # compute macro topics
    #create_macro_topics(gperyear, tperyear)




#------------------------------------------------------------------------------
if __name__ == "__main__":
    task2()


