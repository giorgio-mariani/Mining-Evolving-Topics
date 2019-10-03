import os
from typing import Tuple
from typing import Dict
from typing import List


import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import tqdm

import visualization

def _get_topic_keywords(g:nx.Graph, topic:np.ndarray) -> Dict[str,float]:
    return {g.nodes[ui]["name"]:topic[ui] for ui in g.nodes if topic[ui] > 0}

def keyword_similarity(keyword1:str, keyword2:str) -> float:
    """
    This function computes a simple word similarity between keywords.
    """
    wordbag1 = set(keyword1.split())
    wordbag2 = set(keyword2.split())
    jaccard = len(wordbag1&wordbag2)/len(wordbag1|wordbag2)
    return jaccard#float(keyword1==keyword2)#

def create_topic_mapping(g1:nx.DiGraph, g2:nx.DiGraph, T1:np.ndarray, T2:np.ndarray, method="chamfer") -> list:
    n1 = g1.number_of_nodes()
    n2 = g2.number_of_nodes()
    tn1 = T1.shape[1]
    tn2 = T2.shape[1]

    # compute word similarity for each pair
    S:np.ndarray = np.zeros([g1.number_of_nodes(), g2.number_of_nodes()])
    for ui in g1:
        w1 = g1.nodes[ui]["name"]
        for vi in g2:
            w2 = g2.nodes[vi]["name"]
            S[ui,vi] = keyword_similarity(w1,w2)
    #visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=S)

    #jaccard similarity:
    if method == "jaccard":
        score = np.zeros([tn1,tn2])
        for ti in range(tn1):
            wordsi = set(_get_topic_keywords(g1,T1[:,ti]))
            for tj in range(tn2):
                wordsj = set(_get_topic_keywords(g2,T2[:,tj]))
                intersection = len(wordsi.intersection(wordsj))
                union = len(wordsi.union(wordsj))
                score[ti,tj] = intersection/union if union != 0 else 0
    elif method=="vector":
        # create mapping between topics from g1 to g2
        transposedT1 = np.transpose(T1)
        sizeT1 = transposedT1@S@np.ones([n2])
        tmp = np.expand_dims(sizeT1,axis=1)
        A = np.repeat(tmp, tn2, axis=1)

        sizeT2 = np.ones([n1])@S@T2
        tmp = np.expand_dims(sizeT2, axis=0)
        B = np.repeat(tmp, tn2, axis=0)

        AB = transposedT1@S@T2     #M[t1,t2] = sum_{u,v} T1[u,t1]*T2[v,t2]*S[u,v]
        AUB = A+B-AB
        score =  np.divide(AB, AUB, where=AUB!=0)
    elif method == "chamfer": #SEE "A Point Set Generation Network for 3D Object Reconstruction from a Single Image"
        score = np.zeros([tn1,tn2])
        for ti in range(tn1):
            maski = T1[:,ti]>0
            similarityi = S[T1[:,ti]>0, :]
            sumi = maski.sum()
            for tj in range(tn2):
                maskj = T2[:,tj]>0
                similarityij = similarityi[:,maskj]
                sumj = maskj.sum()
                score[ti,tj] = 1/sumi*similarityij.max(axis=1).sum() + 1/sumj*similarityij.max(axis=0).sum()
    else:
        raise Exception("Wrong mapping method: available values for topic mapping method: 'jaccard', 'vector', and 'chamfer'!")

    # compute mapping
    g1_map_g2 = score.argmax(axis=1) # array containing mapping between topics in g1 to topics in g2
    g2_map_g1 = score.argmax(axis=0) # array containing mapping between topics in g2 to topics in g1

    # REMOVEME visualization of mapping ------------------------------
    visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=score)
    tn1, tn2 = score.shape
    X = np.zeros([tn1,tn2])
    X[range(tn1), g1_map_g2] = score[range(tn1), g1_map_g2]
    X[g2_map_g1, range(tn2)] += score[g2_map_g1, range(tn2)]
    visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=X)

    scorelist = []
    for ti in range(tn1):
        for tj in range(tn2):
            wordsi = set(_get_topic_keywords(g1,T1[:,ti]))
            wordsj = set(_get_topic_keywords(g2,T2[:,tj]))
            if score[ti,tj] != 0:
                scorelist.append( (score[ti,tj], len(wordsi), len(wordsj), ti, tj))
    scorelist.sort()
    #-----------------------------------------------------------------

    mapping = []
    for t1, t2 in enumerate(g1_map_g2):
        if score[t1,t2] != 0:
            mapping.append( (t1, t2, score[t1,t2]) )

    for t2, t1 in enumerate(g2_map_g1):
        if score[t1,t2] != 0:
            mapping.append( (t1, t2, score[t1,t2]) )
    return mapping

'''
def create_chain(
    paths:dict, #TODO add precise type
    tdag: nx.DiGraph,
    keyword_graphs: Dict[int,nx.DiGraph],
    keyword_topics: Dict[int,np.ndarray],
    directory:str,
    k=3):

    #create output directory
    if os.path.exists(directory):
        import shutil
        shutil.rmtree(directory)
    os.makedirs(directory)

    # create loop variables
    year_number = len(keyword_graphs)
    path_weights = dict()
    for u, path in paths.items():
        weight, pathlen = 0, len(path)
        if pathlen == year_number:
            for i in range(pathlen-1):
                weight += tdag[path[i]][path[i+1]]["map_strength"]
            path_weights[u] = weight

    # get heaviest paths
    value_key_list = [(pweight, u) for u, pweight in path_weights.items()]
    value_key_list.sort()
    hpaths_targets = value_key_list[:min(k, len(value_key_list))]
    hpaths = [paths[u] for _, u in hpaths_targets]

    # store paths
    for pi in range(len(hpaths)): # decreasing order
        path = hpaths[pi]
        pdirectory = os.path.join(directory, "path_"+str(len(hpaths)-pi)+".png")
        if not os.path.exists(pdirectory):
            os.makedirs(pdirectory)
        for u in path:
            y, ti = u
            g = keyword_graphs[y]
            t = keyword_topics[y][:,ti]
            f = os.path.join(pdirectory, str(y)+".png") 
            visualization.create_topic_picture_gradient(g=g, topic=t, file=f)
'''

#========================================================================================
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

    print("opening directory "+topic_directory +".\n loading topics ...")
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

def task2(parent_directory:str="topics", topic_directory:str="last", visualize=False):
    gperyear, tperyear = load_topics(parent_directory, topic_directory)
    years:List[int] = [year for year in tperyear.keys()]
    topic_DAG = nx.DiGraph()

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
            attr = {"map_strength": 2-mapstrength}
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
    
    # REMOVE show the topic DAG----------------------------------------
    if visualize:
        import matplotlib.cm as cm
        node_color = [y for y,ti in topic_DAG.nodes()]
        edge_color = [w for u,v,w in topic_DAG.edges.data("map_strength")]
        ypos = {y:i for i,y in enumerate(gperyear.keys())}
        pos = {(y,ti):(ti, ypos[y]*100) for y,ti in topic_DAG.nodes()}
        nx.draw_networkx(
            topic_DAG, pos=pos,
            arrows=False, node_size=40,
            node_color=node_color,
            edge_color=edge_color,
            edge_cmap=cm.get_cmap("binary"),
            linewidths=0, width=0.25,
            with_labels=False)
        plot.show()
    #----------------------------------------------------------------------

    # compute chains
    paths = _create_chains(tdag=topic_DAG, startyear=firstyear)

    '''
    # store heaviest chains
    create_chain(
        tdag=topic_DAG,
        paths=paths,
        keyword_graphs=gperyear,
        keyword_topics=tperyear,
        directory="chains",
        k=10)
    '''


#------------------------------------------------------------------------------
if __name__ == "__main__":
    task2(visualize=True)


