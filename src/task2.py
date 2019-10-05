import os
from typing import Tuple
from typing import Dict
from typing import List


import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import tqdm

import visualization
import misc


def keyword_similarity(keyword1:str, keyword2:str, method="jaccard") -> float:
    """
    This function computes a simple word similarity between keywords.
    The available methods are 'jaccard' and 'equal': in case of jaccard
    the jaccard similarity is used, in case of 'equal' then two words are 
    similar only if they are the same. 
    """
    if method == "jaccard":
        wordbag1 = set(keyword1.split())
        wordbag2 = set(keyword2.split())
        jaccard = len(wordbag1&wordbag2)/len(wordbag1|wordbag2)
        return jaccard
    elif method == "equal":
        return float(keyword1==keyword2)

def create_topic_mapping(
    g1:nx.DiGraph, g2:nx.DiGraph, 
    T1:np.ndarray, T2:np.ndarray, 
    mapping_method="chamfer",
    word_sim_method="jaccard") -> list:

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
            S[ui,vi] = keyword_similarity(w1,w2, method=word_sim_method)
    #visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=S)

    #jaccard similarity:
    if mapping_method == "jaccard":
        score = np.zeros([tn1,tn2])
        for ti in range(tn1):
            wordsi = set(misc.get_topic_keywords(g1,T1[:,ti]))
            for tj in range(tn2):
                wordsj = set(misc.get_topic_keywords(g2,T2[:,tj]))
                intersection = len(wordsi.intersection(wordsj))
                union = len(wordsi.union(wordsj))
                score[ti,tj] = intersection/union if union != 0 else 0
    elif mapping_method == "vector":
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
    elif mapping_method == "chamfer": #SEE "A Point Set Generation Network for 3D Object Reconstruction from a Single Image"
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

    # remove score that is too low
    score[score<0.2] = 0

    # compute mapping
    g1_map_g2 = score.argmax(axis=1) # array containing mapping between topics in g1 to topics in g2
    g2_map_g1 = score.argmax(axis=0) # array containing mapping between topics in g2 to topics in g1

    # REMOVEME visualization of mapping ------------------------------
    #visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=score)
    tn1, tn2 = score.shape
    X = np.zeros([tn1,tn2])
    X[range(tn1), g1_map_g2] = score[range(tn1), g1_map_g2]
    X[g2_map_g1, range(tn2)] += score[g2_map_g1, range(tn2)]
    #visualization.show_word_similarity(g1=g1, g2=g2, wordsimilarity=X)

    scorelist = []
    for ti in range(tn1):
        for tj in range(tn2):
            wordsi = set(misc.get_topic_keywords(g1,T1[:,ti]))
            wordsj = set(misc.get_topic_keywords(g2,T2[:,tj]))
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

def create_chain(
    path:list, 
    keyword_graphs: Dict[int,nx.DiGraph],
    keyword_topics: Dict[int,np.ndarray],
    directory:str):

    #create output directory
    if os.path.exists(directory):
        import shutil
        shutil.rmtree(directory)
    os.makedirs(directory)
    
    for u in path:
        y, ti = u
        g = keyword_graphs[y]
        t = keyword_topics[y][:,ti]
        f = os.path.join(directory, str(y)+".png")
        visualization.create_topic_picture_gradient(g=g, topic=t, file=f)

#========================================================================================
def load_topics(topic_directory:str) -> Tuple[
                                                                                Dict[int, nx.DiGraph], 
                                                                                Dict[int, np.ndarray]]:
    """
    This function loads into memory the graphs and topics generated during task1.
    """
    graphs_per_year:Dict[int, nx.DiGraph] = dict()
    topics_per_year:Dict[int, np.ndarray]  = dict()
    topic_directory_absolute = os.path.abspath(topic_directory)

    print("opening directory "+topic_directory +".\n loading topics ...")
    for name in tqdm.tqdm(os.listdir(topic_directory_absolute)):
        year_directory_abs = os.path.join(topic_directory_absolute, name)
        if os.path.isdir(year_directory_abs):
            if name != "chains":
                year = int(name)
                graph_filename = os.path.join(year_directory_abs, "graph.pickle")
                topic_filename = os.path.join(year_directory_abs, "topics.npy")

                graphs_per_year[year] = nx.read_gpickle(graph_filename)
                topics_per_year[year] = np.load(topic_filename)
    print("topics loaded.")
    return graphs_per_year, topics_per_year

def task2(
    parent_folder:str="topics", 
    topic_folder:str="last", 
    visualize=False,
    mapping_method="chamfer",
    word_sim_method="jaccard"):

    if topic_folder is None:
        topic_directory = os.path.abspath(parent_folder)
    else:
        parent_directory_absolute = os.path.abspath(parent_folder)
        if topic_folder=="last":
            topic_folder=max(os.listdir(parent_directory_absolute))
        topic_directory = os.path.join(parent_directory_absolute, topic_folder)

    gperyear, tperyear = load_topics(topic_directory)
    years:List[int] = [year for year in tperyear.keys()]
    topic_DAG = nx.DiGraph()

    def _create_DAG_nodes(g:nx.Graph, topics:np.ndarray, year:int) -> list:
        nodes_DAG = []
        topic_count = topics.shape[1]
        for ti in range(topic_count):
            nodeid = (year, ti) # node identified by topicindex AND year
            attr = {"topic":misc.get_topic_keywords(g, topics[:,ti])} # attribute to the topic: a dictionary 
                                                                  # describing how much a keyword is
                                                                  # inherent to topic 'ti'
            nodes_DAG.append( (nodeid, attr) )
        return nodes_DAG

    def _create_DAG_edges(mapping:list, source_year:int, target_year:int ) -> list:
        edges = []
        for sourcetopic, targettopic, mapstrength in mapping:
            source_node = (source_year, sourcetopic)
            target_node = (target_year, targettopic)
            attr = {"map_strength": mapstrength}
            edges.append( (source_node, target_node, attr) )
        return edges

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
        mapping = create_topic_mapping(
            g1,g2,T1,T2,
            mapping_method=mapping_method,
            word_sim_method=word_sim_method)
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

    # compute chains ------------------------------------------------------
    tmp_dag = topic_DAG.copy()
    chains_folder = "chains"
    chains_directory = os.path.join(topic_directory, chains_folder)
    if os.path.exists(chains_directory):
        import shutil
        shutil.rmtree(chains_directory)
    os.makedirs(chains_directory)
    for i in tqdm.trange(10):
        longest_path:nx.DiGraph = nx.dag_longest_path(tmp_dag, weight='map_strength', default_weight=None)
        tmp_dag.remove_nodes_from(longest_path)

        chain_directory = os.path.join(chains_directory, "chain_"+str(i))
        # store heaviest chain
        create_chain(
            path=longest_path,
            keyword_graphs=gperyear,
            keyword_topics=tperyear,
            directory=chain_directory)

def create_macrotopic(path, keyword_graphs, keyword_topics):
    return



#------------------------------------------------------------------------------
if __name__ == "__main__":
    task2(visualize=False, word_sim_method="equal")


