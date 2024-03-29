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
    Compute similarity between two keywords.

    Extended Summary
    ----------------
    This function computes a simple word similarity between keywords.
    The available methods are 'jaccard' and 'equal': in case of jaccard
    the jaccard similarity is used, in case of 'equal' then two words are 
    similar only if they are the same.

    Parameters
    ----------
    keyword1 : str
    keyword2 : str
    method : str
        method to use in order to compute the similarity. The available values are
        'equal', which uses exact match, and 'jaccard', which uses the jaccard similarity
        of the bag-of-words representation of keyword1 and keyword2.

    Returns
    -------
    float (in [0,1])
        similarity between keyword1 and keyword2
    """
    if method == "jaccard":
        wordbag1 = set(keyword1.split())
        wordbag2 = set(keyword2.split())
        jaccard = len(wordbag1&wordbag2)/len(wordbag1|wordbag2)
        return jaccard
    elif method == "equal":
        return float(keyword1==keyword2)

def create_topic_links(
    g1:nx.DiGraph, g2:nx.DiGraph, 
    T1:np.ndarray, T2:np.ndarray, 
    mapping_method="chamfer",
    word_sim_method="jaccard",
    mode="full") -> list:
    """
    Compute a map betweeen topics in a first keyword co-occurrence graph to a second one.

    Parameters
    ----------
    g1: DiGraph
        first keyword co-occurrence graph
    g2: DiGraph
        second keyword co-occurrence graph
    T1: ndarray
        keywords-topics matrix, storing membershio information between keywords in g1 to topics.
    T2: ndarray
        keywords-topics matrix, storing membershio information between keywords in g2 to topics.
    mapping_method : str
        method used to score the similarity between two topics. The available values are:
        'chamfer', 'vector', 'jaccard'.
    word_sim_method : str
        method used to score the similiraty between two keywords. The available values are:
        'jaccard', and 'equal'.

    Returns
    -------
    List[( int, int, float)]
        list of tuple representing a mapping between topics in T1 to topics in T2.
        These tuples contain 3 elements: (ti,tj, ms) with ti a topic-id in T1, tj a topic-id in Tj 
        and ms the map-strength between ti and tj.
    """

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

    # low scores removal
    score[score<0.2] = 0

    # compute mapping
    mapping = []
    if mode == "reduced": # reduced mode --------------------------------------
        g1_map_g2 = score.argmax(axis=1) # array containing mapping between topics in g1 to topics in g2
        g2_map_g1 = score.argmax(axis=0) # array containing mapping between topics in g2 to topics in g1
        # compute mapping
        for t1, t2 in enumerate(g1_map_g2):
            if score[t1,t2] != 0:
                mapping.append( (t1, t2, score[t1,t2]) )
        for t2, t1 in enumerate(g2_map_g1):
            if score[t1,t2] != 0:
                mapping.append( (t1, t2, score[t1,t2]) )

    elif mode == "full": # full mode ------------------------------------------
        for ti in range(tn1):
            for tj in range(tn2):
                if score[ti][tj] > 0:
                    mapping.append( (ti, tj, score[ti,tj]) )
    else:
        raise Exception("Wrong link generation mode: available values are: 'full' or 'reduced'!")
    return mapping

# =============================================================================
def candidate_chain(tdag:nx.DiGraph, s:Tuple[int,int]) -> List[Tuple[int,int]]:
    """
    Compute a candidate topic-chain using a greedy approach.
    
    parameters
    ----------
    tdag : DiGraph 
        full topic dag (with connection between also non-adjacent years)
    s : 
        (year, topic_id), starting node for the chain
    
    Returns
    -------
    List[(int,int)]
        topic-chain containing nodes [(y0,t0) ... (yi,ti)]
    """
    alpha = 1/3
    path = [s]
    sumscore = 0
    while True:
        maxscore = 0
        argmax = None
        u = path[-1]

        # control successors
        for v in tdag.successors(u):
            score = 0
            for z in reversed(path):
                x =  tdag[z][v]["map_strength"] if tdag.has_edge(z,v) else 0
                score = alpha*x + (1-alpha)*score
            if score >= maxscore:
                argmax = v
                maxscore = score 

        # check loop conditions
        if argmax is not None:
            path.append(argmax)
            sumscore += maxscore
        else:
            score = sumscore#/np.sqrt(len(path))
            words = set().union(*[tdag.nodes[u]["topic"] for u in path])
            return score, path, words

def chains_extraction(tdag:nx.DiGraph, k:int = 20)->list:
    """
    Compute a set of k optimal topic-chains from full topic-DAG.
    
    Parameters
    ----------
    tdag : DiGraph
        full topic-DAG
    k : int
        desired number of output topic-chains

    Returns
    -------
    List[(flaot, list, set)]
        list of tuples (score, chain, words), where score is the chains score, chain
        is the actual chain as a list of nodes and words are the keywords in the chain.
    """

    '''REMOVEME
    tdag = tdag.copy()
    paths = []
    for i in tqdm.trange(k):
        longest_path:nx.DiGraph = nx.dag_longest_path(tdag, weight='map_strength', default_weight=None)
        paths.append(longest_path)
        tdag.remove_nodes_from(longest_path)
    return paths
    '''
    chains = []
    for y,t in tqdm.tqdm(tdag):
        score, path, words = candidate_chain(tdag, s=(y,t))
        chains.append( (score, path, words) )

    top_chains = []
    for _ in tqdm.tqdm(range(k)):
        score, top_chain, top_words = max(chains)
        top_chains.append(top_chain)
        for i in range(len(chains)):
            score, path, words = chains[i]
            s = len(top_words.intersection(words))/len(top_words.union(words))
            chains[i] = score*(1-s), path, words
    return top_chains


#========================================================================================
def load_topics(topic_directory:str) -> Tuple[Dict[int, nx.DiGraph], Dict[int, np.ndarray]]:
    """
    Load into memory the graphs and topics generated during Task-1.

    Parameters
    ----------
    topic_directory : str
        directory containing the topic data.

    Returns
    -------
    Dict[int, DiGraph], Dict[int, ndarray]
        the first return value is a dictionary mapping years to the respective keyword co-occurrence graph,
        while the second return value is a dictionary mapping years to the respective keywords-topics matrix.
    """
    graphs_per_year:Dict[int, nx.DiGraph] = dict()
    topics_per_year:Dict[int, np.ndarray]  = dict()
    topic_directory_absolute = os.path.abspath(topic_directory)

    print("opening directory "+topic_directory +".\n loading topics ...")
    for name in tqdm.tqdm(os.listdir(topic_directory_absolute)):
        year_directory_abs = os.path.join(topic_directory_absolute, name)
        if os.path.isdir(year_directory_abs):
            if name.isdigit():
                year = int(name)
                graph_filename = os.path.join(year_directory_abs, "graph.pickle")
                topic_filename = os.path.join(year_directory_abs, "topics.npy")
                graphs_per_year[year] = nx.read_gpickle(graph_filename)
                topics_per_year[year] = np.load(topic_filename)
    print("topics loaded.")
    return graphs_per_year, topics_per_year

def store_topics(topic_directory:str, fused_topics: List[Dict[str, int]]):
    """
    Store the time-fused topics generated during task 2.

    Parameters
    ----------
    topic_directory : str
        directory in which the fused topics will be stored.
    fused_topics : list[dict(str,int)]
        list containing the time-fused topics, which are represented through a
        dictionary (the keys are the keywords and the values their frequencies).
    """
    # create directory for chains
    fused_topics_directory = os.path.join(topic_directory, "fused_topics")
    if os.path.exists(fused_topics_directory):
        import shutil
        shutil.rmtree(fused_topics_directory)
    os.makedirs(fused_topics_directory)
    
    # store time-fused topics
    print("storing fused-topics ...")
    for i, topic in enumerate(tqdm.tqdm(fused_topics)):
        from wordcloud import WordCloud
        wcloud = WordCloud(background_color="white")
        wcloud.generate_from_frequencies(topic)
        
        # store
        fname = os.path.join(fused_topics_directory, str(i)+".png")
        plot.imshow(wcloud, interpolation="bilinear")
        plot.axis("off")
        plot.savefig(fname, dpi=300)

    # store
    fname = os.path.join(fused_topics_directory, "time-fused-topics.txt")
    with open(fname, "w") as f:
        i = 0
        for topic in fused_topics:
            for w, c in topic.items():
                f.write("T"+str(i)+"\t"+w+"\t"+str(c)+"\n")
            i += 1

def store_chains(topic_directory:str, topic_chains:list, gperyear:dict, tperyear:dict):
    """
    Store the topic-chains in 'topic_chain' inside the topic directory 'topic_directory'.

    Parameters
    ----------
    topic_directory : str
        directory in which the topic-chains will be stored.
    topic_chains : List[List[(int,int)]]
        topic-chains, represented as a sequence of lists containing pairs of (year, topic-id).
    gperyear : Dict[int,DiGraph]
        dictionary mapping years to the respective keyword co-occurrence graph.
    tperyear : Dict[int,ndarray]
        dictionary mapping years to the respective keywords-topics matrix.
    """
    chains_directory = os.path.join(topic_directory, "chains")
    if os.path.exists(chains_directory):
        import shutil
        shutil.rmtree(chains_directory)
    os.makedirs(chains_directory)

    # store chains
    print("storing chains ...")
    for i,p in enumerate(tqdm.tqdm(topic_chains)):
        # store heaviest chain
        chain_directory = os.path.join(chains_directory, "chain_"+str(i))
        os.makedirs(chain_directory)
        for u in p:
            y, ti = u
            g, t = gperyear[y], tperyear[y][:,ti]
            f = os.path.join(chain_directory, str(y)+".png")
            visualization.create_topic_picture_gradient(g=g, topic=t, file=f)

#========================================================================================

def task2(
    input_folder:str, 
    **kwargs):
    """
    Execute Task-2

    Parameters
    ----------
    input_folder : str
        folder that contains the topics generated during task-1.
    """

    # get parameters
    topic_score = "chamfer" if "topic_score" not in kwargs else kwargs["topic_score"]
    word_similarity = "jaccard" if "word_similarity" not in kwargs else kwargs["word_similarity"]
    visualize = False if "visualize" not in kwargs else kwargs["visualize"]
    chain_number = 20 if "chain_number" not in kwargs else kwargs["chain_number"]
    load_dag = False if "load_dag" not in kwargs else kwargs["load_dag"]


    # load stored topics
    input_directory = os.path.abspath(input_folder)
    topic_directory = os.path.join(input_directory, "topics")
    gperyear, tperyear = load_topics(topic_directory)
    years:List[int] = [year for year in tperyear.keys()]

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

    # create or load dag into memory
    dag_file = os.path.join(topic_directory,"topic-DAG.pickle")
    if load_dag:
        topic_DAG = nx.read_gpickle(dag_file)
    else:
        # add nodes to graph 
        print("creating topic DAG:\nadding nodes to the topic DAG ...")
        topic_DAG = nx.DiGraph()
        for i in range(len(years)):
            y = years[i]
            g, t = gperyear[y], tperyear[y]
            topic_DAG.add_nodes_from(_create_DAG_nodes(g, t, y))

        # add leyers to the DAG 
        print("adding edges to the topic DAG ...")
        bar = tqdm.tqdm(total=len(years)*(len(years)-1)/2)
        for i in range(len(years)-1):
            for j in range(i+1, len(years)):
                bar.update()
                # get years
                y1, y2 = years[i], years[j]
                # load graphs
                g1, g2 = gperyear[y1], gperyear[y2]
                # load topics
                T1, T2 = tperyear[y1], tperyear[y2]
                mapping = create_topic_links(
                    g1,g2,T1,T2,
                    mapping_method=topic_score,
                    word_sim_method=word_similarity)
                # add nodes and edges to the DAG
                topic_DAG.add_edges_from(_create_DAG_edges(mapping, y1, y2))
        nx.write_gpickle(topic_DAG, dag_file) # write DAG into file

    # show the topic DAG ------------------------------------------------------
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

    # compute and store chains 
    paths = chains_extraction(topic_DAG,k=chain_number)
    store_chains(
        topic_directory=input_directory,
        topic_chains=paths,
        gperyear=gperyear,
        tperyear=tperyear)

    # create time-fused topics
    fused_topics = []
    from collections import Counter
    for p in paths:
        merged_keywords = Counter()
        for u in p:
            words = topic_DAG.nodes[u]["topic"]
            for w in words:
                merged_keywords[w] += 1
        fused_topics.append(merged_keywords)
    
    # store topics
    store_topics(input_directory, fused_topics)


#------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Execute Task-2 of the Web&Social Information Extraction Project. It computes the time-fused topics and stores them in the input directory.')
    parser.add_argument('indir',
        metavar='<input-directory>', 
        type=str,  
        help="directory containing the topics generated by task-1.")
    args = parser.parse_args()
    task2(input_folder=args.indir)