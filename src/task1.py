import os
import datetime
import heapq

import networkx as nx
import networkx.algorithms.cluster as cluster
import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plot
import tqdm

# costum modules
import influence
import misc
import parse_data as ps
import visualization

# find topics algorithm ============================================================
def defuzzification(topic:np.ndarray, k:int=6)->np.ndarray:
    """
    Return a vector representing a defuzzification of a fuzzy vector.

    parameters
    ----------
    topic: ndarra
        vector with size [n] containing the membership of each node w.r.t. a topic.
        Hence, for each node ui, topic[ui] in [0,1].
    k : int 
        maximum number of keywords that the defuzzified version of 'topic' must have.

    Returns
    -------
    ndarray
        A vector representing the indicator function of the deffuzzified input vector.
    """
    nzind = topic.nonzero()[0]
    fuzzy_list = [(topic[ui],ui) for ui in nzind]
    fuzzy_list.sort()
    tmp = fuzzy_list[-min(len(fuzzy_list), k):]
    defuzzed_topic = np.zeros(topic.size)
    for _,ui in tmp:
        defuzzed_topic[ui] = 1
    return defuzzed_topic

def find_topics_topicness(
    g:nx.DiGraph,
    visualize=False) -> np.ndarray:
    """
    Computes a keyword-topic matrix containing memebership information of keywords w.r.t topics.

    
    Extended SUmmuray
    -----------------
    The computation of this matrix uses an iterative algorithm, which picks best nodes using a 'topicness'
    metric, derived from pagerank, local cluster coefficient, and betweenness centrality.

    Parameters
    ----------
    g : nx.DiGraph
        keyword co-occurrence graph

    Returns
    -------
    ndarray
        keyword-topic matrix, containing the information about keywords and topics. 
        specifically for the entry [ui,ti] if contains 1 if the topic is in 
    """

    n = g.number_of_nodes()
    m = g.number_of_edges()

    # get betweenness centrality
    betweenness = misc.dictionary_to_numpy(
        nx.betweenness_centrality(g.to_undirected(as_view=True)))

    # get personalized-pagerank using cluster coefficient
    c = cluster.clustering(g, weight="weight")
    z = sum(c.values())
    if z != 0:
        v = {ui:c[ui]/z for ui in range(n)}
        pagerank = misc.dictionary_to_numpy(nx.pagerank_numpy(g, personalization=v))
    else:
        pagerank = misc.dictionary_to_numpy(nx.pagerank_numpy(g))
    
    # compute topicness --------------------------
    pagerank:np.ndarray = pagerank/pagerank.max() # normalize between [0,1]
    betweenness = betweenness/betweenness.max() if betweenness.max() != 0 else betweenness # normalize between [0,1]
    topicness:np.ndarray = pagerank*(1-betweenness)

    # normalize topicness between [0,1]
    topicness = topicness - topicness.min()
    topicness =  topicness/(topicness.max())

    # normalize the edge weights
    influence.normalize_weights(g, mode="mixed")

    # compute area of influence for each node
    node_influence = np.zeros([n,n], dtype=float)
    for u in tqdm.trange(n):
        node_influence[u,:] = influence.linear_threshold_mean(g, {u}, 15)

    # estiamte topics
    topics_list, surface = list(), np.array(topicness)
    #for i in tqdm.trange(k):
    while (surface > 0).any():
        # compute topic influence
        fuzzy_topic = node_influence[surface.argmax(), :]
        crisp_topic = defuzzification(fuzzy_topic)

        # add topic to extracted topics
        topics_list.append(crisp_topic)

        # update surface
        surface = surface*(1-fuzzy_topic)

    # fill node-topic matrix
    topic_number = len(topics_list)
    T = np.zeros([n, topic_number], dtype=float) # node-topic matrix 
    for i in range(topic_number):
        T[:,i] = topics_list[i]

    # visualize various information about the estiamted topics
    if visualize:
        coverage = T.sum(axis=1)
        sources = np.zeros([n])
        sources[T.argmax(axis=0)] = 1

        visualization.plot_edge_weights(g)
        visualization.show_graphfunction(g, topicness, with_labels=False)
        visualization.show_graphfunction(g, sources, with_labels=False)
        visualization.show_graphfunction(g, coverage, with_labels=False)
    return T

#=========================================================================================
def store_topics(g:nx.DiGraph, topics:np.ndarray, output_directory:str, savefig=True, gradient_topic=True):
    """
    Store topics passed in input, possibly with images representing the topics.

    Extended Summary
    ----------------
    This function stores the topics described by the array 'topics' inside the directory 'output_directory'.
    Note that 'topics' must be an array with shape: [keywords number, topics number].

    Parameters
    ----------
    g : DiGraph
        keywords co-occurrence graph.
    topics : ndarray
        keyword-topic matrix (hence, shape [keywords number, topics number]) describing the memebership of
        each kewords w.r.t. a topic.
    output_directory : str
        Directory in which the topics will be stored.
    savefig : bool
        Whether the image generated from the topic sould also be stored, set to falso for better storing.
     """
    vertex_count = g.number_of_nodes()
    topic_count = topics.shape[1]

    # save vertex-topic matrix
    topics_file = os.path.join(output_directory,"topics.npy")
    np.save(topics_file, topics)
    graph_filename = os.path.join(output_directory,"graph.pickle")
    nx.write_gpickle(g, graph_filename)

    if not savefig:
        return

    # save info about each topic
    for ti in tqdm.trange(topic_count):
        vertices = topics[:,ti]
        topic_directory = os.path.join(
            output_directory,"topic_"+str(ti)+"-"+str((vertices!=0).sum())+"_elements")
        keywords_filename = os.path.join(topic_directory,"keywords.txt")
        img_filename = os.path.join(topic_directory,"img.png")
        os.mkdir(topic_directory) # NOTE assuming this directory does not already exists

        # write keywords
        with open(keywords_filename,"w", encoding='utf-8') as f:
            if gradient_topic:
                keywords = [g.nodes[ui]["name"]+":"+str(vertices[ui])+"\n" for ui in range(vertex_count) if vertices[ui] != 0]
                visualization.create_topic_picture_gradient(g, vertices, img_filename)
            else:
                keywords = [g.nodes[ui]["name"]+"\n" for ui in range(vertex_count) if vertices[ui] != 0]
                topic_set = vertices.nonzero()[0] # set of vertices in the topic, represented through an indicator-vector
                visualization.show_topics(g, topic_set, img_filename)
            f.writelines(keywords)

def task1(start:int, end:int, output_directory:str="topics"):
    """
    Execute Task 1 of the project.

    Extended Summary
    ----------------
    This function creates an output directory containing the topics computed for each year. 

    Parameters
    ----------
    start : int
        first year from which estimate topics.
    end : int
        last year in which estimate topics.
    output_directory : str
        output directory to be created. This directory MUST NOT exist.
    """
    # load graph data into memory
    GraphsPerYear = ps.parseKeyword(start, end)

    # create output directory
    #datestr = datetime.datetime.now().isoformat().replace(":","_")
    #output_directory = os.path.join(parent_directory, datestr)
    if os.path.exists(output_directory):
        print("Error: directory "+str(output_directory)+ " already exists!")
        return

    topics_directory = os.path.join(output_directory, "topics")
    os.mkdir(output_directory)
    os.mkdir(topics_directory)

    # main loop for topic extraction, iterating over the years for topic extraction
    for year in range(start, end+1):

        # compute topics for given year
        g:nx.DiGraph = GraphsPerYear[year]
        print("\extracting topics for year "+str(year)+" ...")
        print("keywords count: "+str(g.number_of_nodes()))
        topics = find_topics_topicness(g, visualize=False)
        topic_count = topics.shape[1]
        print("topics extracted.\nnumber of topics: "+str(topic_count))

        # store resulting topics
        print("storing topics ...")
        year_dir = os.path.join(topics_directory, str(year))
        os.mkdir(year_dir)
        store_topics(g, topics, output_directory=year_dir, savefig=True)
        print("storing complete.")
    
# if this module is invoked, then solve task 1
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Execute Task-1 of the Web&Social Information Extraction Project.')
    parser.add_argument('outdir',  
        metavar='<output-directory>', 
        type=str,  
        help="directory that will contain the output directory generated by the task. This directory MUST NOT exist.")
    parser.add_argument('-s','--start', 
        metavar='<start year>', 
        type=int, 
        required=True, 
        help="starting year for tracking, must be >= 2000 and <= <end year>")
    parser.add_argument('-e','--end', 
        metavar='<end year>', 
        type=int,
        required=True,
        help="ending year for tracking, must be >= <start year> and <= 2018")
    args = parser.parse_args()
    task1(args.start, args.end, args.outdir)