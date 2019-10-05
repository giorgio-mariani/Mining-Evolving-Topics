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
def estimate_topic_influence(
    g:nx.DiGraph, topicness:np.ndarray, node_influence:np.ndarray, source_limit:int=3):
    
    # estimate topic sources
    topic_influence = np.power(topic_influence, 0.5)
    return topic_influence

def defuzzification(topic:np.ndarray, k=6):
    nzind = topic.nonzero()[0]
    fuzzy_list = [(topic[ui],ui) for ui in nzind]
    fuzzy_list.sort()
    tmp = fuzzy_list[-min(len(fuzzy_list), k):]
    defuzzed_topic = np.zeros(topic.size)
    for _,ui in tmp:
        defuzzed_topic[ui] = 1
    return defuzzed_topic

def find_topics_topicness(
    g:nx.Graph, k:int,
    fuzzy_output=False,
    fuzzy_deformation=False,
    visualize=False) -> np.ndarray:
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
        if not fuzzy_output:
            topics_list.append(crisp_topic)
        else:
            topics_list.append(fuzzy_topic)

        # update surface
        if not fuzzy_deformation:
            surface = np.maximum(surface - crisp_topic, 0)
        else:
            surface = surface*(1-fuzzy_topic)

    # create permutation for nodes (ordered by their influence)
    tmp = [(-t.sum(),i) for i,t in enumerate(topics_list)]
    tmp.sort()
    P = [i for _,i in tmp]

    # fill node-topic matrix
    topic_number = len(topics_list)
    T = np.zeros([n, topic_number], dtype=float) # node-topic matrix 
    for i in range(topic_number):
        T[:,i] = topics_list[P[i]]

    # visualize various information about the estiamted topics
    if visualize:
        coverage = T.sum(axis=1)
        sources = np.zeros([n])
        sources[T.argmax(axis=0)] = 1

        visualization.plot_edge_weights(g)
        
        '''
        # REMOVEME
        alpha=0.8
        pr = misc.dictionary_to_numpy(nx.pagerank_numpy(g,alpha=alpha))
        ppr = misc.dictionary_to_numpy(nx.pagerank_numpy(g,alpha=alpha,personalization=v))
        pos = nx.drawing.spring_layout(g)
        visualization.show_graphfunction(g, pr, with_labels=False,title="Pagerank",pos=pos, savefile="pr.png")
        visualization.show_graphfunction(g, ppr, with_labels=False,title="Personalized Pagerank",pos=pos,savefile="ppr.png")
        visualization.show_graphfunction(g, betweenness, with_labels=False,title="Betwenness Centrality",pos=pos,savefile="bc.png")
        visualization.show_graphfunction(g, topicness, with_labels=False,title="Topicness",pos=pos,savefile="top.png")
        '''

        visualization.show_graphfunction(g, topicness, with_labels=False)
        visualization.show_graphfunction(g, sources, with_labels=False)
        visualization.show_graphfunction(g, coverage, with_labels=False)
    return T

#=========================================================================================
def store_topics(g:nx.DiGraph, topics:np.ndarray, output_directory:str="topics", savefig=True, gradient_topic=True):
    """
    This function stores the topics described by the array 'topics' inside the directory 'output_directory'.
    Note that 'topics' must be an array with shape: [vertices number, topics number].
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

def task1(start, end, parent_directory="topics"):
    """
    This function creates an output directory containing the computed topics (foreach year). 
    This directory is named using the timestamp of when it is executed, and is placed inside 
    'parent_directory' (which must already exists).
    """
    # load graph data into memory
    GraphsPerYear = ps.parseKeyword(start, end)

    # create output directory
    datestr = datetime.datetime.now().isoformat().replace(":","_")
    output_directory = os.path.join(parent_directory, datestr)    #TODO check for existance of output directory
    os.mkdir(output_directory) # NOTE assuming this directory does not already exists, since it depends on the time

    # main loop for topic extraction, iterating over the years for topic extraction
    for year in range(start, end+1):

        # compute topics for given year
        g:nx.DiGraph = GraphsPerYear[year]
        print("\extracting topics for year "+str(year)+" ...")
        print("keywords count: "+str(g.number_of_nodes()))
        topics = find_topics_topicness(g, k=60, visualize=False)
        topic_count = topics.shape[1]
        print("topics extracted.\nnumber of topics: "+str(topic_count))

        # store resulting topics
        print("storing topics ...")
        year_dir = os.path.join(output_directory, str(year))
        os.mkdir(year_dir)
        store_topics(g, topics, output_directory=year_dir, savefig=True)
        print("storing complete.")
    
# if this module is invoked, then solve task 1
if __name__ == "__main__":
    task1(2010,2010)