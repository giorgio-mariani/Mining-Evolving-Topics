import os
import datetime
import heapq

import networkx as nx
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

def find_topics_topicness(g:nx.Graph, k:int) -> np.ndarray:
    n = g.number_of_nodes()
    m = g.number_of_edges()

    pagerank_dict:dict = nx.pagerank_numpy(g)
    pagerank = np.zeros(len(pagerank_dict))
    for i,r in pagerank_dict.items():
        pagerank[i] = r

    betweenness_dict:dict = nx.betweenness_centrality(g.to_undirected(as_view=True))
    betweenness:np.ndarray = np.zeros(len(pagerank_dict))
    for i,r in betweenness_dict.items():
        betweenness[i] = r
    
    # compute topicness
    a = 0.5
    pagerank:np.ndarray = pagerank/pagerank.max() # normalize between [0,1]
    betweenness = 1 - betweenness/betweenness.max() # normalize between [0,1]
    topicness:np.ndarray = pagerank*(betweenness*betweenness*betweenness) # a*pagerank + (1-a)*betweenness

    # normalize topicness between [0,1]
    topicness = topicness - topicness.min()
    topicness =  topicness/(topicness.max())

    #visualization.show_graphfunction(g, topicness)

    # compute area of influence for each node
    influence.normalize_weights(g, mode="mixed") 
    node_influence = np.zeros([n,n], dtype=float)
    for u in tqdm.trange(n):
        node_influence[u,:] = influence.linear_threshold_mean(g, {u}, 15)

    # compute topics
    topicness2 = np.array(topicness)
    topic_sources = []
    source_limit = 1
    for _ in tqdm.trange(k):

        # estimate topic sources
        argmax = topicness2.argmax()
        sources = [(topicness2[argmax],argmax)] 
        heapq.heapify(sources)
        for u in g[argmax]:
            heapq.heappush(sources, (topicness2[u],u))
            if len(sources) > source_limit:
                heapq.heappop(sources)

        # compute average topic
        topic_influence = np.zeros([n])
        for _,s in sources:
            topic_influence += node_influence[s]
        topic_influence = topic_influence/len(sources)

        # update loop variables
        topic_sources.append(topic_influence)   # add topic to extracted topics
        topicness2 = np.maximum(topicness2 - topic_influence, 0)  # update topicness

    # compute node-topic matrix
    topic_number = len(topic_sources)
    topics = np.zeros([n,topic_number], dtype=float)
    for i in range(topic_number):
        topics[:,i] = topic_sources[i]
    return topics

# ------------------------------------------------------------------------------
'''
OLD STUFF

def find_topics_divergence(g:nx.Graph, threshold=0.6):
    """
    This function estimates the topics inside the input graph 'g' using the 
    linear-threshold spread-of-influence algorithm; for each node 'u', the probability
    of infecting every other node is estimated (by exploiting the linear-threshold 
    algorithm), then all these probability distribution are compared, using the jensen-shannon
    divergence; nodes that infects similar groups of nodes are then joined by an edge, and by using
    clique percolation 
    """
    # initialize matrix of probabilities, in which P[u,v] describes the prob. that u infects node v
    vertex_count:int = g.number_of_nodes()
    P:np.ndarray = np.zeros([vertex_count, vertex_count])
    
    # preprocess the weights for the spread-of-influence algorithm
    influence.normalize_weights(g, mode="mixed") 

    for ui in tqdm.trange(vertex_count):
        tmp = influence.linear_threshold_mean(g, {ui}, 12)
        tmp = tmp/tmp.sum() # NOTE is this really wise?
        P[ui,:] = tmp

    # initialization and assignment of a matrix of distances between nodes
    D:np.ndarray = np.ones([vertex_count,vertex_count])
    with tqdm.tqdm(total=vertex_count) as bar:
        for cc in nx.connected_components(g.to_undirected(as_view=True)):
            for ui in cc:
                bar.update()
                for vi in cc:
                    #D[ui,vi] = 1 - misc.cosine_similarity(P[ui,:],P[vi,:]) #NOTE since all values are positive it is well defined as distance
                    #D[ui,vi] = misc.euclidean_distance(P[ui,:], P[vi,:])
                    D[ui,vi] = misc.jensen_shannon_divergence(P[ui,:],P[vi,:]) #NOTE requires input vector to be prob. distributions
    
    # show distribution of distances
    visualization.KernelDensityEstimation(D.flatten())

    #topics = topics_from_distances_CPM(D)
    topics = topics_from_distances_GN(D)

    #visualization.show_graphfunction(g, 1+topics.sum(axis=1), cmap="autumn")
    return topics

def topics_from_distances_CPM(D:np.ndarray, threshold:float=0.6) -> np.ndarray:
    vertex_count = D.size
    g_cliques:nx.Graph = nx.from_numpy_array(D<=threshold, create_using=nx.Graph)
    visualization.graph_comparison(g, g_cliques.to_directed(True), cmap="coolwarm")

    topic_list = []
    for k in range(3, 7):
        topic_list = topic_list + list(nx.algorithms.community.k_clique_communities(g_cliques,k))
    topic_count = len(topic_list)

    # TODO: merge topics that have high similarity (i.e. jaccard over 0.85)
    
    # create vertex-topic matrix
    topics = np.zeros([vertex_count, topic_count])
    for ti,tset in enumerate(topic_list):
        for ui in tset:
            topics[ui, ti] = 1
    return topics

def topics_from_distances_GN(D:np.ndarray, origin_graph) -> np.ndarray:
    vertex_count = D.size
    graph = nx.from_numpy_array(D<=0.5)
    visualization.graph_comparison(origin_graph, graph, cmap="coolwarm")

    def _edge_selector(g:nx.Graph):
        maximum = -1
        argmax = (None, None)
        for u,v in g.edges:
            if  D[u,v] >= maximum:
                maximum =  D[u,v]
                argmax = (u,v)
        return argmax

    communities = nx.algorithms.community.centrality.girvan_newman(graph)
    topic_list = [community for community in communities]
    topic_count = len(topic_list)
    
    # create vertex-topic matrix
    topics = np.zeros([vertex_count, topic_count])
    for ti,tset in enumerate(topic_list):
        for ui in tset:
            topics[ui, ti] = 1
    return topics
'''

#=========================================================================================
def store_topics(g:nx.DiGraph, topics:np.ndarray, output_directory:str="topics", gradient_topic=True):
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
        topics = find_topics_topicness(g, k=50) #find_topics_divergence(g) # actual topic estimation
        topic_count = topics.shape[1]
        print("topics extracted.\nnumber of topics: "+str(topic_count))

        # store resulting topics
        print("storing topics ...")
        year_dir = os.path.join(output_directory, str(year))
        os.mkdir(year_dir)
        store_topics(g, topics, output_directory=year_dir)
        print("storing complete.")

# if this module is invoked, then solve task 1
if __name__ == "__main__":
    task1(2015,2018)


###############################################################################
#TODO complete these functions
#------------------------------------------------------------------------------
def merge_topics(topics:np.ndarray, threshold=0.85, show=False) -> np.ndarray:
    vertex_count, topic_count = topics.shape
    jaccard = np.zeros([topic_count, topic_count], dtype=float)
    for ti in range(topic_count):
        for tj in range(topic_count):
            topic1 = topics[:,ti]
            topic2 = topics[:,tj]
            jaccard[ti,tj] = np.dot(topic1,topic2)/np.maximum(topic1,topic2).sum()

    T = range(topic_count)
    converged = False
    while not converged:
        
        #converged = True
        new_topic_count = len(T)
        new_topics = np.zeros([vertex_count, new_topic_count])
        for ti in range(new_topic_count):
            new_topics[:,ti] = T[ti]

    # show the distribution of the jaccard similarities
    if show:
        visualization.histogram(jaccard.flatten(), bins=int(topic_count))
    
    return new_topics