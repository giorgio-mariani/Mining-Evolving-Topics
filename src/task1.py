import os
import datetime

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

def store_topics(g:nx.DiGraph, topics:np.ndarray, output_directory:str="topics"):
    """
    This function stores the topics described by the array 'topics' inside the directory 'output_directory'.
    Note that 'topics' must be an array with shape: [vertices number, topics number].
    """
    vertex_count = g.number_of_nodes()
    topic_count = topics.shape[1]

    # save vertex-topic matrix
    topics_file = os.path.join(output_directory,"np_array")
    np.save(topics_file, topics)

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
            keywords = [g.nodes[ui]["name"]+"\n" for ui in range(vertex_count) if vertices[ui] != 0]
            f.writelines(keywords)
        
        topic_set = vertices.nonzero()[0] # set of vertices in the topic, represented through an indicator-vector
        visualization.create_topic_picture(g, topic_set ,img_filename)


# find topics algorithm -------------------------------------------------------
def find_topics_divergence(g:nx.Graph, threshold=0.6):
    """
    This function estimates the topics inside the input graph 'g' using the 
    linear-threshold spread-of-influence algorithm; for each node 'u', the probability
    of infecting every other node is estimated (by exploiting the linear-threshold 
    algorithm), then all these probability distribution are compared, using the jensen-shannon
    divergence; nodes that infects similar groups of nodes are then joined by an edge, and by using
    clique percolation 
    """
    vertex_count:int = g.number_of_nodes()
    P:np.ndarray = np.zeros([vertex_count, vertex_count])
    influence.normalize_weights(g,mode="mixed")
    for ui in tqdm.trange(vertex_count):
        tmp = influence.linear_threshold_mean(g, {ui}, 12)
        P[ui,:] = tmp/tmp.sum()
    
    D:np.ndarray = np.ones([vertex_count,vertex_count])
    with tqdm.tqdm(total=vertex_count) as bar:
        for cc in nx.connected_components(g.to_undirected(as_view=True)):
            for ui in cc:
                bar.update()
                for vi in cc:
                    D[ui,vi] = misc.jensen_shannon_divergence(P[ui,:],P[vi,:])
    #visualization.KernelDensityEstimation(D.flatten())
    
    g_cliques:nx.Graph = nx.from_numpy_array(D<=threshold, create_using=nx.Graph)
    #visualization.graph_comparison(g,g_cliques.to_directed(True), cmap="coolwarm")

    # find topics using clique perculation
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

    #visualization.show_graphfunction(g, 1+topics.sum(axis=1), cmap="autumn")
    return topics

#------------------------------------------------------------------------------
def task1(parent_directory="topics"):
    """
    This function creates an output directory containing the computed topics (foreach year). 
    This directory is named using the timestamp of when it is executed, and is placed inside 
    'parent_directory' (which must already exists).
    """
    G = ps.parseKeyword()
    datestr = datetime.datetime.now().isoformat().replace(":","_")
    output_directory = os.path.join(parent_directory, datestr)
    os.mkdir(output_directory) # NOTE assuming this directory does not already exists, since it depends on the time
    for year in range(2000,2019):
        g:nx.DiGraph = G[year]
        print("\ncomputing topics for year "+str(year)+" ...")
        print("graph's vertex count: "+str(g.number_of_nodes()))

        # compute topics for given year
        print("extracting candidate topics ...")
        topics = find_topics_divergence(g)
        #old_topic_count = topics.shape[1]
        #print("candidate extracted.\nmerging into topics ...")
        #topics = merge_topics(topics, threshold=0.85)
        topic_count = topics.shape[1]
        #print("merged candidates: "+str(old_topic_count-topic_count))
        print("number of topics: "+str(topic_count))
        year_dir = os.path.join(output_directory, str(year))

        # store resulting topics
        print("storing topics ...")
        os.mkdir(year_dir)
        store_topics(g, topics, out_dir=year_dir)
        print("storing complete.")

# if this module is invoked, then solve task 1
if __name__ == "__main__":
    task1()


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

def find_topics_topicness(g:nx.Graph) -> np.ndarray:
    pagerank_dict:dict = nx.pagerank_numpy(g)
    pagerank = np.zeros(len(pagerank_dict))
    for i,r in pagerank_dict.items():
        pagerank[i] = r

    betweenness_dict:dict = nx.betweenness_centrality(g.to_undirected(as_view=True))
    betweenness = np.zeros(len(pagerank_dict))
    for i,r in betweenness_dict.items():
        betweenness[i] = r
    
    a = 0.5
    pagerank = pagerank/pagerank.max()
    betweenness = 1 - betweenness/betweenness.max()
    topicness = a*pagerank + (1-a)*betweenness
    topicness = topicness - topicness.min()
    topicness =  topicness/(topicness.max())
    visualization.show_graphfunction(g, topicness)
    
    '''
    maxima = misc.local_maxima(g, topicness)
    maxima_compact = np.flatnonzero(maxima)
    maxima_number = len(maxima_compact)

    values = np.zeros([maxima_number, g.vcount()])

    for i in range(maxima_number):
        mi = maxima_compact[i]
        random_color = colors[np.random.randint(1, 12), :]
        vertices_color[mi] = random_color
        tmp = influence.linear_threshold_mean(g,{mi}, 15)
        values[i,:] = tmp

    for i in range(g.vcount()):
        color = np.zeros([3])
        for j,mj in enumerate(maxima_compact):
            tmp = vertices_color[mj]*np.sqrt(values[j, i])
            color = np.maximum(tmp, color)
        vertices_color[i] = color'''