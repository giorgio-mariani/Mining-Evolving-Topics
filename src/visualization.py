import collections as cl

import matplotlib.pyplot as plot
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import scipy.stats as stats



def plot_edge_weights(g):
    m = g.number_of_edges()
    n = g.number_of_nodes()
    def nsum(ui):
        w = 0
        for vi in g[ui]:
            w += g[vi][ui]["weight"]
        return w

    weights = [w for u,v,w in g.edges.data("weight")]
    weights.sort()

    neighboursums = [nsum(ui) for ui in g.nodes]
    neighboursums.sort()

    plot.subplot(1, 2, 1)
    plot.scatter(range(m),weights, s=1.5*1.5)
    plot.xlabel("edges")
    plot.ylabel("weight")
    plot.title("Edge Weights")

    plot.subplot(1, 2, 2)
    plot.xlabel("nodes")
    plot.ylabel("in-neighbor sum")
    plot.title("Weighted In-neighbors Sums")
    plot.scatter(range(n),neighboursums,s=1.5*1.5)
    plot.tight_layout()
    plot.show()


## show graphs ----------------------------------------------------------------
def show_graphfunction(
    g:nx.Graph,
    f:np.ndarray=None,
    nodecolors:dict=None,
    with_labels:bool=True,
    cmap:str="viridis",
    show_figure=True):

    vertex_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    f = np.ones(vertex_count) if f is None else f
    scale = (f.max()-f.min())
    scale = 1 if scale == 0 else scale
    f = (f - f.min())/scale #normalize between 0 and 1
    vertices_color = f

    if nodecolors is not None:
        for k,v in nodecolors.items():
            vertices_color[k] = v

    edge_weight = np.array([w for u,v,w in g.edges.data(data="weight")])
    scale = (edge_weight.max()-edge_weight.min())
    scale = 1 if scale==0 else scale
    edge_weight = (edge_weight - edge_weight.min())/scale
    labels = {ui:n for ui,n in g.nodes.data("name")}
    nx.draw_networkx(
        g, 
        arrows=False,
        node_color=vertices_color,
        node_size=40,
        node_cmap=cm.get_cmap(cmap),
        edge_color=edge_weight,
        edge_cmap=cm.get_cmap("binary"),
        linewidths=0,
        width=0.15,
        font_size=5,
        font_weight='ultralight',
        with_labels=with_labels,
        labels=labels)
    
    if show_figure:
        plot.show()

def graph_comparison(g1:nx.DiGraph, g2:nx.DiGraph, with_labels=False, cmap="tab10"):
    edge_color = {(ui,vi):-1 for ui,vi in g1.edges()}
    for ui,vi in g2.edges():
        if (ui,vi) in edge_color:
                edge_color[(ui,vi)] = 0
        else:
                edge_color[(ui,vi)] = 1
    g = nx.Graph(g1)
    g.add_edges_from(g2.edges())
    nx.set_edge_attributes(g,edge_color,"color")
    edge_color = [c for ui,vi,c in g.edges(data="color")]
    nx.draw_networkx(
        g, 
        arrows=False,
        node_size=40,
        edge_color=edge_color,
        edge_cmap=cm.get_cmap(cmap),
        linewidths=0,
        width=0.25,
        with_labels=with_labels)
    plot.show()

def create_topic_picture(g:nx.DiGraph, topic:set, file:str, hops=2):
    g = g.to_undirected(as_view=True)
    F = cl.deque(topic)
    N = set(topic)
    counter = 0
    while F and counter <= hops:
        v:int = F.popleft()
        counter += 1
        for u in g.neighbors(v):
            if u not in N:
                N.add(u)
                F.append(u)

    small_g = nx.Graph(g.subgraph(N))
    for ui in small_g:
        small_g.nodes[ui]["label"] = g.nodes[ui]["name"]
        del small_g.nodes[ui]["name"] # necessary in order to use pydot
        if ui in topic:
            small_g.nodes[ui]["style"]="filled"
            small_g.nodes[ui]["fillcolor"]="darkolivegreen3"
        else:
            small_g.nodes[ui]["style"]="filled"
            small_g.nodes[ui]["fillcolor"]="gray77"

    dot = nx.nx_pydot.to_pydot(small_g)
    dot.write(file, prog="dot",format="png")

def create_topic_picture_gradient(g:nx.DiGraph, topic:np.ndarray, file:str):
    g = g.to_undirected(as_view=True)
    topic = np.power(topic,1)
    topic = topic/topic.max()
    N = {ui:topic[ui] for ui in range(len(topic)) if topic[ui]>0}
    small_g = nx.Graph(g.subgraph(N))
    
    def rgb2hex(r,g,b):
        return "#{:02x}{:02x}{:02x}".format(r,g,b)

    for ui in small_g:
        small_g.nodes[ui]["label"] = g.nodes[ui]["name"]
        del small_g.nodes[ui]["name"] # necessary in order to use pydot
        small_g.nodes[ui]["style"]="filled"
        small_g.nodes[ui]["fillcolor"]= rgb2hex(int(topic[ui]*255),0,int(255-topic[ui]*255))

    dot = nx.nx_pydot.to_pydot(small_g)
    dot.write(file, prog="dot",format="png")
    
## show plots -----------------------------------------------------------------
def histogram(p:np.ndarray, bins):
    plot.hist(p,bins)
    plot.show()

def KernelDensityEstimation(x:np.ndarray, n=1000):
    kernel = stats.gaussian_kde(x)
    xmin = x.min()
    xmax = x.max()
    X =  np.arange(start=xmin, stop=xmax, step=(xmax-xmin)/n)
    Y = kernel(X)

    ax:plot.Axes
    ifg:plot.Figure
    fig, ax = plot.subplots()
    ax.plot(X, Y, 'k.', markersize=2)
    plot.show()


def show_word_similarity(g1:nx.DiGraph, g2:nx.DiGraph, wordsimilarity:np.ndarray):
    fig = plot.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(wordsimilarity)
    
    x_labels = [l for u,l in g1.nodes(data="name")]
    y_labels = [l for u,l in g2.nodes(data="name")]
    plot.show()

#------------------------------------------------------------------------------