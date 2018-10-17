import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gem.utils import graph_util

def edgelist2tsv(edgelist, filename):
    """
    Takes an list of edges and writes it to a tsv file
    """
    with open(filename, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for edge in edgelist:
            writer.writerow([edge[0], edge[1]])

def tsv2edgelist(file, isDirected=True):
    """
    load an edgelist from a tsv file for use in the GEM package
    """
    G = (graph_util
         .loadGraphFromEdgeListTxt('../data/Nedgelist.tsv',
                                   directed=isDirected))
    if isDirected:
        return G.to_directed()
    else:
        return G



def poincare_viz(vectors, sc_ind, ncls, colors, title, filename):
    """
    Plots an array of vectors
    sc_ind: list indicating where in the array the subgenre of the books changes
    ncls: the number of genres + subgenres
    colors: list of colors associated with the subgenres
    """
    fig = plt.figure()

    if len(vectors.T) > 2:
        model = TSNE(n_components=2)
        vectors = model.fit_transform(vectors)

    X = vectors.T[0]
    Y = vectors.T[1]

    plt.title(title)

    for i in range(len(sc_ind)-1):
        plt.scatter(X[ncls+sc_ind[i]:ncls+sc_ind[i+1]],
                    Y[ncls+sc_ind[i]:ncls+sc_ind[i+1]],
                    c=colors[i],
                    marker='.')
    plt.scatter(X[:ncls],
                Y[:ncls],
                c='k',
                marker='.')

    plt.savefig(filename)
