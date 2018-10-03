import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import csv
from time import time

from gensim.models.poincare import PoincareModel
from gensim.models.poincare import ReconstructionEvaluation

GEM_PATH = '/Users/christopherwmurphy/GEM'
import sys
sys.path.append(GEM_PATH)
from gem.utils      import graph_util
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.embedding.gf       import GraphFactorization
from gem.embedding.lap      import LaplacianEigenmaps

df = (pd
      .read_csv('../data/loc_literature_reduced.csv')
      .drop_duplicates(subset='title')
      .reset_index(drop=True))

def subclass_indices(df):
    subclass_counts = (df['subclass']
                       .value_counts()
                       .reset_index()
                       .sort_values(by='index'))
    sc_ind = [np.sum(subclass_counts['subclass'].values[:n])
              for n in range(len(subclass_counts)+1)]
    return sc_ind

sc_ind = subclass_indices(df)

def df_recs_tfidf(df):
    """
    Takes a DataFrame of book titles, DDC, and descriptions.
    Computes Tfidf vectors based on book descriptions.
    Returns a DataFrame of similiarity scores for the books
    based on the inner product of their vectors.
    """

    Tfidf_vec = TfidfVectorizer(stop_words = 'english')

    X = (Tfidf_vec
         .fit_transform(df['description'])
         .todense())

    simil = [(i,
              j,
              np.sum(X[i] * X[j].T))
             for i in range(len(X)) for j in range(i)]

    df_out = (pd
              .DataFrame(simil,
                         columns=['book i',
                                  'book j',
                                  'score']))

    # attempt to save memory by using smaller datatypes when possible
    converted_ints = (df_out
                      .select_dtypes(include=['int'])
                      .apply(pd.to_numeric, downcast='unsigned'))
    converted_floats = (df_out
                        .select_dtypes(include=['float'])
                        .apply(pd.to_numeric, downcast='float'))
    df_out[converted_ints.columns] = converted_ints
    df_out[converted_floats.columns] = converted_floats

    return df_out

book_sim = df_recs_tfidf(df)
minthres = 0.78
maxthres = 0.99
scorethres = ((book_sim['score'] > minthres) &
              (book_sim['score'] < maxthres))

def create_edgelist(file, df_b2c, df_b2b):
    # load edges from the LoC classes
    df1 = (pd
           .read_csv(file,
                     dtype='str'))

    # get edges from the books to the classes
    df2 = (df_b2c[['title',
                   'subclass']]
           .rename(columns={'title':'Edge_From',
                            'subclass':'Edge_To'})
           .sort_values(by='Edge_To'))

    # get edges from the books to the classes
    df3 = (df_b2b[scorethres][['book_i',
                               'book_j']]
           .rename(columns={'book_i':'Edge_From',
                            'book_j':'Edge_To'}))
    df4 = (df_b2b[scorethres][['book_j',
                               'book_i']]
           .rename(columns={'book_j':'Edge_From',
                            'book_i':'Edge_To'}))

    # combine the dfs
    df5 = (df1
           .append(df2,
                   ignore_index=True)
           .append(df3,
                   ignore_index=True)
           .append(df4,
                   ignore_index=True))

    # consistently assign categories
    df6 = (df5
           .stack()
           .astype('category')
           .unstack())

    # make the categorical values explicit for later convenience
    for name in df6.columns:
        df6['N' + name] = (df6[name]
                           .cat
                           .codes)

    return df5

edgelist = create_edgelist('../data/loc_class_edgelist.csv',
                           df,
                           book_sim)

def edgelist2tsv(edgelist, filename):
    with open(filename, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for edge in edgelist:
            writer.writerow([edge[0], edge[1]])

edgelist2tsv(edgelist[['Edge_From', 'Edge_To']]
             .values,
             '../data/edgelist.tsv')
edgelist2tsv(edgelist[['NEdge_From', 'NEdge_To']]
             .values,
             '../data/Nedgelist.tsv')

def poincare_viz(vectors, fignum, title, filename):
    fig = plt.figure(fignum)

    X = vectors.T[0]
    Y = vectors.T[1]

    plt.title(title)

    for i in range(len(sc_ind)-1):
        plt.scatter(X[28+sc_ind[i]:28+sc_ind[i+1]], Y[28+sc_ind[i]:28+sc_ind[i+1]])
    plt.scatter(X[:28], Y[:28], c='k')

    plt.savefig(filename)

def poincare_recon(G, file, dim, reg=0, epchs=300):
    lf = open(file, 'a')
    lf.write('Poincare Embbeding in d=%i' %dim)
    print('Poincare Embbeding in d=%i' %dim)

    embedding = PoincareModel(G,
                              size=dim,
                              regularization_coeff=reg)
    (embedding
     .train(epochs=epchs))

     if dim==2:
         poincare_viz(embedding.kv.vectors,
                      1,
                      'Poincare Embedding',
                      'poincare_d2.pdf)

    recon = ReconstructionEvaluation('../data/edgelist.tsv',
                                     embedding.kv)
    lf.write('Reconstruction MAP = %f' %recon.evaluate()['MAP'])
    print('Reconstruction MAP = %f' %recon.evaluate()['MAP'])
    lf.close()

def tsv2graph(file, isDirected = True):
    G = (graph_util
         .loadGraphFromEdgeListTxt('../data/Nedgelist.tsv',
                                   directed=isDirected))
    if isDirected:
        return G.to_directed()
    else:
        return G

G = tsv2graph('../data/Nedgelist.tsv')

def LE_recon(G, file, dim):
    lf = open(file, 'a')
    lf.write('Laplacian Eigenmaps in d=%i' %dim)
    print('Laplacian Eigenmaps in d=%i' %dim)

    embedding = LaplacianEigenmaps(d=dim)
    Y, t = (embedding
            .learn_embedding(graph=G,
                             edge_f=None,
                             is_weighted=False,
                             no_python=False))

     if dim==2:
         poincare_viz(embedding.get_embedding(),
                      2,
                      'Laplacian Eigenmaps',
                      'LE_d2.pdf)

    MAP, prec_curv, err, err_baseline = (gr
                                         .evaluateStaticGraphReconstruction(G,
                                                                            embedding,
                                                                            Y,
                                                                            None))
    lf.write('Reconstruction MAP = %f' %MAP)
    print('Reconstruction MAP = %f' %MAP)
    lf.close()

def GF_recon(G, file, dim, reg=1.0, maxiters=50000):
    lf = open(file, 'a')
    lf.write('Graph Factorization in d=%i' %dim)
    print('Graph Factorization in d=%i' %dim)

    embedding = GraphFactorization(d=dim,
                                   max_iter=maxiters,
                                   eta=1 * 10**-4,
                                   regu=reg)
    Y, t = (embedding
            .learn_embedding(graph=G,
                             edge_f=None,
                             is_weighted=False,
                             no_python=False))

     if dim==2:
         poincare_viz(embedding.get_embedding(),
                      3,
                      'Graph Factorization',
                      'GF_d2.pdf)

    MAP, prec_curv, err, err_baseline = (gr
                                         .evaluateStaticGraphReconstruction(G,
                                                                            embedding,
                                                                            Y,
                                                                            None))
    lf.write('Reconstruction MAP = %f' %MAP)
    print('Reconstruction MAP = %f' %MAP)
    lf.close()

lf = open('embedding_comparison.txt', 'w')
lf.write('This is the log of the analysis comparing different embedding methods.')
lf.close()

dims = [2, 5, 10]

for dim in dims:
    poincare_recon(edgelist[['Edge_From', 'Edge_To']].values,
                   'embedding_comparison.txt',
                   dim)
    LE_recon(G,
             'embedding_comparison.txt',
             dim)
    GF_recon(G,
             'embedding_comparison.txt',
             dim)
