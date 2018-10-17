import pandas as pd
import numpy as np

import ingestion_test

import sys
sys.path.append('../')

from src import ingestion
from src import model
from src import utils

def recommender(title, genre_embed, genPdes_embed, des_df, n_recs=5):
    """
    returns a DataFrame w/ n_recs suggestions based on genre and/or descriptions
    title: the title of book the reader liked
    genre_embed: the vectors from the genre-only analysis
    genPdes_embed: the vectors from the genre + description analysis
    des_df: the DataFrame with the description-only analysis
    """
    df = (des_df[(des_df['book i'] == title)]
          .sort_values(by='score'))[-n_recs:][::-1]

    data = {'genre only':(np
                          .array(genre_embed
                                 .most_similar(title, n_recs))
                          .T[0]),
            'genre and description':(np
                                     .array(genPdes_embed
                                            .most_similar(title, n_recs))
                                     .T[0]),
            'description only':(df[df['book i'] == title]['book j']
                                .values)}

    return (pd
            .DataFrame(data, index=range(1,1+n_recs)))

if __name__ == '__main__':
    from matplotlib import cm
    import json

    # configure
    with open('../configs/tests.json', 'r') as f:
        config = json.load(f)

    literature_file = config['ANALYSIS']['LITERATURE_FILE']
    class_file = config['ANALYSIS']['CLASS_FILE']
    threshold = config['ANALYSIS']['THRESHOLD']
    n_cls = config['ANALYSIS']['N_CLS']
    genre_tsv_fname = config['ANALYSIS']['GENRE_TSV_FNAME']
    genpdes_tsv_fname = config['ANALYSIS']['GENPDES_TSV_FNAME']

    # load data
    df = (pd
          .read_csv(literature_file))
    literature_only = (df['subclass']>=800)

    # process data (while loading more data)
    recs_trad = (ingestion_test
                 .df_recs_tfidf(df[literature_only]))

    ## genre-only
    edgelist1 = (ingestion
                 .create_edgelist(class_file,
                                  df[literature_only]))
    ## genre + descriptions
    thres_crit = (recs_trad['score'] > threshold)
    edgelist = (ingestion
                .combine_edgelists(edgelist1,
                                   recs_trad[thres_crit]))

    sc_ind = (ingestion
              .get_subclass_counts(df[literature_only]))

    # prepare to analyze
    ## genre-only
    (utils
     .edgelist2tsv(edgelist1[['NEdge_From', 'NEdge_To']]
                   .values,
                   genre_tsv_fname))
    G1 = (utils
          .tsv2edgelist(genre_tsv_fname))
    ## genre + description
    (utils
     .edgelist2tsv(edgelist[['NEdge_From', 'NEdge_To']]
                   .values,
                   genpdes_tsv_fname))
    G = (utils
         .tsv2edgelist(genpdes_tsv_fname))

    colors = (cm
              .rainbow(np
                       .linspace(0, 1, len(sc_ind)-1)))

    # analyze!
    ## visualize embeddings
    embedding1 = (model
                  .train_poincare_model(edgelist1[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding1.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=2$ Genre-Only',
                   'poincare_genre_te.pdf'))

    embedding2, Y2 = (model
                      .train_GF_model(G1))
    (utils
     .poincare_viz(embedding2.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=2$ Genre-Only',
                   'GF_genre_te.pdf'))

    embedding3 = (model
                  .train_poincare_model(edgelist[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding3.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=2$',
                   'poincare_d2_te.pdf'))

    embedding4 = (model
                  .train_poincare_model(edgelist[['Edge_From', 'Edge_To']]
                                        .values,
                                        dim=10))
    (utils
     .poincare_viz(embedding4.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=10$',
                   'poincare_d10_te.pdf'))

    embedding5, Y5 = (model
                      .train_GF_model(G))
    (utils
     .poincare_viz(embedding5.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=2$',
                   'GF_d2_te.pdf'))

    embedding6, Y6 = (model
                      .train_GF_model(G,
                      dim=10))
    (utils
     .poincare_viz(embedding6.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=10$',
                   'GF_d10_te.pdf'))

    ## print recommendations
    titles = ['Nineteen Eighty-Four',
              'Clash of Kings, A',
              'Pride and Prejudice and Zombies']
    for title in titles:
        print('Here are recommendations based on %s:' %title)
        recsdf = recommender(title,
                             embedding1.kv,
                             embedding3.kv,
                             recs_trad)
        print(recsdf)
