import pandas as pd
import numpy as np
from matplotlib import cm

import utils
import ingestion
import model

if __name__ == '__main__':
    # load data
    df = (pd
          .read_csv('../data/loc_literature_reduced.csv')
          .drop_duplicates(subset='title')
          .reset_index(drop=True))

    # process data (while loading more data)
    book_sim = (ingestion
                .df_recs_tfidf(df, 0.80, 0.99))

    ## genre-only
    edgelist = (ingestion
                .create_edgelist('../data/loc_class_edgelist.csv', df))
    ## genre + description
    edgelist2 = combine_edgelists(edgelist,
                                  book_sim)

    sc_ind = (ingestion
              .get_subclass_counts(df))
    ncls = 28

    # prepare to analyze
    (utils
     .edgelist2tsv(edgelist[['NEdge_From', 'NEdge_To']].values,
                   '../data/Nedgelist.tsv'))
     G = (utils
          .tsv2edgelist('../data/Nedgelist.tsv'))

    (utils
     .edgelist2tsv(edgelist2[['NEdge_From', 'NEdge_To']].values,
                   '../data/Nedgelist2.tsv'))
     G2 = (utils
           .tsv2edgelist('../data/Nedgelist2.tsv'))

    colors = (cm
              .rainbow(np
                       .linspace(0, 1, len(sc_ind)-1)))

    # analyze!
    embedding1 = (model
                  .train_poincare_model(edgelist[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding1.kv.vectors,
                   sc_ind,
                   ncls,
                   colors,
                   'Poincare Embedding $d=2$ Genre-Only',
                   'poincare_genre.pdf'))

    embedding2, Y2 = (model
                      .train_GF_model(G))
    (utils
     .poincare_viz(embedding2.get_embedding(),
                   sc_ind,
                   ncls,
                   colors,
                   'Graph Factorization $d=2$ Genre-Only',
                   'GF_genre.pdf'))

    embedding3 = (model
                  .train_poincare_model(edgelist2[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding3.kv.vectors,
                   sc_ind,
                   ncls,
                   colors,
                   'Poincare Embedding $d=2$',
                   'poincare_d2.pdf'))

    embedding4 = (model
                  .train_poincare_model(edgelist2[['Edge_From', 'Edge_To']]
                                        .values,
                                        d=10))
    (utils
     .poincare_viz(embedding4.kv.vectors,
                   sc_ind,
                   ncls,
                   colors,
                   'Poincare Embedding $d=10$',
                   'poincare_d10.pdf'))

    embedding5, Y5 = (model
                      .train_GF_model(G2))
    (utils
     .poincare_viz(embedding5.get_embedding(),
                   sc_ind,
                   ncls,
                   colors,
                   'Graph Factorization $d=2$',
                   'GF_d2.pdf'))

    embedding6, Y6 = (model
                      .train_GF_model(G2,
                      d=10))
    (utils
     .poincare_viz(embedding6.get_embedding(),
                   sc_ind,
                   ncls,
                   colors,
                   'Graph Factorization $d=10$',
                   'GF_d10.pdf'))
