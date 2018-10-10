import pandas as pd
import numpy as np
from matplotlib import cm

from ..configs import production
from . import ingestion
from . import model
from . import utils

if __name__ == '__main__':
    # load data
    df = (pd
          .read_csv(production.LITERATURE_FILE)
          .drop_duplicates(subset='title')
          .reset_index(drop=True))

    # process data (while loading more data)
    book_sim = (ingestion
                .df_recs_tfidf(df, production.MIN_CUT, production.MAX_CUT))

    ## genre-only
    edgelist = (ingestion
                .create_edgelist(production.CLASS_FILE, df))
    ## genre + description
    edgelist2 = (ingestion
                 .combine_edgelists(edgelist, book_sim))

    sc_ind = (ingestion
              .get_subclass_counts(df))
    ncls = production.N_CLS

    # prepare to analyze
    (utils
     .edgelist2tsv(edgelist[['NEdge_From', 'NEdge_To']].values,
                   production.GENRE_TSV_FNAME))
     G = (utils
          .tsv2edgelist(production.GENRE_TSV_FNAME))

    (utils
     .edgelist2tsv(edgelist2[['NEdge_From', 'NEdge_To']].values,
                   production.GENPDES_TSV_FNAME))
     G2 = (utils
           .tsv2edgelist(production.GENPDES_TSV_FNAME))

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
