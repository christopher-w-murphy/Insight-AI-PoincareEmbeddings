if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from matplotlib import cm
    import json

    import ingestion
    import model
    import utils

    # configure
    with open('../configs/production.json', 'r') as f:
        config = json.load(f)

    literature_file = config['ANALYSIS']['LITERATURE_FILE']
    class_file = config['ANALYSIS']['CLASS_FILE']
    min_cut = config['ANALYSIS']['MIN_CUT']
    max_cut = config['ANALYSIS']['MAX_CUT']
    n_cls = config['ANALYSIS']['N_CLS']
    genre_tsv_fname = config['ANALYSIS']['GENRE_TSV_FNAME']
    genpdes_tsv_fname = config['ANALYSIS']['GENPDES_TSV_FNAME']


    # load data
    df = (pd
          .read_csv(literature_file)
          .drop_duplicates(subset='title')
          .reset_index(drop=True))

    # process data (while loading more data)
    book_sim = (ingestion
                .df_recs_tfidf(df, min_cut, max_cut))

    ## genre-only
    edgelist = (ingestion
                .create_edgelist(class_file, df))
    ## genre + description
    edgelist2 = (ingestion
                 .combine_edgelists(edgelist, book_sim))

    sc_ind = (ingestion
              .get_subclass_counts(df))

    # prepare to analyze
    (utils
     .edgelist2tsv(edgelist[['NEdge_From', 'NEdge_To']].values,
                   genre_tsv_fname))
    G = (utils
         .tsv2edgelist(genre_tsv_fname))

    (utils
     .edgelist2tsv(edgelist2[['NEdge_From', 'NEdge_To']].values,
                   genpdes_tsv_fname))
    G2 = (utils
          .tsv2edgelist(genpdes_tsv_fname))

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
                   n_cls,
                   colors,
                   'Poincare Embedding $d=2$ Genre-Only',
                   'poincare_genre.pdf'))

    embedding2, Y2 = (model
                      .train_GF_model(G))
    (utils
     .poincare_viz(embedding2.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=2$ Genre-Only',
                   'GF_genre.pdf'))

    embedding3 = (model
                  .train_poincare_model(edgelist2[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding3.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=2$',
                   'poincare_d2.pdf'))

    embedding4 = (model
                  .train_poincare_model(edgelist2[['Edge_From', 'Edge_To']]
                                        .values,
                                        dim=10))
    (utils
     .poincare_viz(embedding4.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=10$',
                   'poincare_d10.pdf'))

    embedding5, Y5 = (model
                      .train_GF_model(G2))
    (utils
     .poincare_viz(embedding5.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=2$',
                   'GF_d2.pdf'))

    embedding6, Y6 = (model
                      .train_GF_model(G2,
                      dim=10))
    (utils
     .poincare_viz(embedding6.get_embedding(),
                   sc_ind,
                   n_cls,
                   colors,
                   'Graph Factorization $d=10$',
                   'GF_d10.pdf'))
