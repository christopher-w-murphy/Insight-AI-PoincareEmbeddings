if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import json
    from time import time
    from matplotlib import cm
    from subprocess import call

    import analysis_test
    import ingestion_test

    import sys
    sys.path.append('../')

    from src import ingestion
    from src import model
    from src import utils

    # configure
    with open('../configs/tests.json', 'r') as f:
        config = json.load(f)

    literature_file = config['ANALYSIS']['LITERATURE_FILE']
    class_file = config['ANALYSIS']['CLASS_FILE']
    threshold = config['ANALYSIS']['THRESHOLD']
    n_cls = config['ANALYSIS']['N_CLS']

    # time how long the test takes
    t1 = time()
    print('Running Minimal Analysis Test...')

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
                   'poincare_genre_mintest.png'))
    call(['open', 'poincare_genre_mintest.png'])

    embedding2 = (model
                  .train_poincare_model(edgelist[['Edge_From', 'Edge_To']]
                                        .values))
    (utils
     .poincare_viz(embedding2.kv.vectors,
                   sc_ind,
                   n_cls,
                   colors,
                   'Poincare Embedding $d=2$',
                   'poincare_d2_mintest.png'))
    call(['open', 'poincare_d2_mintest.png'])

    ## print recommendations
    title = 'Nineteen Eighty-Four'
    print('If you liked the book %s, here are some recommendations based on:' %title)
    recsdf = (analysis_test
              .recommender(title,
                           embedding1.kv,
                           embedding2.kv,
                           recs_trad,
                           n_recs=10))
    print(recsdf)

    print('Minimal Analysis Test ran in %.1f seconds' %(time() - t1))
