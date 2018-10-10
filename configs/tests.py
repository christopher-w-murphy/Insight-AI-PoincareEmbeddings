from envparse import env

# settings for the test analysis
LITERATURE_FILE = (env
                   .str('LITERATURE_FILE',
                        default='../data/books_descriptions_small.csv'))

CLASS_FILE = (env
              .str('CLASS_FILE', default='../data/books_edgelist_small.csv'))

THRESHOLD = (env
             .float('THRESHOLD', default=0.07))

N_CLS = (env
         .int('N_CLS', default=30))

GENRE_TSV_FNAME = (env
                   .str('GENRE_TSV_FNAME',
                        default='../data/books_edgelist1_small.tsv'))

GENPDES_TSV_FNAME = (env
                     .str('GENPDES_TSV_FNAME',
                          default='../data/books_edgelist_small.tsv'))
