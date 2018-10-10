from envparse import env

# settings for utils and model
GEM_PATH = (env
            .str('GEM_PATH', default='/Users/christopherwmurphy/GEM'))

# settings for analysis and preprocessing
LITERATURE_FILE = (env
                   .str('LITERATURE_FILE',
                        default='../data/loc_literature_reduced.csv'))

# settings for analysis
CLASS_FILE = (env
              .str('CLASS_FILE',
                   default='../data/loc_class_edgelist.csv'))

MIN_CUT = (env
           .float('MIN_CUT', default=0.80))

MAX_CUT = (env
           .float('MAX_CUT', default=0.99))

N_CLS = (env
         .int('N_CLS', default=28))

GENRE_TSV_FNAME = (env
                   .str('GENRE_TSV_FNAME', default='../data/Nedgelist.tsv'))

GENPDES_TSV_FNAME = (env
                     .str('GENpDES_TSV_FNAME',
                          default='../data/Nedgelist2.tsv'))

# settings for preprocessing
FULL_DATASET = (env
                .str('FULL_DATASET', default='../data/loc_literature_full.csv'))
CLASS_NAMES = (env
               .str('CLASS_NAMES', default='../data/loc_class2name.csv'))
'../data/loc_literature_reduced.csv'
## the settings below are only need to be changed if you download (and hand clean!) the LoC jsons
SP_MIN = (env
          .int('SP_MIN', default=1))
SP_MAX = (env
          .int('SP_MAX', default=79))
LOC_JSON_PATH = (env
                 .str('LOC_JSON_PATH', default='../../data/')
