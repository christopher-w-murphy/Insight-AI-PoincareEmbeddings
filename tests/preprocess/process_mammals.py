import pandas as pd

DATA_PATH = '../../data'

df = (pd
      .read_table(DATA_PATH + '/preprocessed/mammal_closure.tsv',
                  header=None))

df[0] = (df[0]
         .astype('category'))
df[1] = (df[1]
         .astype('category'))

df['0c'] = (df[0]
            .cat
            .codes)
df['1c'] = (df[1]
            .cat
            .codes)

(df
 .to_csv(DATA_PATH + '/processed/mammal_edgelist.tsv',
         sep='\t',
         columns=['0c', '1c'],
         header=False,
         index=False))        
