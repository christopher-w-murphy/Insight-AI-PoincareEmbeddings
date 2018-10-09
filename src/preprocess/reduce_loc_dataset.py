"""
The Library of Congress dataset contains 3570 unique Shelf IDs!
That's way too many classes.
Let's reduce the dataset to make the analysis simplier.
I will only keep IDs where there are more than 30 books,
and remove those that have websites for IDs.
"""

import pandas as pd
import numpy as np

def shelf_query(shelfid):
    return 'shelf_id == "' + shelfid + '"'

def query_str():

    querystr = shelf_query(shelf_ids[0])

    for i in range(1, len(shelf_ids)):
        querystr += '  | '
        querystr += shelf_query(shelf_ids[i])

    return querystr

if __name__ == '__main__':

    df = (pd
          .read_csv('../data/loc_literature_full.csv'))

    # there are 30 IDs that appear more than 30 times
    shelf_ids = ((df['shelf_id']
                  .value_counts() > 30)[:30]
                 .index
                 .drop(['http://ww',
                        'http://lcc',
                        'http://catalo']))

    query_string = query_str()

    df_red = (df
              .query(query_string)
              .dropna()
              .reset_index(drop=True))

    class2name = (pd
                  .read_csv('../data/loc_class2name.csv',
                            index_col='Class'))

    df_red['subclass'] = [class2name['Name'].loc[df_red['shelf_id'][i]]
                          for i in df_red.index]

    (df_red
     .to_csv('../data/loc_literature_reduced.csv',
             columns=['title',
                      'subclass',
                      'description'],
             index=False))
