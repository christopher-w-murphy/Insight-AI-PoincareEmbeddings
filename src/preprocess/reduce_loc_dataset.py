"""
The Library of Congress dataset contains 3570 unique Shelf IDs!
That's way too many classes.
Let's reduce the dataset to make the analysis simplier.
I will only keep IDs where there are more than 30 books,
and remove those that have websites for IDs.
"""

def shelf_query(shelfid):
    return 'shelf_id == "' + shelfid + '"'

def query_str():

    querystr = shelf_query(shelf_ids[0])

    for i in range(1, len(shelf_ids)):
        querystr += '  | '
        querystr += shelf_query(shelf_ids[i])

    return querystr

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import json

    # configure
    with open('../../configs/production.json', 'r') as f:
        config = json.load(f)

    full_dataset = config['PREPROCESSING']['FULL_DATASET']
    class_names = config['PREPROCESSING']['CLASS_NAMES']
    reduced_dataset = config['PREPROCESSING']['REDUCED_DATASET']

    # process
    df = (pd
          .read_csv(full_dataset))

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
                  .read_csv(class_names,
                            index_col='Class'))

    df_red['subclass'] = [class2name['Name'].loc[df_red['shelf_id'][i]]
                          for i in df_red.index]

    (df_red
     .to_csv(reduced_dataset,
             columns=['title',
                      'subclass',
                      'description'],
             index=False))
