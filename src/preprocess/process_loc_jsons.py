"""
Load and clean the Library of Congress 'literature' dataset.
    The data came from their API by searching:
    https://www.loc.gov/books/?q=literature&c=150&sp=1&fo=json
    with sp running from 1 to 79.
    Note the json files are improperly formatted,
    so I needed to select the results by hand.
"""

import pandas as pd

from ...configs import production

def read_loc_json(file):
    df = (pd
          .read_json(file,
                     orient='records'))
    return (df['results']
            .apply(pd
                   .Series))

def read_loc_jsons():

    # read the 1st json file
    df = read_loc_json(production.LOC_JSON_PATH + 'loc' + str(spmin) + '.json')

    # read the rest of the json files
    for i in range(production.SP_MIN+1, production.SP_MAX+1):
        fn = production.LOC_JSON_PATH + 'loc' + str(i) + '.json'
        dfi = read_loc_json(fn)
        df = (df
              .append(dfi,
                      ignore_index=True))

    # select the relevant data
    df = df[['title',
             'shelf_id',
             'description']]

    # clean the data
    df['description'] = (df['description']
                         .apply(pd
                                .Series))
    df['shelf_id'] = [shelfid.split('.')[0][:-1] for shelfid in df['shelf_id']]

    return df

if __name__ == '__main__':

    df = read_loc_jsons()

    (df
     .to_csv(production.FULL_DATASET,
             index=False))
