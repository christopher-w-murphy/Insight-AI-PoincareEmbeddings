"""
Load and clean the Library of Congress 'literature' dataset.
    The data came from their API by searching:
    https://www.loc.gov/books/?q=literature&c=150&sp=1&fo=json
    with sp running from 1 to 79.
    Note the json files are improperly formatted,
    so I needed to select the results by hand.
"""

import pandas as pd
import json

# configure
with open('../../configs/production.json', 'r') as f:
    config = json.load(f)

full_dataset = config['PREPROCESSING']['FULL_DATASET']
sp_min = config['PREPROCESSING']['SP_MIN']
sp_max = config['PREPROCESSING']['SP_MAX']
loc_json_path = config['PREPROCESSING']['LOC_JSON_PATH']

def read_loc_json(file):
    df = (pd
          .read_json(file,
                     orient='records'))
    return (df['results']
            .apply(pd
                   .Series))

def read_loc_jsons():

    # read the 1st json file
    df = read_loc_json(loc_json_path + 'loc' + str(sp_min) + '.json')

    # read the rest of the json files
    for i in range(sp_min+1, sp_min+1):
        fn = loc_json_path + 'loc' + str(i) + '.json'
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
     .to_csv(full_dataset,
             index=False))
