import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def df_recs_tfidf(df, cutmin=0.0, cutmax=1.0):
    """
    Takes a DataFrame of book titles, DDC, and descriptions.
    Computes Tfidf vectors based on book descriptions.
    Returns a DataFrame of similiarity scores for the books
    based on the inner product of their vectors.
    """

    Tfidf_vec = TfidfVectorizer(stop_words = 'english')

    X = (Tfidf_vec
         .fit_transform(df['description'])
         .todense())

    simil = [(i,
              j,
              np.sum(X[i] * X[j].T))
             for i in range(len(X)) for j in range(i)]

    df2 = (pd
           .DataFrame(simil,
                      columns=['Ni',
                               'Nj',
                               'score']))

    threshold = (df2['score'] >= cutmin) & (df2['score'] <= cutmax)

    df3 = (df2[threshold]
           .reset_index(drop=True))
    df3['book i'] = [df['title'].loc[i] for i in df3['Ni'].values]
    df3['book j'] = [df['title'].loc[j] for j in df3['Nj'].values]

    return df3[['book i', 'book j', 'score']]

def create_edgelist(file, df):
    """
    creates an edgelist based on genre info
    """
    # load edges from the (sub)genres themselves
    df1 = (pd
           .read_csv(file,
                     dtype='str'))

    # get edges from the book descriptions df
    df2 = (df[['title',
               'subclass']]
           .rename(columns={'title':'Edge_From',
                            'subclass':'Edge_To'})
           .sort_values(by='Edge_To'))

    # combine the two dfs
    df3 = (df1
           .append(df2,
                   ignore_index=True))

    # consistently assign categories
    df4 = (df3
           .stack()
           .astype('category')
           .unstack())

    # make the categorical values explicit for later convenience
    for name in df4.columns:
        df4['N' + name] = (df4[name]
                           .cat
                           .codes)

    return df4

def combine_edgelists(el1, el2):
    """
    Combines two edgelists.
    The 1st set of edges comes from the subgenre info.
    The 2nd set comes from the similarity of the book descriptions.
    """

    # process 1st edgelist
    df1 = el1[['Edge_From',
               'Edge_To']]

    # process 2nd edgelist
    df2 = (el2[['book i',
                'book j']]
           .rename(columns={'book i':'Edge_From',
                            'book j':'Edge_To'}))
    # these edges are not directed
    df3 = (el2[['book j',
                'book i']]
           .rename(columns={'book j':'Edge_From',
                            'book i':'Edge_To'}))

    # combine edgelists
    el = (df1
          .append(df2,
                  ignore_index=True)
          .append(df3,
                  ignore_index=True))

    # consistently assign categories
    el2 = (el
           .stack()
           .astype('category')
           .unstack())

    # make the categorical values explicit
    for name in el2.columns:
        el2['N' + name] = (el2[name]
                           .cat
                           .codes)

    return el2

def get_subclass_counts(df):
    """
    returns a list that is a running count of
    the number of books in the first i subclasses
    where i is the position within the list
    """
    subclass_counts = (df['subclass']
                       .value_counts()
                       .reset_index()
                       .sort_values(by='index'))

    sc_ind = [np.sum(subclass_counts['subclass'].values[:n])
              for n in range(len(subclass_counts)+1)]
    return sc_ind
