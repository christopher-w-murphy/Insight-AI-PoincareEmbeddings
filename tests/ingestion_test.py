import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def df_recs_tfidf(df):
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

    simil = [(df['title'].iloc[i],
              df['title'].iloc[j],
              np.sum(X[i] * X[j].T))
             for i in range(len(X)) for j in range(len(X))]
    simil.sort(key=lambda x: x[2],
               reverse=True)

    df_out = (pd
              .DataFrame(simil[len(X):], # indexing w/ len(X): eliminates recommending the input book
                         columns=['book i',
                                  'book j',
                                  'score']))

    return df_out
