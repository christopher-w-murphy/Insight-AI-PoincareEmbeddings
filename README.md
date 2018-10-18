# Hierarchical Book Recommendations
Uses Poincar&eacute; Embeddings to make book recommendations to readers based on the descriptions and Library of Congress classifications of books. For more information, see http://bit.ly/2yOzMbo.

## Setup
Clone repository
```
repo_name=Insight-AI-PoincareEmbeddings
username=christopher-w-murphy
git clone https://github.com/$username/$repo_name
cd $repo_name
```

## Requisites
- Python 3
- GEM: `pip install git+https://github.com/palash1992/GEM`
- Gensim: `pip install --upgrade gensim` or for *conda* environments `conda install -c anaconda gensim`

## Test
Testing is done on a scraped dataset of 115 books. Try things out for yourself (it takes just over a minute to run on my MacBook Pro)
```
cd tests
python minimal_analysis_test.py
``` 
This will produce two figures of Poincar&eacute; embeddings, one based only on the genre information of the books, and one based on both description and genre information. The script also prints book recommendations based on "Nineteen Eighty-Four." For those with more time to spare, try running `python analysis_test.py` instead.

## Analysis
The main piece of analysis of this project was to demonstrate that Poincar&eacute embeddings do a better job of separating books of a given (sub)subgenre in the embedding space than traditional embedding methods. This can be seen, for example, by inspecting the figures below. Content-based book recommendations could then be made by finding the 5 (or 10, etc.) nearest neighbors to the book of interest in the embedding space, as measured using the metric of that space.

The data is the same in the two figures below. Each colored point corresponds to a book of a given subsubgenre with each subsubgenre having a unique color. The black points are the locations of genres themselves within the embedding space. The upper panel used the Poincar&eacute; embedding algorithm, whereas the lower panel used a embedding method known as Graph Factorization. See the STATIC folder for more examples of Poincar&eacute; and Graph Factorization embeddings 

![Poincar&eacute Embedding *d*=2](https://github.com/christopher-w-murphy/Insight-AI-PoincareEmbeddings/blob/master/STATIC/Poincare_d2.png)

![Graph Factorization *d*=2](https://github.com/christopher-w-murphy/Insight-AI-PoincareEmbeddings/blob/master/STATIC/GF_d2.png)
