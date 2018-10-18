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
- GEM: ```pip install git+https://github.com/palash1992/GEM```
- Gensim: ```pip install --upgrade gensim``` or for *conda* environments ```conda install -c anaconda gensim```

## Test
Testing is done on a scraped dataset of 115 books. To try things out switch to the tests directory, ```cd tests/```. Then run ```python minimal_analysis_test.py``` to produce 2 .png figures of Poincar&eacute; embeddings, one based only on the genre information of the books, and one based on both description and genre information. This script also prints book recommendations based on "Nineteen Eighty-Four," and takes just over a minute to run on my MacBook Pro. 

## Analysis
See the STATIC folder for examples of Poincar&eacute; and Graph Factorization embeddings 

