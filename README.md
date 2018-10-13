# Hierarchical Book Recommendations
Uses Poincar&eacute; Embeddings to make book recommendations to readers based on the descriptions and Library of Congress classifications of books

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

## Configs
...

## Test
Running ```python tests/analysis_test.py``` should produce 6 .pdf figures, (1 .tsv file,) and print 3 sets of book recommendations

## Analysis
See the STATIC folder for examples of Poincar&eacute; and Graph Factorization embeddings 

