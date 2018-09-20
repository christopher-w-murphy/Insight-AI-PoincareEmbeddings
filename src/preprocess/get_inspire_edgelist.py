"""
    get_inspire_edglist: Reads the data, cleans it, builds the graph, and returns the edges as a zipped csv file.
    The graph is a network of citations from InspireHEP, https://inspirehep.net/info/hep/api. This can be used to determine the PageRank of the papers in the network.
    This code is adapted from that of Eder Izaguirre (2018), https://github.com/ieder1357/InspireHEP-Network-Analysis.
"""
from __future__ import division
from collections import defaultdict
import json
import re
from dateutil.parser import parse

import networkx as nx

# utility functions we will need

def is_bad_publication(publication):
    """Identifies bad publications that should be eliminated
        from the analysis.
        Returns True if title of  article contains a low quality word or
        publication has no citations, references or authors.
        """
    # merge authors and co-authors into one
    authors = publication['authors'] + publication['co-authors']
    
    bad_title_strings = ['proceedings',
                         'proceeding',
                         'withdrawn',
                         'thesis',
                         'conference',
                         'canceled',
                         'cancelled']
    bad_re = re.compile('|'.join(bad_title_strings))
    title = publication['title'].lower()
    if  bad_re.search(title): # reject any paper if keyword in title
        return True
    if not authors: # lists are 'Truthy'. Empty lists are considered False
        return True
    if not publication['citations']: # reject papers with 0 citations
        return True
    if not publication['references']: # reject papers with 0 references
        return True
    
    return False

def fix_parse(creation_date):
    """ Return parsed date.
        If we get an obviously wrong date format, e.g. '02-31-1999'
        return default dummy date
        """
    try:
        return parse(creation_date, default = parse('06-25-2017'))
    except:
        return parse('01-01-1900')

def is_good_date(publication, date_range):
    """
    return True if publication falls within date range
    """
    if date_range: # apply date range cut
        pub_date = fix_parse(publication['creation_date'])
        min_date = parse(date_range[0]) # lower end of date range
        max_date = parse(date_range[1]) # higher end of date range
        return ((pub_date > min_date) and (pub_date < max_date))
    
    else: # if date_range = None, don't apply time range cut
        return True

def compute_graph(DG, recid_info, line, date_range = None):
    """Computes edges (i,j) for directed graph DG for current publication
        between (citation,publication) and (publication,reference)
        for each citation to publication and reference in publication,
        and builds recid: info dict
        where info = {'authors': , 'num_citations': , pub_date: '}
        """
    publication = json.loads(line)
    if not is_bad_publication(publication) and is_good_date(publication, date_range):
        recid = publication['recid']
        references = publication['references']
        citations = publication['citations']
        
        # add (citation, publication) edges
        for citation in citations:
            DG.add_edge(citation, recid)
        
        # add (publication, reference) edges
        for reference in references:
            DG.add_edge(recid,reference)
        

# let's read the data, clean it a bit, and build network

# key is recid, value is info
# info = {authors: [], citations: int, pub_date: datetime}
recid_info = defaultdict(dict)
json_path = '/home/ubuntu/data/'
DG = nx.DiGraph() # directed graph. Edges are (paper that cites, paper that's cited)
with open(json_path + 'hep_records.json') as f:
    list(
        map(lambda x: compute_graph(DG,
                                     recid_info,
                                     x),
             f.readlines()))

# write to a .csv file and gzip it
nx.write_edgelist(DG,
                  'inspire_edgelist.csv.gz',
                  delimiter=',',
                  data=False)
