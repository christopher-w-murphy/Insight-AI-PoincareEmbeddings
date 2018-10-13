from gensim.models.poincare import PoincareModel
from gensim.models.poincare import ReconstructionEvaluation
from gem.embedding.gf import GraphFactorization
from gem.evaluation import evaluate_graph_reconstruction as gr

def train_poincare_model(edgelist, dim=2, reg=0, nepochs=300):
    """
    Load and train the Poincare embedding model
    """
    embedding = PoincareModel(edgelist,
                              size=dim,
                              regularization_coeff=reg)

    (embedding
     .train(epochs=nepochs))

    return embedding

def poincare_reconstruction(filename, embedding):
    """
    returns the mean average precision from
    reconstructing the graph from the edgelist
    where the edgelist is loaded from a tsv file
    """
    recon = ReconstructionEvaluation(filename,
                                     embedding.kv)

    return (recon
            .evaluate()['MAP'])

def train_GF_model(graph, dim=2, reg=0.0, max_iter=50000, eta=1 * 10**-4):
    """
    Load and train the Graph Factorization model
    """
    embedding = GraphFactorization(d=dim,
                                   max_iter=max_iter,
                                   eta=eta,
                                   regu=reg)

    Y, t = (embedding
            .learn_embedding(graph=graph,
                             edge_f=None,
                             is_weighted=False,
                             no_python=False))

    return (embedding, Y)

def GF_reconstruction(graph, embedding, Y):
    """
    returns the mean average precision from reconstructing the graph
    """
    metrics = (gr
               .evaluateStaticGraphReconstruction(graph,
                                                  embedding,
                                                  Y2,
                                                  None))

    return metrics[0]
