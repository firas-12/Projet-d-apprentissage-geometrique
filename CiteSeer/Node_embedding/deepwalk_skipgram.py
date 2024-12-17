from sample_random_walks import sample_random_walks
import numpy as np
from gensim.models import Word2Vec # type: ignore
def deepwalk_skipgram(adj_matrix, embedding_dim=64, walk_length=80, walks_per_node=10,
                      workers=8, window_size=10, num_neg_samples=1):
    """
    Compute DeepWalk embeddings using skip-gram on the given adjacency matrix.
    
    Parameters:
    - adj_matrix: Adjacency matrix of the graph.
    - embedding_dim: Dimensionality of node embeddings.
    - walk_length: Length of each random walk.
    - walks_per_node: Number of random walks per node.
    - workers: Number of parallel workers for Word2Vec training.
    - window_size: Size of the context window for the skip-gram model.
    - num_neg_samples: Number of negative samples for training.
    
    Returns:
    - embeddings: The learned node embeddings.
    """
    # Sample random walks from the graph
    walks = sample_random_walks(adj_matrix, walk_length, walks_per_node)
    
    # Convert walks into string format as expected by Word2Vec
    walks = [list(map(str, walk)) for walk in walks]
    
    # Train the Word2Vec model using the skip-gram approach
    model = Word2Vec(walks, vector_size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                     epochs=1, negative=num_neg_samples, hs=0, compute_loss=True)
    
    # Get the embeddings for each node, using a default vector for missing nodes
    embeddings = []
    for node in range(len(adj_matrix)):
        try:
            # Ensure the embedding is a numpy array
            embeddings.append(model.wv[str(node)])
        except KeyError:
            # If the node is not present in the learned embeddings, assign a random embedding
            embeddings.append(np.random.uniform(-1, 1, embedding_dim))
    
    # Convert the list of embeddings into a numpy array
    return np.array(embeddings)