import numpy as np
def _random_walk(indptr, indices, walk_length, walks_per_node, seed):
    """Sample r random walks of length l per node in parallel from the graph.

    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    seed : int
        Random seed

    Returns
    -------
    walks : array-like, shape [r*N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for ir in range(walks_per_node):
        for n in range(N):
            for il in range(walk_length):
                walks.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])

    return np.array(walks)