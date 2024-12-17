import scipy.sparse as sp # type: ignore
import numpy as np
def sample_random_walks(adj_matrix, walk_length, walks_per_node):
    """
    Perform random walks on the adjacency matrix, skipping isolated nodes.

    Parameters:
    - adj_matrix: The adjacency matrix (as a sparse CSR matrix).
    - walk_length: The length of each random walk.
    - walks_per_node: The number of walks to sample per node.
    
    Returns:
    - walks: A list of random walks, where each walk is a list of node IDs.
    """
    # Convert adjacency matrix to a sparse format (if not already)
    adj_matrix = sp.csr_matrix(adj_matrix)
    
    walks = []
    for node in range(adj_matrix.shape[0]):
        for _ in range(walks_per_node):
            walk = [node]
            for _ in range(walk_length - 1):
                # Get the neighbors of the current node
                neighbors = adj_matrix.indices[adj_matrix.indptr[walk[-1]]:adj_matrix.indptr[walk[-1] + 1]]
                
                # If the node has no neighbors (isolated node), break early
                if len(neighbors) == 0:
                    break

                # Perform random walk: sample a random neighbor
                walk.append(np.random.choice(neighbors))
            
            # Only add the walk if it's not empty
            if len(walk) > 1:
                walks.append(walk)
    return walks
