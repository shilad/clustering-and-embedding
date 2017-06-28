import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def embedTrustworthiness(vecs, embedding, k):
    """
    Computes the trustworthiness of the embedding: To what extent the local structure of data is retained in a low-dim
    embedding in a value between 0 - 1.
    :return: Precision between 0 - 1
    """
    # Compute pairwise distances in high and low dimension.
    vecs = vecs.as_matrix()
    dist_hd = squareform(pdist(vecs))
    neighbors_hd = dist_hd.argsort() + 1  # neighbors ordered by similarity
    k_neighbors_hd = neighbors_hd[:, :k]  # Get k nearest neighbors in high dimension

    points = embedding.as_matrix()
    dist_ld = squareform(pdist(points))
    neighbors_ld = dist_ld.argsort() + 1  # neighbors ordered by similarity
    neighbors_ld = neighbors_ld[:, :k]  # Get k nearest neighbors in low dimension

    # Compute trustworthiness
    n = len(vecs)
    T = 0
    for i in range(n):
        hd = k_neighbors_hd[i]
        ld = neighbors_ld[i]
        for j in ld:
            if j not in hd:
                pos = np.where(neighbors_hd[i] == j)[0][0]  # Find index of j in the sorted neighbors of i in hd.
                T += pos - k
    T = 1 - (2 / float(n * k * (2 * n - 3 * k - 1))) * T
    return T


def embedTrustworthiness_percent(vecs, embedding, k):
    """
    Trustworthiness is based on the percentage of local neighbors in high dimensional data that are retained in the
    low dimensional embedding.
    :return: Precision between 0 - 1
    """
    # Compute pairwise distances in high and low dimension.
    vecs = vecs.as_matrix()
    dist_hd = squareform(pdist(vecs))
    neighbors_hd = dist_hd.argsort() + 1  # neighbors ordered by similarity
    neighbors_hd = neighbors_hd[:, :k]  # Get k nearest neighbors in high dimension

    points = embedding.as_matrix()
    dist_ld = squareform(pdist(points))
    neighbors_ld = dist_ld.argsort() + 1  # neighbors ordered by similarity
    neighbors_ld = neighbors_ld[:, :k]  # Get k nearest neighbors in low dimension

    # Compute trustworthiness
    count = 0
    for (hd, ld) in zip(neighbors_hd, neighbors_ld):
        for i in hd:
            if i in ld:
                count += 1
    return count/float(neighbors_ld.shape[0]*neighbors_ld.shape[1])  # Percentage of retained neighbors

if __name__ == '__main__':
    # Read data
    input_dir = 'data/ext/simple'
    vecs = pd.read_table(input_dir + '/vectors.sample_500.tsv', skiprows=1, skip_blank_lines=True, header=None, index_col=0)
    points = pd.read_table(input_dir + '/coordinates.sample_500.tsv', index_col='index')

    import time
    start = time.time()

    print embedTrustworthiness(vecs, points, k=10)
    print embedTrustworthiness_percent(vecs, points, k=10)

    print "Time: ", time.time() - start
