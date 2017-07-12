from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.

    From https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def testCountryQuality():
    N = 1000


    for i in range(10):
        circle = np.random.normal(size=(N // 3, 2))

        # Totally random
        coords = np.random.random((N, 2))
        coords[:, 0] *= 10.0
        coords[:, 1] *= 5.0

        clusters = np.random.randint(1, 10, (N,))

        q1 = countryQuality(coords, clusters)

        # cluster ids: first, second, and third circles
        clusters = np.concatenate((
            np.zeros((N//3,), dtype=int),
            np.zeros((N//3,), dtype=int) + 1,
            np.zeros((N//3,), dtype=int) + 2
        ))

        # mixed together circles
        coords = np.concatenate((
            np.copy(circle),
            np.copy(circle),
            np.copy(circle)
        ), axis=0)
        q2 = countryQuality(coords, clusters)

        # slightly separated
        coords = np.concatenate((
            np.copy(circle),
            np.copy(circle) + 0.5,
            np.copy(circle) + 1.0
        ), axis=0)
        q3 = countryQuality(coords, clusters)

        # more separated
        coords = np.concatenate((
            np.copy(circle),
            np.copy(circle) + 0.8,
            np.copy(circle) + 1.6
        ), axis=0)
        q4 = countryQuality(coords, clusters)

        # very separated
        coords = np.concatenate((
            np.copy(circle),
            np.copy(circle) + 1.0,
            np.copy(circle) + 2.0
        ), axis=0)
        q5 = countryQuality(coords, clusters)

        # ridiculously separated
        coords = np.concatenate((
            np.copy(circle),
            np.copy(circle) + 100000000,
            np.copy(circle) + 200000000
        ), axis=0)
        q6 = countryQuality(coords, clusters)

        assert(q1 < q3)
        assert(q1 < q4)
        assert(q2 < q3)
        assert(q3 < q4)
        assert(q4 < q5)
        assert(q5 < q6)


def countryQuality(coords, clusters, minPerCell=5):
    """
    Returns a score between 0.0 and 1.0 that indicates the "quality" of countries in 2-d space.

    :param coords: 2-D array of X,Y coordinates
    :param clusters: 1-D array of cluster ids.
    :return:
    """

    coords = np.array(coords)
    clusters = np.array(clusters)
    assert(clusters.shape[0] == coords.shape[0])
    assert(coords.shape[1] == 2)
    C = np.max(clusters)
    N = coords.shape[0]
    mins = np.min(coords, axis=0)
    maxes = np.max(coords, axis=0)

    # Minimum zoom where clusters could be distinctly embedded
    startZoom = int(np.ceil(np.log2(C ** 0.5)))

    nZooms = 0
    total = 0
    for zoom in range(startZoom, 1000):
        numCells = 2**zoom
        if N / (numCells ** 2) < minPerCell:
            break
        sz = np.min(maxes - mins + 0.0001) / numCells
        dims = np.ceil((maxes - mins) / sz).astype(int)
        counts = np.zeros((dims[0], dims[1], C))
        gridxys = np.floor((coords - mins) / sz).astype(int)
        for i in range(N):
            x,y = gridxys[i,:]
            c = clusters[i]
            counts[x, y, c-1] += 1

        counts = counts.reshape((dims[0] * dims[1], C))
        cell_maxes = np.max(counts, axis=1)             # Actual max per cell
        sums = np.sum(counts, axis=1)                   # Number of points per cell
        cell_mins = np.ceil(sums / C).astype(int)       # Theoretical minimum cell max for that sum

        # The following is a ratio of actual score / max possible score
        # Both are adjusted by subtracting off the MINIMUM possible max for a cell.
        cell_scores = np.nan_to_num((cell_maxes - cell_mins) / (sums - cell_mins + 0.0000000001))

        # Weights are based on degrees of freedom, so a cell with one point has weight zero
        weights = np.sqrt(np.maximum(0, sums - 1))

        score = np.sum(weights * cell_scores) / np.sum(weights)
        total += score
        nZooms += 1

    return total / nZooms

def embedTrustworthiness(vecs, embedding, k=10):
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

    if type(embedding) == np.ndarray:
        points = embedding
    else:
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


def neighborOverlap(vecs, embedding, k=10):
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

    if type(embedding) == np.ndarray:
        points = embedding
    else:
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


def labelMetric(tfidf_scores):
    """
    :param tfidf_scores: A data frame with the following columns: articles ids, tf-idf scores, clusters 
    :return: A score of clusters' labels based on the average tf-idf scores of articles in the clusters 
    """
    stat = defaultdict(list)
    for id, row in tfidf_scores.iterrows():
        stat[row['cluster']].append(np.mean([x[1] for x in row['score']]))
    return np.mean([np.mean(stat[i]) for i in stat])


if __name__ == '__main__':
    # Read data
    input_dir = 'data/ext/simple'
    vecs = pd.read_table(input_dir + '/vectors.sample_500.tsv', skiprows=1, skip_blank_lines=True, header=None, index_col=0)
    points = pd.read_table(input_dir + '/coordinates.sample_500.tsv', index_col='index')

    import time
    start = time.time()

    print embedTrustworthiness(vecs, points, k=10)
    print neighborOverlap(vecs, points, k=10)

    print "Time: ", time.time() - start
