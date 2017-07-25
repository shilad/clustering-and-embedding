import numpy as np
import multiprocessing as mp


def Hbeta(D, beta):
    """
    Compute the perplexity (H) and P-row for a specific value of the precision of a
    Gaussian distribution.
    :param D: The distances
    :param beta: Guess at (1 / variance)
    :return: log(perplexity) and p_row
    """

    P = np.exp(-D * beta)       # Probability for each point
    sumP = np.sum(P)            # Normalizing constant

    # WTF is this? I think log(perplexity), but not sure why
    H = np.log(sumP) + beta * np.sum(D * P) / sumP

    P = P / sumP                # Normalized probabilities

    assert(sumP != 0)
    return H, P


def x2p_job(data):
    """
    Bisection search for the correct theta
    :param data:
    :return:
    """
    i, Di, tol, logU = data
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf
    H, thisP = Hbeta(Di, beta)

    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0:
            betamin = beta
            if betamax == np.inf or betamax == -np.inf:
                beta = beta * 2
            else:
                beta = (betamin + betamax) / 2
        else:
            betamax = beta
            if betamin == np.inf or betamin == -np.inf:
                beta = beta / 2
            else:
                beta = (betamin + betamax) / 2

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return i, thisP


def x2p(X, perplexity, n_jobs):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)

    D = dist_matrix2(X, np)

    # Ignore distance to itself
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])

    def generator():
        for i in xrange(n):
            yield i, D[i], tol, logU

    pool = mp.Pool(n_jobs)
    result = pool.map(x2p_job, generator())
    P = np.zeros([n, n])
    for i, thisP in result:
        P[i, idx[i]] = thisP

    return P


def calculate_P(X, n_jobs=4, perplexity=30):
    print "Computing pairwise distances..."
    P = x2p(X, perplexity, n_jobs)
    P[np.isnan(P)] = 0
    P = P + P.T
    P = P / P.sum()
    P = np.maximum(P, 1e-12)
    return P

def dist_matrix2(X, M):
    """
    Returns the square of a euclidean distance matrix for the matrix X.
    Distances are squared (e.g. [(x1-x2)**2 + (y1-y2)**2])
    :param X: Input matrix, N x D
    :param M: Math package: keras.backend or numpy
    :return: N x N distance matrix
    """

    # Square of L2 norms for each point
    sum_X = M.sum(M.square(X), axis=1)

    # This is a shortcut for calculating pairwise distances.
    # https://stackoverflow.com/a/37040451
    return (sum_X + (M.reshape(sum_X, [-1, 1]) - 2 * M.dot(X, M.transpose(X))))

def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:,0:no_dims])
	return Y

def test_dist_matrix():
    from pytest import approx
    import keras.backend as K

    N = 10
    X = np.random.rand(N, 2)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = np.sum((X[i] - X[j]) ** 2)

    D2 = dist_matrix2(X, np)
    for i in range(N):
        for j in range(N):
            assert D2[i][j] == approx(D[i][j], abs=0.001)

    D3 = dist_matrix2(K.constant(X), K)
    D4 = K.eval(D3)
    for i in range(N):
        for j in range(N):
            assert D4[i][j] == approx(D[i][j], abs=0.001)