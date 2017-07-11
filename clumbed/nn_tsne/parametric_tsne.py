# Taken from https://github.com/zaburo-ch/Parametric-t-SNE-in-Keras/blob/master/mlp_param_tsne.py


import numpy as np

from clumbed.evaluate.metrics import embedTrustworthiness, neighborOverlap

np.random.seed(71)
import pandas as pd

import matplotlib

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation

import multiprocessing as mp


initial_dims = 50
low_dim = 2
nb_epoch = 500
shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 3.0


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

    print((1 / beta) ** 0.5)

    return i, thisP


def x2p(X):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)

    # Square of L2 norms for each point
    sum_X = np.sum(np.square(X), axis=1)

    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

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


def calculate_P(X):
    print "Computing pairwise distances..."
    P = x2p(X)
    P[np.isnan(P)] = 0
    P = P + P.T
    P = P / P.sum()
    P = np.maximum(P, 1e-12)
    return P


def KLdivergence(P, Y):
    alpha = low_dim - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(500))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:,0:no_dims])
	return Y

print "load data"
# # cifar-10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# n, channel, row, col = X_train.shape

input_dir = './data/simple_500'
df = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

vecs = df.as_matrix()


# Sanity check



X = pca(vecs, initial_dims).real
n, d = X.shape

print "build model"
model = Sequential()
model.add(Dense(500, input_shape=(d,)))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dense(2))

model.compile(loss=KLdivergence, optimizer="adam")

print "fit"
images = []

P = calculate_P(X)
P *= 4

for epoch in range(nb_epoch):
    if epoch == 100:
        P /= 4
    # train
    loss = model.train_on_batch(X, P)
    print "Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss)

    # visualize training process
    coords = model.predict(X)

    print(embedTrustworthiness(pd.DataFrame(vecs), pd.DataFrame(coords), 10))
    print(neighborOverlap(pd.DataFrame(vecs), pd.DataFrame(coords), 10))
