# Taken from https://github.com/zaburo-ch/Parametric-t-SNE-in-Keras/blob/master/mlp_param_tsne.py


import numpy as np
import tsne

from clumbed.evaluate.metrics import embedTrustworthiness, neighborOverlap
from clumbed.nn_tsne.utils import dist_matrix2, pca
from scipy import sparse

np.random.seed(71)
import pandas as pd

import matplotlib

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation

import tensorflow as tf

from sklearn.utils import shuffle

import multiprocessing as mp


initial_dims = 50
low_dim = 2
nb_epoch = 1000
shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 30
batch_size = 100


def Hbeta(i, D, beta):
    """
    Compute the perplexity (H) and P-row for a specific value of the precision of a
    Gaussian distribution.
    :param D: The distances
    :param beta: Guess at (1 / variance)
    :return: log(perplexity) and p_row
    """

    P = np.exp(-D * beta)       # Probability for each point
    P[i] = 0
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
    H, pRow = Hbeta(i, Di, beta)

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

        H, pRow = Hbeta(i, Di, beta)
        Hdiff = H - logU
        tries += 1


    # sum1 = pRow.sum()
    index = pRow.shape[0] - perplexity*3
    threshold = np.partition(pRow, index)[index]
    pRow[pRow < threshold] = 0.0
    pRow[np.isnan(pRow)] = 0
    # sum2 = pRow.sum()
    # print(sum1, sum2)

    return i, sparse.lil_matrix(pRow)


def x2p(X):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)

    # Square of L2 norms for each point
    sum_X = np.sum(np.square(X), axis=1)
    # print(sum_X.shape)

    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))
    # print(D.shape)

    # idx = (1 - np.eye(n)).astype(bool)
    # D = D[idx].reshape([n, -1])
    # print(D.shape)

    def generator():
        for i in xrange(n):
            yield i, D[i], tol, logU

    pool = mp.Pool(n_jobs)
    result = pool.map(x2p_job, generator())
    P = sparse.lil_matrix((n, n))
    for i, pRow in result:
        P[i,:] = pRow
    P = sparse.csr_matrix(P)

    return P


def calculate_P(X):
    print "Computing pairwise distances..."
    P = x2p(X)
    P = P + P.T
    P = P / P.sum()
    # P = np.maximum(P, 1e-12)
    print(P, P.data.shape)
    return P


def KLdivergence(P, Y):
    eps = K.variable(10e-15)

    # Calculate euclidean distance matrix squared
    D = dist_matrix2(Y, K)

    print(P)

    # Calculate TSNE probability in embedded space
    Q = K.pow(1 + D, -1)
    Q *= (1 - K.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)

    # Calculate KL divergence between original prob P and TSNE prob Q
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)

    return C

print "load data"
# # cifar-10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# n, channel, row, col = X_train.shape

input_dir = './data/simple_5000'
df = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

vecs = df.as_matrix()


# Sanity check
coords = tsne.bh_sne(vecs, perplexity=perplexity)
print('original values:')
print(embedTrustworthiness(pd.DataFrame(vecs), pd.DataFrame(coords), 5))
print(neighborOverlap(pd.DataFrame(vecs), pd.DataFrame(coords), 5))


# X = pca(vecs, initial_dims).real
X = vecs
n, d = X.shape

print "build model"
model = Sequential()
model.add(Dense(25, input_shape=(d,)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(2))

model.compile(loss=KLdivergence, optimizer="adam")

print "fit"
images = []

P = calculate_P(X)
P *= 4

print(P.shape)

for epoch in range(nb_epoch):
    if epoch == 250:
        P /= 4

    # train
    indexes = np.random.permutation(n)
    loss = 0
    for i in range(0, n, batch_size):
        batch_indexes = indexes[i:i+batch_size]
        X_ = X[batch_indexes]
        P_ = P[batch_indexes].T[batch_indexes].T
        loss += model.train_on_batch(X_, P_.toarray())

    # visualize training process
    coords = model.predict(X)

    if epoch % 50 == 0:
        print "Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss)
        print('trustworthiness %f, overlap %f' %
              (embedTrustworthiness(pd.DataFrame(vecs), pd.DataFrame(coords), 5),
              neighborOverlap(pd.DataFrame(vecs), pd.DataFrame(coords), 5)))
