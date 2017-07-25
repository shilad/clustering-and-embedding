'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import tsne

from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda
from keras.losses import kullback_leibler_divergence
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from numpy.linalg import norm
from sklearn.utils import shuffle


from clumbed.evaluate.metrics import embedTrustworthiness, neighborOverlap
from clumbed.nn_tsne.utils import calculate_P, dist_matrix2, pca

MAX_DIST = None

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def tsne_prob(vects):
    """
    Computes TSNE's q[i,j] which is distance in the embedded 2-D space:

                            (1 + D[i,j])^-1)
    q[i,j] = ----------------------------------------------
              sum-over-all k != i : (1 + D[i,k])^-1
    :param vects:
    :return:
    """
    x, y = vects
    xyDist = euclidean_distance(vects)

    stacked = K.concatenate([x, y], axis=1)
    allDists = K.sqrt(K.maximum(dist_matrix2(stacked, K), K.epsilon()))

    num = K.pow(1.0 + xyDist, -1.0)

    # We calculate the denominator sum over ALL cells, even those where k == i
    # To compensate, we subtract off the cells where k == i (the diagonals)
    # These will have a distance of zero, and (1 + 0)^-1 = 1.0 so we just
    # subtract off 1.0
    denom = K.sum(K.pow(1.0 + allDists, -1.0), axis=1, keepdims=True) - 1.0
    P = num / denom
    # P = tf.Print(P, [num, denom, P])

    return P

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def joint_loss(y_true, y_pred):
    P = y_true[:,0]
    D = y_true[:,1]
    R = y_true[:,2]

    kl = kullback_leibler_divergence(P, y_pred)
    return kl
    # st = stratified_loss(P, R, y_pred)
    #
    # kl = tf.Print(kl, (kl, st, R, y_pred))
    # return st

    # return 0.5 * kl + 0.5 * st

def stratified_loss(P_true, R, P_pred):
    for0 = (R - 1) * (R - 2) / 2 * K.maximum(0.0, 10.0 / N - P_pred)
    for1 = 0
    # for1 = (R - 0) * (R - 2) / -1 * K.maximum(0.0, P_pred - MEAN_MAX * 10)
    for2 = (R - 0) * (R - 1) / 2 * K.maximum(0.0, P_pred - 5.0 / N)

    return (for0 + for1 + for2) * N

def shilad_loss(high_dim_dist, low_dim_dist):
    """L2 Loss weighted by distance"""
    return K.mean((1 - high_dim_dist / MAX_DIST) * K.square(low_dim_dist - high_dim_dist), axis=-1)

def kl_divergence(p, q):
    return K.mean(p * K.log(p / K.maximum(K.epsilon(), q)), axis=-1)

def euclidean_loss(y_true, y_pred):
    """L2 Loss"""
    return K.mean(K.square(y_pred - y_true), axis=-1)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(25, input_shape=(input_dim,), activation='relu'))
    seq.add(Dense(50, input_shape=(input_dim,), activation='relu'))
    seq.add(Dense(500, input_shape=(input_dim,), activation='relu'))
    seq.add(Dense(2, activation='relu'))
    return seq

input_data = './data/simple_500/vectors.tsv'
df = pd.read_table(input_data, index_col=0, skiprows=1, header=None)

vecs = df.as_matrix()
N = vecs.shape[0]

vecP = calculate_P(vecs)

MEAN_MAX = np.mean(np.max(vecP, axis=0))
print('mean max', MEAN_MAX)

# get dimensions and normalize to unit vector
vecs = vecs / norm(vecs, axis=1).reshape(N, 1)
# vecs = pca(vecs, 50)
ndims = vecs.shape[1]

coords = tsne.bh_sne(vecs, perplexity=30)
print('original values:')
print(embedTrustworthiness(pd.DataFrame(vecs), pd.DataFrame(coords), 10))
print(neighborOverlap(pd.DataFrame(vecs), pd.DataFrame(coords), 10))

N = vecs.shape[0]
ids = df.index.tolist()


dists = dist_matrix2(vecs, np) # distance matrix
neighbors = dists.argsort()     # neighbors ordered by similarity
samples_per_item = 200

M = np.copy(dists)
M.sort(axis=1)

# Sample ranks are the neighbor indexes that are going to be selected
# We want to sample near neighbors much more than far neighbors
# We add one to skip the item itself
# sample_ranks = (np.random.exponential(0.05, N * samples_per_item * 5) * N).astype(int) + 1
# sample_ranks = np.random.randint(1, N, N * samples_per_item * 5)
sample_ranks = np.maximum(1, np.abs(np.random.normal(0, N/10, N * samples_per_item * 2)).astype(int))
# sample_ranks = (np.random.pareto(3, N * samples_per_item * 2) / 5 *  N).astype(int) + 1
sample_ranks = sample_ranks[np.where(sample_ranks < (N-1))]
print(sample_ranks)

# TODO: This should really be done as a batch...
A = np.zeros((samples_per_item * N, ndims))
B = np.zeros((samples_per_item * N, ndims))
D = np.zeros((samples_per_item * N, ))
P = np.zeros((samples_per_item * N, ))
R = np.zeros((samples_per_item * N, ), dtype=int)

for i in range(samples_per_item * N):
    r = sample_ranks[i]
    j = i % N
    k = neighbors[j][r]
    A[i] = vecs[j]
    B[i] = vecs[k]
    P[i] = vecP[j][k]
    D[i] = dists[j][k]
    if r <= 10:
        R[i] = 0
    elif R[i] < N * 0.1:
        R[i] = 1
    else:
        R[i] = 2

epochs = 500
batch_size = 1000

# network definition
base_network = create_base_network(ndims)

input_a = Input(shape=(ndims,))
input_b = Input(shape=(ndims,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(tsne_prob, output_shape=eucl_dist_output_shape, name='distance')([processed_a, processed_b])

model = Model([input_a, input_b], distance)

P *= 25

# train
model.compile(loss=joint_loss, optimizer=RMSprop())
for epoch in range(epochs):
    print('doing', epoch)
    # A, B, P, D, R = shuffle(A, B, P, D, R)

    Y = np.column_stack((P.T, D.T, R.T))
    loss = 0
    for i in xrange(0, A.shape[0], batch_size):
        loss += model.train_on_batch([A[i:i+batch_size], B[i:i+batch_size]], Y[i:i+batch_size])

    # write coordinates
    print('loss is', loss)
    coords = base_network.predict(vecs)
    print(embedTrustworthiness(pd.DataFrame(vecs), pd.DataFrame(coords), 10))
    print(neighborOverlap(pd.DataFrame(vecs), pd.DataFrame(coords), 10))

    if epoch == 5:
        P /= 25
