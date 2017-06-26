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

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from numpy.linalg import norm

import utils



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


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

def euclidean_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    return K.mean(K.square(y_pred - y_true), axis=-1)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(20, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(2, activation='relu'))
    return seq

input_dir = './data/ext/simple'
sample_size= 500
df = pd.read_table(input_dir + '/vectors.sample_' + str(sample_size) + '.tsv', index_col=0, skiprows=1, header=None)

vecs = df.as_matrix()

# get dimensions and normalize to unit vector
assert vecs.shape[0] == sample_size
ndims = vecs.shape[1]
vecs = vecs / norm(vecs, axis=1).reshape(sample_size, 1)
ids = df.index.tolist()

cor = np.corrcoef(vecs)                 # correlation matrix
neighbors = np.fliplr(cor.argsort())    # neighbors ordered by similarity

samples_per_item = 100

# Sample ranks are the neighbor indexes that are going to be selected
# We add one to skip the item itself
sample_ranks = np.random.exponential(0.05, sample_size * samples_per_item * 5)
sample_ranks = (sample_ranks * sample_size).astype(int) + 1
sample_ranks = sample_ranks[np.where(sample_ranks < (sample_size-1))]

# TODO: This should really be done as a batch...
A = np.zeros((samples_per_item * sample_size, ndims))
B = np.zeros((samples_per_item * sample_size, ndims))
for i in range(samples_per_item * sample_size):
    r = sample_ranks[i]
    j = i % sample_size
    k = neighbors[j][r]
    A[i] = vecs[j]
    B[i] = vecs[k]

S = np.sum(A * B).T         # Cosine similarity
D = norm(A - B, axis=1)     # Euclidean distance

input_dim = 200
epochs = 5

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=euclidean_loss, optimizer=rms)
model.fit([A, B], D,
          batch_size=128,
          epochs=epochs)

coords = base_network.predict(vecs)

utils.plot(ids, coords[:,0], coords[:,1])

# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))