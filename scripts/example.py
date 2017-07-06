import sys

sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from tsne import bh_sne

from zalimar.evaluate.metrics import *

reload(sys)
sys.setdefaultencoding('utf8')


input_dir = 'data/simple_5000'

VECTOR_NAMES = [
    'vectors.tsv',
    'vectors_kmeans_10.tsv',
    'vectors_label_20_kmeans_10.tsv',
    'vectors_label_20.tsv'
]

# Load original vectors:
ovecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

for vfname in VECTOR_NAMES:
    print('evaluting ', vfname)
    vecs = pd.read_table(input_dir + '/' + vfname, index_col=0, skiprows=1, header=None)
    clusters = pd.read_table(input_dir + '/cluster_10.tsv', index_col=0)
    assert(clusters.shape[0] == vecs.shape[0])
    N = vecs.shape[0]
    D = vecs.shape[1]

    # Run t-SNE
    xy = bh_sne(vecs, pca_d=None)

    # Merge datasets on index to get clusters ordered consistent with vectors
    clusters = vecs.merge(clusters, left_index=True, right_index=True)['cluster']
    assert(clusters.shape[0] == N)

    print('\nresults for ' + vfname)
    print('trustworthiness', embedTrustworthiness(vecs,xy))
    print('overlap', neighborOverlap(vecs, xy))
    print('orig-trustworthiness', embedTrustworthiness(ovecs,xy))
    print('orig-overlap', neighborOverlap(ovecs, xy))
    print('country quality', countryQuality(xy, clusters))
    print()