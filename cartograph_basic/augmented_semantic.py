"""
Create an augmented matrix with additional columns of labels to perform embedding, clustering and labeling on.
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import ast
from sklearn.decomposition import TruncatedSVD

input_dir = 'data/ext/simple'
output_dir = input_dir + '/GeneratedFiles'
sample_size = 500

allGraph = pd.read_table(input_dir + '/AllGraph_dict.sample_' + str(sample_size) + '.tsv', index_col='id')

# Find dimension of sparse matrix
catIndex = {}
i = 0
for id, row in allGraph.iterrows():
    catDict = ast.literal_eval(row['category'])
    for label in catDict.keys():
        if label not in catIndex.keys():
            catIndex[label] = i
            i += 1

numCol = len(catIndex.keys())
numRow = sample_size

# Create a matrix of proper format
mat = sp.dok_matrix((numRow, numCol), dtype=np.int64)
for id, row in allGraph.iterrows():
    catDict = ast.literal_eval(row['category'])
    for label, value in catDict.iteritems():
        mat[id-1, catIndex[label]] = value
mat = mat.transpose().tocsr()

# TruncatedSVD to reduce to dim 20
svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
svd.fit(mat)
truncatedLabels = svd.components_.T

# Normalize rows in truncatedLabels
# for i in range(truncatedLabels.shape[0]):
#     if truncatedLabels[i, :].sum() != 0:
#         truncatedLabels[i, :] = truncatedLabels[i, :]/truncatedLabels[i, :].sum()


vecs = pd.read_table(input_dir + '/vectors.sample_' + str(sample_size) + '.tsv', index_col=0, skiprows=1, header=None)
vecs.to_csv(input_dir + '/vectors_norm.sample_' + str(sample_size) + '.tsv', sep='\t')

# Normalize rows in vecs
# vecs = vecs.div(vecs.sum(axis=1), axis=0)

for i in range(truncatedLabels.shape[1]):
    vecs[len(vecs.columns)+1] = truncatedLabels[:, i]*10

vecs.to_csv(input_dir + '/vecs_augmented.sample_500.tsv', sep='\t')

