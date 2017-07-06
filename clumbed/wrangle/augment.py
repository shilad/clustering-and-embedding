import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from result_dir import ResultDir


def augment_label_svd(input_dir, output, label_dims=20, label_weight=0.2):
    """
    Create an augmented matrix with additional columns composed of an svd on labels.
    """

    if isinstance(output, ResultDir):
        output.log('Label dims is %d, label weight is %f' % (label_dims, label_weight))

    # Read in categories
    cat_df = pd.read_table(input_dir + '/categories.tsv', index_col='id')

    # Find dimension and one-hot encoding of sparse matrix
    cat_indexes = {}  # A dictionary of all categories with indices of categories.
    for id, row in cat_df.iterrows():
        for label in json.loads(row['category']):
            if label not in cat_indexes:
                cat_indexes[label] = len(cat_indexes)

    ncols = len(cat_indexes.keys())
    nrows = cat_df.shape[0]

    # Create a matrix of proper format
    mat = sp.dok_matrix((nrows, ncols), dtype=np.int64)
    for i, (id, row) in enumerate(cat_df.iterrows()):
        for label, value in json.loads(row['category']).items():
            mat[i, cat_indexes[label]] = value
    mat = mat.transpose().tocsr()

    # TruncatedSVD to reduce to dim 20
    svd = TruncatedSVD(n_components=label_dims, n_iter=7, random_state=42)
    svd.fit(mat)
    label_svds= svd.components_.T * label_weight

    # Read in vecs
    vecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

    # normalize and combine vecs
    vecs = vecs.div(np.linalg.norm(vecs, axis=1), axis=0)
    colnames = ['l_' + str(i) for i in range(label_dims)]
    label_svds = normalize(label_svds, axis=1, norm='l2') * label_weight
    label_df = pd.DataFrame(data=label_svds, columns=colnames, index=cat_df.index.tolist())

    merged = vecs.merge(label_df, how='left', left_index=True, right_index=True).fillna(0.0)
    merged.index.rename('id', inplace=True)
    merged.to_csv(str(output) + '/vectors_label.tsv', sep='\t')

    return merged


def augment_clusters(input_dir, output, k=10, clust_weight=0.25):
    """
    Create an augmented matrix with additional columns composed of one-hot kmeans indicators.
    """

    if isinstance(output, ResultDir):
        output.log('Num clusters is %d, cluster weight is %f' % (k, clust_weight))

    # Read in vectors
    vecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

    # Kmeans cluster and write out files
    clusters = KMeans(k).fit_predict(vecs)
    df = pd.DataFrame(data={ 'cluster' : clusters }, index=vecs.index)
    df.index.rename('id', inplace=True)
    df.to_csv(str(output) + '/cluster.tsv', sep='\t')

    # One-hot encode clusters and write out merged result
    cluster_df = pd.get_dummies(df, columns=['cluster'], prefix='c') * clust_weight
    merged = vecs.merge(cluster_df, how='left', left_index=True, right_index=True)
    merged.to_csv(str(output) + '/vectors_kmeans.tsv', sep='\t')

    return merged


def augment_everything(input_dir, output, k=10, label_dims=20, label_weight=0.2, clust_weight=0.25):
    """
    Create an augmented matrix with additional columns composed of one-hot kmeans indicators and a label svd.
    """

    # Make calls to augment clusters and labels
    df_clust = augment_clusters(input_dir, output, k, clust_weight)
    df_lab = augment_label_svd(input_dir, output, label_dims, label_weight)

    # remove original vector columns from cluster dataframe
    vec_columns =  [c for c in df_clust.columns.values if not str(c).startswith('c_')]
    df_clust = df_clust.drop(vec_columns, axis=1)

    # merge and write out
    merged = df_lab.merge(df_clust, how='left', left_index=True, right_index=True)
    merged.to_csv(str(output) + '/vectors_label_kmeans.tsv', sep='\t')
