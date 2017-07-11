import sys
from collections import defaultdict

from sklearn.cluster import KMeans

sys.path.insert(0, '..')

from tsne import bh_sne

from clumbed.evaluate.metrics import *
from clumbed.wrangle.augment import *
from clumbed.wrangle.result_dir import ResultDir
from clumbed.tfidf import tfidf


# Handle non-ASCII characters
reload(sys)
sys.setdefaultencoding('utf8')

# Prepare input and output directories
input_dir = 'data/simple_5000'
k = 10
res = ResultDir()
res.log('Input dir: ' + input_dir)
res.log('Output dir: ' + res.get())

# These are the vector embeddings we will consider.
VECTOR_CONFIG = {
    'plain' : {
        'add_labels' : False,
        'add_clusters' : False,
    },
    'kmeans' : {
        'add_labels' : False,
        'add_clusters' : True,

    },
    'labels_kmeans' : {
        'add_labels' : True,
        'add_clusters' : True,

    },
    'labels' : {
        'add_labels' : True,
        'add_clusters' : False,
    },
}
# VECTOR_NAMES = [
#     input_dir + '/vectors.tsv']

# Load original vectors as a reference
ovecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

for vname, vconfig in VECTOR_CONFIG.items():
    vpath = res.get() + '/vectors_' + vname + '.tsv'
    cpath = res.get() + '/clusters_' + vname + '.tsv'

    # Start with raw vectors
    vecs_df = ovecs.copy()

    # Maybe add labels SVD to vecs
    if vconfig['add_labels']:
        vecs_df = augment_label_svd(input_dir, vecs_df)

    # Kmeans cluster; may or may not include labels SVD as columns
    clusters = KMeans(k).fit_predict(vecs_df)
    clusters_df = pd.DataFrame(data={'cluster' : clusters}, index=vecs_df.index)
    clusters_df.index.rename('id', inplace=True)

    # Maybe add one-hot cluster columsn to vecs
    if vconfig['add_clusters']:
        vecs_df = augment_clusters(vecs_df, clusters_df)

    assert(clusters.shape[0] == vecs_df.shape[0])
    N = vecs_df.shape[0]
    D = vecs_df.shape[1]

    # Create embedding
    xy = bh_sne(vecs_df.as_matrix(), pca_d=None)

    # Write out vectors and clusters
    vecs_df.to_csv(vpath, sep='\t')
    clusters_df.to_csv(cpath, sep='\t')

    # Log evaluation results
    res.log_and_print('\nresults for ' + vpath)
    res.log_and_print('trustworthiness: %f' % embedTrustworthiness(vecs_df,xy))
    res.log_and_print('overlap %f' % neighborOverlap(vecs_df, xy))
    res.log_and_print('orig-trustworthiness %f' % embedTrustworthiness(ovecs,xy))
    res.log_and_print('orig-overlap %f' % neighborOverlap(ovecs, xy))
    res.log_and_print('country quality %f' % countryQuality(xy, clusters_df))

    # Label metric
    categoryPath = input_dir + '/categories.tsv'
    tfidfPath = res.get() + '/tfidfScores_' + vname + '.tsv'
    labelPath = res.get() + '/labels_' + vname + '.tsv'
    tfidf(categoryPath, cpath, labelPath, tfidfPath=tfidfPath)

    # Average tfidf score per cluster
    tfidf_scores = pd.read_table(tfidfPath, index_col='id')
    tfidf_scores = tfidf_scores.merge(clusters_df, left_index=True, right_index=True)

    import ast
    import numpy as np
    stat = defaultdict(list)
    for id, row in tfidf_scores.iterrows():
        scores = ast.literal_eval(row['score'])
        stat[row['cluster']].append(np.mean([x[1] for x in scores]))
    res.log_and_print('label metric %f' % np.mean([np.mean(stat[i]) for i in stat]))


    # TODO: Plot the results and save the images
