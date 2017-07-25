import sys
from sklearn.cluster import KMeans
from tsne import bh_sne

from clumbed.evaluate.metrics import *
from clumbed.evaluate.plot import *
from clumbed.wrangle.augment import *
from clumbed.wrangle.result_dir import ResultDir
from clumbed.tfidf import tfidf
from clumbed.hierarchical_clusters import agglomerativeHierarchicalClusters

# Handle non-ASCII characters
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.insert(0, '..')

# Prepare input and output directories
input_dir = 'data/simple_5000'
res = ResultDir()
res.log('Input dir: ' + input_dir)
res.log('Output dir: ' + res.get())

# Set parameters
k = 10
label_dims = 20
label_weight = 0.2
cluster_weight = 0.25
minPerCell = 5

# These are the vector embeddings we will consider.
VECTOR_CONFIG = {
    'plain': {
        'add_labels': False,
        'add_clusters': False,
    },
    'kmeans': {
        'add_labels': False,
        'add_clusters': True,

    },
    'labels_kmeans': {
        'add_labels': True,
        'add_clusters': True,

    },
    'labels': {
        'add_labels': True,
        'add_clusters': False,
    },
}

# Load original vectors as a reference
ovecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

for vname, vconfig in VECTOR_CONFIG.items():
    vpath = res.get() + '/vectors_' + vname + '.tsv'
    cpath = res.get() + '/clusters_' + vname + '.tsv'
    lpath = res.get() + '/labels_' + vname + '.tsv'

    # Start with raw vectors
    vecs_df = ovecs.copy()

    # Maybe add labels SVD to vecs
    if vconfig['add_labels']:
        vecs_df = augment_label_svd(input_dir, vecs_df, label_dims, label_weight)

    # Kmeans cluster; may or may not include labels SVD as columns
    clusters = KMeans(k).fit_predict(vecs_df)
    clusters_df = pd.DataFrame(data={'cluster': clusters}, index=vecs_df.index)
    clusters_df.index.rename('id', inplace=True)

    # Maybe add one-hot cluster columsn to vecs
    if vconfig['add_clusters']:
        vecs_df = augment_clusters(vecs_df, clusters_df, cluster_weight)

    assert (clusters.shape[0] == vecs_df.shape[0])
    N = vecs_df.shape[0]
    D = vecs_df.shape[1]

    # Create embedding
    xy = bh_sne(vecs_df.as_matrix(), pca_d=None)

    # Calculate tf-idf and find labels for clusters
    category_df = pd.read_table(input_dir + '/categories.tsv', index_col='id')
    label_df, tfidf_scores = tfidf(category_df, clusters_df)
    tfidf_scores = tfidf_scores.merge(clusters_df, left_index=True, right_index=True)
    cluster_to_label = { c : l[0] for (c, l) in zip(label_df.index, label_df['labels'])}

    # Write out vectors and clusters
    vecs_df.to_csv(vpath, sep='\t')
    clusters_df.to_csv(cpath, sep='\t')

    # Write out labels and names of articles in each cluster
    names_df = pd.read_table(input_dir + '/names.tsv', index_col=0)
    names_df = names_df.merge(clusters_df, left_index=True, right_index=True)
    articles = []
    for cluster in label_df.index:
        articles.append(names_df[names_df['cluster'] == cluster]['name'].values.tolist())
    label_df['articles'] = articles
    label_df.to_csv(lpath, sep='\t')

    # Log evaluation results and important parameters
    res.log_and_print('\nresults for ' + vpath)
    res.log_and_print('Parameters: k (num_clusters): %d, label_dim: %d, label_weight: %.2f, cluster_weight %.2f'
                      % (k, label_dims, label_weight, cluster_weight))
    res.log_and_print('trustworthiness: %f' % embedTrustworthiness(vecs_df, xy, k))
    res.log_and_print('overlap %f' % neighborOverlap(vecs_df, xy, k))
    res.log_and_print('orig-trustworthiness %f' % embedTrustworthiness(ovecs, xy, k))
    res.log_and_print('orig-overlap %f' % neighborOverlap(ovecs, xy, k))
    res.log_and_print('country quality %f' % countryQuality(xy, clusters_df))
    res.log_and_print('label metric %f' % labelMetric(tfidf_scores))

    # Plot the results and save the images
    names = pd.read_table(input_dir + '/names.tsv', index_col=0, skiprows=1, header=None)
    plt = plot(names, vecs_df.index, xy[:, 0], xy[:, 1], clusters_df['cluster'], cluster_to_label)
    plt.savefig(res.get() + '/plot_' + vname + '.svg', format="svg")
    # plt.show()
    plt.clf()  # Clear plot

    # TODO: Hierarchical

    data, hac_label_df, hac_clusters = agglomerativeHierarchicalClusters(vecs_df, names_df, category_df, 7)
    data.to_csv(res.get() + '/hac_data_' + vname + '.tsv', sep='\t')
    hac_label_df.to_csv(res.get() + '/hac_labels_' + vname + '.tsv', sep='\t')
    hac_clusters.to_csv(res.get() + '/hac_clusters_' + vname + '.tsv', sep='\t')
