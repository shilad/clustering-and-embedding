import sys

sys.path.insert(0, '..')

from tsne import bh_sne

from clumbed.evaluate.metrics import *
from clumbed.wrangle.augment import augment_everything
from clumbed.wrangle.result_dir import ResultDir


# Handle non-ASCII characters
reload(sys)
sys.setdefaultencoding('utf8')

# Prepare input and output directories
input_dir = 'data/simple_500'
res = ResultDir()
res.log('Input dir: ' + input_dir)
res.log('Output dir: ' + res.get())
augment_everything(input_dir, res)

# These are the vector embeddings we will consider.
VECTOR_NAMES = [
    input_dir + '/vectors.tsv',
    res.get() + '/vectors_kmeans.tsv',
    res.get() + '/vectors_label_kmeans.tsv',
    res.get() + '/vectors_label.tsv'
]

# Load original vectors as a reference
ovecs = pd.read_table(input_dir + '/vectors.tsv', index_col=0, skiprows=1, header=None)

for vfname in VECTOR_NAMES:
    print('evaluting ', vfname)
    vecs = pd.read_table(vfname, index_col=0, skiprows=1, header=None)
    clusters = pd.read_table(res.get() + '/cluster.tsv', index_col=0)
    assert(clusters.shape[0] == vecs.shape[0])
    N = vecs.shape[0]
    D = vecs.shape[1]

    # Run t-SNE
    xy = bh_sne(vecs, pca_d=None)

    # Merge datasets on index to get clusters ordered consistent with vectors
    clusters = vecs.merge(clusters, left_index=True, right_index=True)['cluster']
    assert(clusters.shape[0] == N)

    # Log evaluation results
    res.log_and_print('\nresults for ' + vfname)
    res.log_and_print('trustworthiness: %f' % embedTrustworthiness(vecs,xy))
    res.log_and_print('overlap %f' % neighborOverlap(vecs, xy))
    res.log_and_print('orig-trustworthiness %f' % embedTrustworthiness(ovecs,xy))
    res.log_and_print('orig-overlap %f' % neighborOverlap(ovecs, xy))
    res.log_and_print('country quality %f' % countryQuality(xy, clusters))


    # TODO: Plot the results and save the images
