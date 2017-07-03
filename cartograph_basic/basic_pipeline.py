import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tsne import bh_sne
import sys
import tfidf
import ast
import numpy as np

from metrics import countryQuality, embedTrustworthiness, neighborOverlap

reload(sys)
sys.setdefaultencoding('utf8')

input_dir = 'data/ext/simple'
output_dir = input_dir + '/GeneratedFiles'

sample_size = 500

vecs = pd.read_table(input_dir + '/vecs_augmented.sample_' + str(sample_size) + '.tsv', index_col=0, skiprows=1, header=None)

# Run t-SNE
out = bh_sne(vecs, pca_d=None, theta=0.5, perplexity=30)

# Clustering
# kMeans = KMeans(10).fit(out)
kMeans = KMeans(10).fit(vecs)
clusters = list(kMeans.labels_)
centroids = list(kMeans.cluster_centers_)


print('trustworthiness', embedTrustworthiness(vecs, out))
print('overlap', neighborOverlap(vecs, out))
print('country quality', countryQuality(out, clusters))

# Get names for plotting
names = pd.read_table(input_dir + '/names.tsv', index_col=0, skiprows=1, header=None)
name = []
k = 0  # For logging
for (id, row) in vecs.iterrows():
    if k % 1000 == 0: print "Found names of {}/{} vectors".format(k, len(vecs))
    name.append(str(names.loc[id][1]))
    k += 1

# Get externalIds
ids = pd.read_table(input_dir + '/ids.tsv', index_col=0)
ids = ids.astype(str)
ids.index = ids.index.astype(str)

# Save cluster file
vecs['cluster'] = clusters
vecs['name'] = name
vecs.sort_values('cluster', inplace=True)
vecs.drop(vecs.columns[0:-2], axis=1,
          inplace=True)  # drop all columns but the cluster and name column
vecs.to_csv(input_dir + '/cluster_with_internalID_augmented.sample_' + str(sample_size) + '.tsv', sep='\t', index_label='index')

# Run TF-IDF Labeling
categoryPath = input_dir + '/AllGraph_dict.sample_' + str(sample_size) + '.tsv'
clusterPath = input_dir + '/cluster_with_internalID_augmented.sample_' + str(sample_size) + '.tsv'
labelPath = output_dir + '/augmentedAllGraph_candidateLabels.sample_' + str(sample_size) + '.tsv'
tfidf.tfidf(categoryPath, clusterPath, labelPath)

# Get popularity of nodes for plotting
# Display large labels for the most popular nodes overall and the most popular node in each cluster

popularity = pd.read_table(input_dir + '/popularity.tsv', index_col=0, skiprows=1, header=None)

popularity['label'] = [str(i) for i in names[1]]
popularity.sort_values(1, inplace=True, ascending=False)
order = [i + 1 for i in range(len(popularity))]
popularity['order'] = order
popularity['rank in cluster'] = [False] * len(popularity)

# Find most popular node in each cluster to plot, might take a lot of time

# for i in vecs['cluster'].unique():
#     nameCluster = vecs.loc[vecs['cluster'] == i]['name'].values  # Get all names of nodes in a cluster
#     maxPop = len(popularity) + 1
#     for aName in nameCluster:
#         zpop = popularity.loc[popularity['label'] == aName]['order'].values[0]
#         if maxPop > zpop: maxPop = zpop
#     maxPopName = str(popularity.loc[popularity['order'] == maxPop]['label'].values[0])  # Most popular node in cluster
#     popularity.loc[popularity['label'] == maxPopName, 'rank in cluster'] = True

# Plot svg

# Helper function to make sure points are not on top of each other
def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
           verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin,
                       vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, hold=hold, **kwargs)


s = [2] * len(clusters)  # size of nodes
jitter(out[:, 0], out[:, 1], s=s, c=clusters, alpha=0.5)

for label, x, y in zip(name, out[:, 0], out[:, 1]):
    rank = popularity.loc[popularity['label'] == label]['order'].values
    # Choose different font sizes.
    if rank <= 60:
        size = 5 - int(rank / 20)
    elif popularity.loc[popularity['label'] == label]['rank in cluster'].values:
        size = 3
    elif rank <= 300:
        size = 1.5
    else:
        size = 1

    plt.text(x, y, label, horizontalalignment='center',
             verticalalignment='bottom', fontsize=size, color='black')

# Plot centroids and labels of cluster
xCentroids = [i[0] for i in centroids]
yCentroids = [i[1] for i in centroids]
jitter(xCentroids, yCentroids, s=6, c='r')

labels_df = pd.read_table(labelPath,
                          index_col='cluster')
labels_df.sort_index(inplace=True)
labelsList = [ast.literal_eval(i) for i in labels_df['labels']]
labels = [i[0].strip() for i in labelsList]

for label, x, y in zip(labels, xCentroids, yCentroids):
    plt.text(x, y, label, horizontalalignment='center',
             verticalalignment='bottom', fontsize=6, color='red')

plt.savefig("test.svg", format="svg")
plt.show()
