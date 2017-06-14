import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tsne import bh_sne
import sys

reload(sys)
sys.setdefaultencoding('utf8')

input_dir = 'data/ext/simple'

sample_size = 500

vecs = pd.read_table(input_dir + '/vectors.sample_' + str(sample_size) + '.tsv', index_col=0, skiprows=1, header=None)

# Run t-SNE
out = bh_sne(vecs, pca_d=None, theta=0.5, perplexity=30)
# Clustering
clusters = list(KMeans(15).fit(out).labels_)

# Get names for plotting
names = pd.read_table(input_dir + '/names.tsv', index_col=0, skiprows=1, header=None)
name = []
k = 0  # For logging
for (id, row) in vecs.iterrows():
    if k % 1000 == 0: print "Found names of {}/{} vectors".format(k, len(vecs))
    name.append(str(names.loc[id][1]))
    k += 1

# Get popularity of nodes for plotting
# Display large labels for the most popular nodes overall and the most popular node in each cluster

popularity = pd.read_table(input_dir + '/popularity.tsv', index_col=0, skiprows=1, header=None)
vecs['cluster'] = clusters
vecs['name'] = name
vecs.sort_values('cluster', inplace=True)
popularity['label'] = [str(i) for i in names[1]]
popularity.sort_values(1, inplace=True, ascending=False)
order = [i + 1 for i in range(len(popularity))]
popularity['order'] = order

popularity['rank in cluster'] = [False] * len(popularity)

vecs.drop(vecs.columns[0:-2], axis=1,
          inplace=True)  # drop all columns but the cluster and name column

vecs.to_csv(input_dir + '/cluster_with_internalID.sample_' + str(sample_size) + '.tsv', sep='\t', index_label='index')
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
s = [2] * len(clusters)  # size of nodes
plt.scatter(out[:, 0], out[:, 1], s=s, c=clusters, alpha=0.5)

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

plt.savefig("test.svg", format="svg")
plt.show()
