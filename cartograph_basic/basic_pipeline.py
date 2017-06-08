import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tsne import bh_sne
import sys

reload(sys)
sys.setdefaultencoding('utf8')

input_dir = 'data/ext/simple'

vecs = pd.read_table(input_dir + '/vectors.sample_5000.tsv', index_col=0, skiprows=1, header=None)

# Run t-SNE
out = bh_sne(vecs, pca_d=None, theta=0.5, perplexity=30)
# Clustering
clusters = list(KMeans(15).fit(out).labels_)

# Get names for plotting
names = pd.read_table(input_dir + '/names.tsv', index_col=0, skiprows=1, header=None)
name = []
k = 0  # For logging
for (id, row) in vecs.iterrows():
    if k % 100 == 0: print "Found names of {}/{} vectors".format(k, len(vecs))
    name.append(str(names.loc[id][1]))
    k += 1

# Get popularity of nodes for plotting
# Display large labels for the most popular nodes overall and the most popular node in each cluster

popularity = pd.read_table(input_dir + '/popularity.tsv', index_col=0, skiprows=1, header=None)
vecs['cluster'] = clusters
vecs['name'] = name
popularity['label'] = [str(i) for i in names[1]]
popularity.sort_values(1, inplace=True, ascending=False)
order = [i + 1 for i in range(len(popularity))]
popularity['order'] = order

popularity['rank in cluster'] = [False] * len(popularity)
for i in vecs['cluster'].unique():
    nameCluster = vecs.loc[vecs['cluster'] == i]['name'].values  # Get all names of nodes in a cluster
    maxPop = len(popularity) + 1
    for aName in nameCluster:
        zpop = popularity.loc[popularity['label'] == aName]['order'].values[0]
        if maxPop > zpop: maxPop = zpop
    maxPopName = str(popularity.loc[popularity['order'] == maxPop]['label'].values[0])  # Most popular node in cluster
    popularity.loc[popularity['label'] == maxPopName, 'rank in cluster'] = True

# Plot
s = [0.5] * len(clusters)  # size of nodes
plt.scatter(out[:, 0], out[:, 1], s=s, c=clusters, alpha = 0.2)

for label, x, y in zip(name, out[:, 0], out[:, 1]):
    rank = popularity.loc[popularity['label'] == label]['order'].values
    if rank <= 60:
        size = 5 - int(rank / 20)
    elif popularity.loc[popularity['label'] == label]['rank in cluster'].values:
        size = 3
    elif rank <= 300:
        size = 1
    else:
        size = 0.7

    plt.text(x, y, label, horizontalalignment='center',
             verticalalignment='bottom', fontsize=size, color='black')

plt.savefig("test.svg", format="svg")
plt.show()
