# Hierarchical agglomerative clustering
from collections import defaultdict
import pandas as pd
import ast
import numpy as np
import itertools
from sklearn.cluster import AgglomerativeClustering
import time
from clumbed.tfidf import tfidf


def agglomerativeHierarchicalClusters(vecs_df, names_df, category_df, numCluster):
    # Get names of articles
    nameList = []
    for (id, row) in vecs_df.iterrows():
        nameList.append(str(names_df.loc[id]['name']))


    # Build hierarchical agglomerative clustering
    model = AgglomerativeClustering(n_clusters=numCluster)
    model.fit(vecs_df)
    label = model.labels_

    # Save the hierarchy to a data frame
    # ii = itertools.count(max(vecs_df.index) + 1)
    ii = itertools.count(len(vecs_df) + 1)
    cluster = [next(ii) for x in model.children_]
    children = [(x + 1).tolist() for x in model.children_]
    hac_df = pd.DataFrame({'cluster': cluster, 'children': children}, index=cluster)
    hac_df.sort_index(inplace=True)

    # Create a dictionary representing the hierarchy
    ii = itertools.count(len(vecs_df) + 1)
    hierarchyDict = {}
    for x in model.children_:
        y = {x[0] + 1: None, x[1] + 1: None}  # Clustering id starts from 0, add 1 to be consistent with names.tsv
        if x[0] < len(nameList):
            y[x[0] + 1] = nameList[x[0]]
        if x[1] < len(nameList):
            y[x[1] + 1] = nameList[x[1]]
        hierarchyDict[next(ii)] = y

    finish = False
    while not finish:
        finish = True
        for key in hierarchyDict:
            for val in hierarchyDict[key]:
                if val > len(nameList) and not hierarchyDict[key][val]:
                    finish = False
                    hierarchyDict[key][val] = hierarchyDict[val]

    # Create a dictionary listing all articles in clusters (not represent the hierarchy)
    def recursive_items(dictionary):
        # A helper function to traverse to leaf nodes in a nested dictionary
        for key, value in dictionary.items():
            if type(value) is dict:
                for val in recursive_items(value):
                    yield val
            else:
                yield (key, value)

    leafDict = {}  # A dict with leaf nodes per cluster
    for key in hierarchyDict:
        aDict = {}
        for k, v in recursive_items(hierarchyDict[key]):
            aDict[k] = v
            leafDict[key] = aDict

    # Create a cluster dataframe with article ids
    index = []
    cluster = []
    name = []
    for k, v in leafDict.iteritems():
        for i in v:
            index.append(vecs_df.index[i-1])
            cluster.append(k)
            name.append(v[i])
    clusters_df = pd.DataFrame({'index': index, 'cluster': cluster, 'name': name})
    clusters_df.set_index('index', inplace=True)
    clusters_df.sort_values('cluster', inplace=True)

    data, label_df = createClusterData(hierarchyDict, leafDict, vecs_df, names_df, category_df, clusters_df)
    hac_final = prune(label_df, data, hierarchyDict, len(vecs_df))
    return data, label_df, hac_final


def createClusterData(nestedDict, leafDict, vecs_df, names_df, category_df, clusters_df):
    """
    Create a tsv file with all information about the hierarchical cluster
    :param nestedDict: A json-like dictionary that represents hierarchy of clusters. {999 : {998 : {a}, 997 : {b}}
    :param leafDict: A dictionary listing all leaf nodes (articles) in a cluster { 999: [a,b]}
    :return:
    """
    sample_size = len(vecs_df)
    cluster = []
    child1 = []
    child2 = []
    size = []
    child1size = []
    child2size = []

    # Calculate size of clusters and their children
    for key in leafDict:
        count = {}
        for i in nestedDict[key]:
            count[i] = len(leafDict[i]) if i > sample_size else 1
        cluster.append(key)
        size.append(len(leafDict[key]))
        child1_ = nestedDict[key].keys()[0]
        child2_ = nestedDict[key].keys()[1]
        child1.append(child1_)
        child2.append(child2_)
        child1size.append(len(leafDict[child1_]) if child1_ > sample_size else 1)
        child2size.append(len(leafDict[child2_]) if child2_ > sample_size else 1)


    # Calculate cohesion values of clusters and their children
    cohesionList = []
    cohesion = {}
    child1cohesion = []
    child2cohesion = []
    # Cluster cohesion: Average distance from points to centroids
    from scipy.spatial.distance import cdist
    for i in cluster:
        # vecInCluster = leafDict[i].keys()
        vecInCluster = [j-1 for j in leafDict[i]]
        vecInCluster = vecs_df.iloc[vecInCluster]
        centroid = vecInCluster.mean(axis=0)
        dist = cdist(vecInCluster.as_matrix(), [centroid.tolist(), ])
        cohesionList.append(np.sum(dist) / len(vecInCluster))
        cohesion[i] = np.sum(dist) / len(vecInCluster)

    for i in child1:
        if i > sample_size:
            child1cohesion.append(cohesion[i])
        else:
            child1cohesion.append(1)

    for i in child2:
        if i > sample_size:
            child2cohesion.append(cohesion[i])
        else:
            child2cohesion.append(1)

    # child1cohesion.append(cohesion[i] if i > sample_size else 1 for i in child1)
    # child2cohesion.append(cohesion[i] if i > sample_size else 1 for i in child2)

    # Find all articles in child clusters
    children1articles = []
    children2articles = []
    for i in child1:
        children1articles.append(
            [str(names_df.iloc[j-1]['name']) for j in leafDict[i]] if i > sample_size else [str(names_df.iloc[i-1]['name'])])
    for i in child2:
        children2articles.append(
            [str(names_df.iloc[j-1]['name']) for j in leafDict[i]] if i > sample_size else [str(names_df.iloc[i-1]['name'])])

    # Run tf-idf to find label for each cluster
    label_df, _ = tfidf(category_df, clusters_df)

    labels = []
    child1labels = []
    child2labels = []

    for i in cluster:
        labels.append(label_df.loc[i]['labels'])
    for i in child1:
        child1labels.append(label_df.loc[i]['labels'] if i > sample_size else [str(names_df.iloc[i-1]['name'])])
    for i in child2:
        child2labels.append(label_df.loc[i]['labels'] if i > sample_size else [str(names_df.iloc[i-1]['name'])])

    # Save data
    data = pd.DataFrame({'cluster': cluster, 'size': size, 'child2': child2,
                         'child1': child1, 'child1size': child1size, 'child2size': child2size,
                         'cohesion': cohesionList, 'child1cohesion': child1cohesion, 'child2cohesion': child2cohesion,
                         'label': labels, 'child1label': child1labels, 'child2label': child2labels,
                         'child1articles': children1articles, 'child2articles': children2articles})
    data.set_index('cluster', inplace=True)
    data.sort_index(inplace=True)
    return data, label_df
    # data.to_csv(output_dir + '/hierarchical_clusters_data.sample_' + str(sample_size) + '.tsv', sep='\t',
    #             index_label='cluster', columns=['child1', 'child2', 'size', 'child1size', 'child2size', 'cohesion',
    #                                             'child1cohesion', 'child2cohesion', 'label', 'child1label',
    #                                             'child2label', 'child1articles', 'child2articles'])


def prune(label_df, data, nestedDict, sample_size, num_level=3):

    keep = []  # A boolean array indicating whether to keep a cluster / to merge the child clusters
    prunedChildren = {}
    # Based on size (Size difference is not too much)
    keepSize = []
    for cluster, row in data.iterrows():

        sizeDiff = [i * 100.0 / row['size'] for i in (row['child1size'], row['child2size'])]
        keepSize.append(any(i < 30 for i in sizeDiff))

        # TODO: Break down children that have large size
        goodChildren = []
        child1size, child2size = row['child1size'], row['child2size']
        child1, child2 = row['child1'], row['child2']
        aList = [child1size, child2size]
        meanSize = np.mean(aList)
        while max(child1size, child2size) * 1.0 / meanSize > 1:
            childToBreak = child1 if child1size > child2size else child2
            goodChildren.append(child1 if child1size < child2size else child2)
            if childToBreak > sample_size:
                row = data.loc[childToBreak]
                child1size, child2size = row['child1size'], row['child2size']
                child1, child2 = row['child1'], row['child2']
                aList.extend([child1size, child2size])
                meanSize = np.mean(aList)
        goodChildren.extend([child1, child2])
        prunedChildren[cluster] = goodChildren

    # Find 3 levels in hierarchy:
    currentLevel = [data.index[-1]]
    clusterPerLevel = defaultdict(list)
    clusterList = {}
    k = 0

    while k <= num_level:
        for i in currentLevel:
            clusterPerLevel[k].append(i)
            clusterList[i] = prunedChildren[i]
        currentLevel = [i for sublist in [prunedChildren[j] for j in currentLevel] for i in sublist]
        k += 1

    clusterList.pop(data.index[-1], None)

    clusterPerLevel.pop(0, None)  # Delete the highest level (which contains all articles)
    
    # Prune clusters based on label
    clusterFinal = []
    childClusterFinal = []
    labelFinal = []
    childLabelFinal = []

    for cluster in clusterList:
        parentLabel = label_df.loc[cluster]['labels']
        childLabels = {}
        overlapPercent = 0  # Percentage of overlapping labels between child and parent
        overlapFirstChoice = 0  # Overlapping first choice of label between child and parent
        for child in prunedChildren[cluster]:
            if child in label_df.index:
                childLabels[child] = label_df.loc[child]['labels']
                overlapPercent += len([w for w in childLabels[child] if w in parentLabel]) * 1.0 / 5
                if childLabels[child][0] == parentLabel[0]:
                    overlapFirstChoice += 1
        if len(childLabels) > 0:
            overlapPercent /= len(childLabels)
            if overlapPercent < 0.8 or overlapFirstChoice*1.0/len(prunedChildren[cluster]) < 0.8:
                # Keep all children
                clusterFinal.append(cluster)
                childClusterFinal.append(prunedChildren[cluster])
                labelFinal.append(parentLabel[0])
                childLabelFinal.append([childLabels[i][0] for i in childLabels])
            else:
                # Do not keep children
                clusterFinal.append(cluster)
                labelFinal.append(parentLabel[0])
                childClusterFinal.append([])
                childLabelFinal.append([])

    hac_final = pd.DataFrame({'cluster': clusterFinal, 'label': labelFinal, 'children': childClusterFinal, 'childrenLabels': childLabelFinal})
    hac_final.set_index('cluster', inplace=True)
    hac_final.sort_index(inplace=True)

    # Check if 3 levels include all articles
    num = 0
    for cluster in clusterList:
        for child in prunedChildren[cluster]:
            if child > sample_size:
                num += len(nestedDict[child])
            else:
                num += 1

    print num, sample_size
    # assert num == sample_size
    
    return hac_final