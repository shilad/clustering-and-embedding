from collections import defaultdict
import pandas as pd
import sys
import ast


def tfidf(categoryPath, clusterPath , labelPath, tfidfPath=None):
    """
    Calculate tf-idf
    :param categoryPath: Path to data file with a dictionary of categories
    :param clusterPath: Path to cluster file
    :param tfidfPath: Path to save tf-idf scores output
    :param labelPath: Path to save candidate labels output
    :return:
    """
    # reload(sys)
    # sys.setdefaultencoding('utf8')

    allGraphDict = pd.read_table(categoryPath, index_col='id')
    allGraphDict.index = allGraphDict.index.astype(str)
    allGraphDict.sort_index(inplace=True)

    sampleCluster = pd.read_table(clusterPath, index_col='index')
    sampleCluster = sampleCluster.astype(str)
    sampleCluster.index = sampleCluster.index.astype(str)
    sampleCluster.sort_index(inplace=True)

    # Calculate TF-IDF
    # tf (how many time a term appears). idf(number of total docs / number of docs with the term)

    docScores = []  # Nested list of tf-idf scores per document
    catCounts = defaultdict(int)
    for i, (id, row) in enumerate(allGraphDict.iterrows()):
        catDict = ast.literal_eval(row['category'])
        for key in catDict:
            catCounts[key] += 1

    for i, (_, row) in enumerate(allGraphDict.iterrows()):
        if i % 100 == 0: print 'Calculated TF-IDF for {}/{}'.format(i, len(allGraphDict))
        catDict = ast.literal_eval(row['category'])
        for key, tf in catDict.items():
            df = catCounts[key]
            # catDict[key] = tf * math.log(len(allGraphDict) / df)
            catDict[key] = tf * (1.0 * len(allGraphDict) / df)
            # catDict[key] = tf * (1.0 * len(allGraphDict) / df) ** 0.5
        docScores.append(sorted(catDict.items(), key=lambda x: x[1], reverse=True))

    # Save tf-idf scores as a tsv
    score_df = pd.DataFrame()
    score_df['id'] = allGraphDict.index
    score_df.set_index('id', inplace=True)
    # score_df['name'] = sampleCluster['name']
    score_df['score'] = docScores
    score_df.to_csv(tfidfPath, sep='\t', index_label='id', columns=['score'])

    # Choose label for each cluster
    candidateLabel = []  # Nested array for best labels per cluster
    cluster = []
    for i in sampleCluster['cluster'].unique():
        cluster.append(i)
        idCluster = sampleCluster.loc[sampleCluster['cluster'] == i].index  # Get all id of nodes in a cluster
        totalLabel = {}
        for id in idCluster:
            if id in score_df.index:
                # Sum up tfidf scores for all articles in the  cluster, select labels with highest score
                for label in score_df.loc[id]['score']:
                    if label[0] in totalLabel.keys():
                        totalLabel[label[0]][0] += label[1]  # Sum up tfidf scores
                        totalLabel[label[0]][1] += 1  # Number of occurrences of this label
                    else:
                        totalLabel[label[0]] = [label[1]]
                        totalLabel[label[0]].append(1)
        totalLabel = sorted(totalLabel.items(), key=lambda x: x[1][1]*x[1][0], reverse=True)
        top5Label = [i[0] for i in totalLabel[:5]]
        candidateLabel.append(top5Label)

    label_df = pd.DataFrame({'cluster': cluster, 'labels': candidateLabel})
    label_df.set_index('cluster', inplace=True)
    label_df.sort_index(inplace=True)
    label_df.to_csv(labelPath, sep='\t', index_label='cluster', columns=['labels'])  # Cluster

if __name__ == "__main__":
    input_dir = 'data/ext/simple'
    output_dir = input_dir + '/GeneratedFiles'
    sample_size = 500

    categoryPath = input_dir + '/AllGraph_dict.sample_' + str(sample_size) + '.tsv'
    clusterPath = input_dir + '/hierarchical_clusters_withInternalID.sample_' + str(sample_size) + '.tsv'
    tfidfPath = output_dir + '/hierarchical_tfidf.sample_' + str(sample_size) + '.tsv'
    labelPath = output_dir + '/hierarchical_tfidf_candidateLabels.sample_' + str(sample_size) + '.tsv'

    import time
    start = time.time()
    tfidf(categoryPath, clusterPath, labelPath, tfidfPath)
    print("Time elapsed: ", time.time() - start)