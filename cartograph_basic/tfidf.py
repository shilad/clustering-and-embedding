import pandas as pd
import sys
import ast
import math

def run ():
    reload(sys)
    sys.setdefaultencoding('utf8')
    # #
    input_dir = 'data/ext/simple'
    output_dir = input_dir + '/GeneratedFiles'
    allGraphDict = pd.read_table(input_dir + '/AllGraph_dict.sample_500.tsv', index_col='id')
    allGraphDict.index = allGraphDict.index.astype(str)
    allGraphDict.sort_index(inplace=True)

    sampleCluster = pd.read_table(input_dir + '/cluster_with_internalID.sample_500.tsv', index_col='index')
    sampleCluster = sampleCluster.astype(str)
    sampleCluster.index = sampleCluster.index.astype(str)
    sampleCluster.sort_index(inplace=True)

    # Calculate TF-IDF
    # tf (how many time a term appears). idf(number of total docs / number of docs with the term)

    scoreList = []
    catList = []
    for i, (id, row) in enumerate(allGraphDict.iterrows()):
        catDict = ast.literal_eval(row['category'])
        for key in catDict.keys():
            catList.append(key)

    for i, (_, row) in enumerate(allGraphDict.iterrows()):
        if i % 100 == 0: print '{}/{}'.format(i, len(allGraphDict))
        catDict = ast.literal_eval(row['category'])
        for key in catDict.keys():
            docFreq = catList.count(key)
            score = catDict[key] * math.log(len(allGraphDict) / docFreq)
            catDict[key] = score
        scoreList.append(sorted(catDict.items(), key=lambda x: x[1], reverse=True))

    score_df = pd.DataFrame()
    score_df['id'] = allGraphDict.index
    score_df.set_index('id', inplace=True)
    score_df['name'] = sampleCluster['name']
    score_df['score'] = scoreList

    score_df.to_csv(output_dir + '/AllGraph_tfidf.sample_500.tsv', sep='\t', index_label='id',
                    columns=['name', 'score'])  # Cluster

    # Choose label for each cluster
    candidateLabel = []
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
                        totalLabel[label[0]][1] += 1 # Number of occurrences of this label
                    else:
                        totalLabel[label[0]] = [label[1]]
                        totalLabel[label[0]].append(1)
        # print type(totalLabel.items()), totalLabel.items()
        # totalLabel = sorted(totalLabel.items(), key=lambda x: x[1], reverse=True)  # Sort by tfidf sums
        totalLabel = sorted(totalLabel.items(), key=lambda x: x[1][1]*x[1][0], reverse=True)
        top5Label = [i[0] for i in totalLabel[:5]]
        candidateLabel.append(top5Label)

    label_df = pd.DataFrame()
    label_df['cluster'] = cluster
    label_df.set_index('cluster', inplace=True)
    label_df['labels'] = candidateLabel
    label_df.sort_index(inplace=True)
    label_df.to_csv(output_dir + '/AllGraph_tfidf_candidateLabels.sample_500.tsv', sep='\t', index_label='cluster',
                    columns=['labels'])  # Cluster