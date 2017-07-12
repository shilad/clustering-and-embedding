from collections import defaultdict
import pandas as pd
import ast
    
    
def tfidf(category_df, clusters_df, tfidf=None):
    """
    Calculate tf-idf
    :param category_df: A data frame with a dictionary of categories
    :param clusterPath: A data frame of clusters
    :return:
    """
    # tf (how many time a term appears). idf(number of total docs / number of docs with the term)

    docScores = []  # Nested list of tf-idf scores per document
    catCounts = defaultdict(int)
    for i, (id, row) in enumerate(category_df.iterrows()):
        catDict = ast.literal_eval(row['category'])
        for key in catDict:
            catCounts[key] += 1

    for i, (_, row) in enumerate(category_df.iterrows()):
        if i % 1000 == 0: print 'Calculated TF-IDF for {}/{}'.format(i, len(category_df))
        catDict = ast.literal_eval(row['category'])
        for key, tf in catDict.items():
            df = catCounts[key]
            # catDict[key] = tf * math.log(len(allGraphDict) / df)
            catDict[key] = tf * (1.0 * len(category_df) / df)
            # catDict[key] = tf * (1.0 * len(allGraphDict) / df) ** 0.5
        docScores.append(sorted(catDict.items(), key=lambda x: x[1], reverse=True))

    # Save tf-idf scores as a data frame
    score_df = pd.DataFrame({'id' : category_df.index, 'score': docScores})
    score_df.set_index('id', inplace=True)

    # Choose label for each cluster
    candidateLabel = []  # Nested array for best labels per cluster
    cluster = []
    for i in clusters_df['cluster'].unique():
        cluster.append(i)
        idCluster = clusters_df.loc[clusters_df['cluster'] == i].index  # Get all id of nodes in a cluster
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
    
    return label_df, score_df

if __name__ == "__main__":
    pass
