import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import ast
from collections import defaultdict, OrderedDict
import math

def mutualInformation(categories_df,cluster_df):
    """
    Calculate mutual information for each article
    See https://nlp.stanford.edu/IR-book/html/htmledition/cluster-labeling-1.html
    See https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html#sec:mutualinfo
    :param categories_df: A data frame with a dictionary of categories
    :param cluster_df: A data frame of clusters
    :return:
    """
    assert sorted(cluster_df.index)==sorted(categories_df.index)

    len_dict=defaultdict(int) #dictionary for each cluster length
    list_cluster = np.unique(cluster_df.loc[:,'cluster']) #list of unique cluster id

    for i in list_cluster:
        cluster_ids = cluster_df.index[cluster_df['cluster']==i]
        len_dict[i] = len(cluster_ids)


    N = len(cluster_df) #length of data set

    # Calculate N11 for each cluster
    # N11[i] is dictionary that counts the number of articles, within the cluster i^th, that belong to a given category.
    # The keys are the categories and the value is the counter.
    N11 = defaultdict(lambda: defaultdict(int))
    for i in list_cluster:
        cluster_ids = cluster_df.index[cluster_df['cluster']==i]
        #dict stores counts the number of times a category occurs in the i^th cluster
        dict = defaultdict(int)
        for id in cluster_ids:
                catdict = ast.literal_eval(categories_df.loc[id,'category'])

                #iterates over category dictionary for each article to count the number of times a category occurs
                for cats in catdict:
                    dict[cats] +=1
        #Stores dict in N11
        N11[i] = dict

    # Calculate N01
    # N10[i] is a dictionary that counts the number of articles that belong to a category but fall outside of the i^th cluster.
    # The keys are the categories and the value is the counter.
    N10 = defaultdict(lambda: defaultdict(int))
    for i in list_cluster:
        dict=defaultdict(int)
        for cats in N11[i]:
            for j, cat_dict in N11.iteritems():
                if not j == i:
                    #Iterates over N11[j] for all j clusters except the i^th clyster
                    if cats in cat_dict:
                        #sums all articles that belong to the cat for the j^th cluster
                        dict[cats]+=cat_dict[cats]
                    else:
                        dict[cats]+=0
        N10[i] = dict

    # Calculate N10 for each cluster
    # N01[i] is a dictionary that counts the number of articles within the i^th cluster that do not belong to a given category
    # The keys are the categories and the value is the counter.
    N01 = defaultdict(lambda: defaultdict(int))
    for i in list_cluster:
        dict = defaultdict(int)
        dict_N11 = N11[i]
        for cat in N11[i]:
            # dict_N11[cat] is the number of articles that belong to category, 'cat'
            # Length of cluster - dict_N11[cat] = the number of articles that do not belong to category, 'cat'
            dict[cat] = len_dict[i] - dict_N11[cat]
        N01[i] = dict

    # Calculate N00
    # N00[i] is a dictionary that counts the number of articles that do not belong to a given category and fall outside of the i^th cluster.
    # The keys are the categories and the value is the counter.
    N00 = defaultdict(lambda : defaultdict(int))
    for i in list_cluster:
        dict = defaultdict(int)
        for cat, counter in N10[i].iteritems():
                #N - len_dict[i] i = number of articles not in the i^th cluster
                #counter = number of articles that belong to the cat category and are not in the i^th cluster. (See N01[i])
                #  N - len_dict[i] - counter = number of articles that dont belog to the cat category and are not in the i^th cluster
                dict[cat] = N - len_dict[i] - counter
        N00[i] = dict


    #Calculates mutual information
    score_dict = defaultdict(lambda: defaultdict(int))
    for i in list_cluster:
        categories = N11[i]
        dict = defaultdict(int)
        for cats in categories:
                dict_N11 = N11[i]
                dict_N01 = N01[i]
                dict_N10 = N10[i]
                dict_N00 = N00[i]

                N1_ = dict_N11[cats] + dict_N10[cats] #Number of articles that belong to cats category
                N_1 = dict_N11[cats] + dict_N01[cats] #Number of articles that are contained in the i^th cluster
                N0_ = dict_N01[cats] + dict_N00[cats] #Number of articles that do not belong to cats category
                N_0 = dict_N10[cats] + dict_N00[cats] #Number of articles that are not contained in the i^th cluster


                if not dict_N11[cats] == 0:
                    dict[cats] += (1.0/N)*dict_N11[cats]*math.log(1.0 * N*dict_N11[cats] / (N1_*N_1))
                if not dict_N01[cats] == 0:
                    dict[cats] += (1.0/N)*dict_N01[cats]*math.log(1.0 * N*dict_N01[cats] / (N0_*N_1))
                if not dict_N10[cats] == 0:
                    dict[cats] += (1.0/N)*dict_N10[cats]*math.log(1.0 * N*dict_N10[cats] / (N1_*N_0))
                if not dict_N00[cats] == 0:
                    dict[cats] += (1.0/N)*dict_N00[cats]*math.log(1.0 * N*dict_N00[cats] / (N0_*N_0))

        score_dict[i]= dict

        score_df = pd.DataFrame(categories_df.index, columns=['id'])

    #top 5 candidate labels
    for i in categories_df.index:
        cluster_id = cluster_df.loc[i,'cluster']
        catDict= ast.literal_eval(categories_df.loc[i,'category'])
        for cat in catDict:
            catDict[cat] = score_dict[cluster_id].get(cat)
        score_df.loc[score_df.loc[:,'id'] == i,'labels'] = str(catDict)
    score_df.set_index('id')
    label_df = pd.DataFrame([[k,sorted(dict, key=dict.get, reverse=True)[0:5]] for k, dict in score_dict.iteritems()], columns=['cluster', 'top5names'])
    label_df.set_index('cluster')

    return label_df,score_df


size = 50000
categories_df = pd.read_table('data/simple_'+str(size)+'/categories.tsv', index_col='id')
#vectors = pd.read_table('data/simple_'+str(size)+'/vectors.tsv',index_col='id')
#names = pd.read_table('data/simple_'+str(size)+'/names.tsv',index_col='id')
cluster_df = pd.read_table('data/simple_'+str(size)+'/cluster_names_df.tsv',index_col='id')

#kmeans = KMeans(200).fit(vectors)
#data={'cluster': kmeans.labels_,'id': vectors.index}
#cluster_df= pd.DataFrame(data,columns=['id','cluster'])
#cluster_df=cluster_df.set_index('id')
#cluster_df = cluster_df.loc[categories_df.index]

label_df, score_df  = mutualInformation(categories_df,cluster_df)

print label_df

#cluster_names_df = (pd.merge(cluster_df.reset_index(),names.reset_index(),left_on='id',right_on='id',left_index=True)).sort_values('cluster')
#cluster_names_df.set_index('id')
#cluster_names_df.to_csv('data/simple_'+str(size)+'/cluster_names_df.tsv',sep='\t',index='id')

label_df.to_csv('data/simple_'+str(size)+'/cluster_labels_mutualinfo.tsv',sep='\t',index='cluster')


