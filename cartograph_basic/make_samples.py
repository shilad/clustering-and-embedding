import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf8')

input_dir = 'data/simple_all'


def make_dataset(input_dir, output_dir, sample_size):

    # Get the sample
    pops = pd.read_table(input_dir + '/popularity.tsv', index_col=0)
    pops = pops.sort_values('popularity', ascending=False)
    internal_ids = set(pops.index.tolist()[:sample_size])

    # Filter pops
    pops[pops.index.isin(internal_ids)].to_csv(output_dir + '/pop.tsv', sep='\t')

    # Filter ids
    ids = pd.read_table(input_dir + '/ids.tsv', index_col=0)
    pops[pops.index.isin(internal_ids)].to_csv(output_dir + '/ids.tsv', sep='\t')

    # Read in the category vector and clean it up

    # Match internal and external IDs
    colnames = [i for i in range(0, sample_size)]
    allGraph = pd.read_table(input_dir + '/AllGraph.tsv', names=colnames, error_bad_lines=False)
    allGraph['vectorTemp'] = allGraph.iloc[:, 1:].apply(lambda x: tuple(x), axis=1)  # join all vector columns into same column
    # drop all columns but the index and the vectorTemp column
    allGraph.drop(allGraph.columns[1:-1], axis=1, inplace=True)
    allGraph.columns = ['externalId', 'category']
    allGraph = allGraph.set_index('externalId')
    allGraph.index = allGraph.index.astype(str)

    # Change category vector to dictionary
    dictList = []
    for id, row in allGraph.iterrows():
            catList = row['category']
            aDict = {}
            for i in catList:
                if str(i) != 'nan':
                    cat, count = str(i).split(':')
                    aDict[cat] = count
            dictList.append(aDict)
    allGraph['category'] = dictList

    # Construct internal id column
    internalIds = []
    i = 0
    for id, row in allGraph.iterrows():
        # if int(id) < 100:
        if i % 1000 == 0: print('processing row', i)
        inId = ids[ids['externalId'] == id].index.values
        if len(inId) == 0:
            internalIds.append(None)
        else:
            internalIds.append(inId[0])
        i += 1

    allGraph['id'] = internalIds

    # Rearrange columns in AllGraph_dict
    # allGraphDict = pd.read_table(input_dir + '/AllGraph_dict.tsv', skip_blank_lines=True, skiprows=1, header=None)
    # allGraphDict.columns = ['id','externalId', 'category']
    # # allGraphDict = allGraphDict[['id', 'externalId', 'category']]
    # allGraphDict = allGraphDict.dropna().set_index('id')
    # allGraphDict.index = allGraphDict.index.astype(int)
    #
    # allGraphDict.to_csv(input_dir + '/AllGraph_dict.tsv', sep='\t', index_label='id', columns=['externalId', 'category'])
    #
    allGraph = pd.read_table(input_dir + '/AllGraph_dict.tsv', index_col='id')
    # print len(allGraph)
    #
    # ids = pd.read_table(input_dir + '/ids.tsv', index_col=0)
    # print ids.columns
    # allGraph.columns = ['exId', 'cat']
    #
    featureDict = pd.read_table(input_dir + '/vectors.sample_5000.tsv', skiprows=1, skip_blank_lines=True, header=None)
    featureDict['vectorTemp'] = featureDict.iloc[:, 1:].apply(lambda x: tuple(x),
                                                              axis=1)  # join all vector columns into same column
    featureDict.drop(featureDict.columns[1:-1], axis=1,
                     inplace=True)  # drop all columns but the index and the vectorTemp column
    featureDict.columns = ['index', 'vector']
    featureDict = featureDict.set_index('index')
    # featureDict.index = featureDict.index.astype(str)

    newdf = pd.DataFrame()
    newdf['id'] = featureDict.index
    listId = featureDict.index.tolist()
    # for id, row in featureDict.iterrows():
        # print allGraph.loc[id].values
    allGraph = allGraph.filter(items=listId, axis=0)

    allGraph.to_csv(input_dir + '/AllGraph_dict.sample_5000.tsv', sep='\t', index_label='id', columns=['externalId', 'category'])


make_dataset('data/simple_all', 'data/simple_500', 500)