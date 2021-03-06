import json
import os

import pandas as pd
import sys

from tsne import bh_sne


def make_dataset(input_dir, output_dir, sample_size):
    """
    Extracts a sample of a full dataset of a certain size. The most popular articles are selected.
    :param input_dir:
    :param output_dir:
    :param sample_size:
    :return:
    """

    # Get the sample
    pops = pd.read_table(input_dir + '/popularity.tsv', index_col=0)
    pops = pops.sort_values('popularity', ascending=False)
    internal_ids = set(pops.index.tolist()[:sample_size])

    # write out popularities
    pops = pops[pops.index.isin(internal_ids)]
    pops.to_csv(output_dir + '/pop.tsv', sep='\t', index_label='id')

    # Filter ids
    ids = pd.read_table(input_dir + '/ids.tsv', index_col=0)
    ids = ids[ids.index.isin(internal_ids)]
    ids.to_csv(output_dir + '/ids.tsv', sep='\t', index_label='id')
    ext_to_internal = dict(zip(ids['externalId'], ids.index))

    # Filter names
    names = pd.read_table(input_dir + '/names.tsv', index_col=0)
    names = names[names.index.isin(internal_ids)]
    names.to_csv(output_dir + '/names.tsv', sep='\t', index_label='id')

    # Filter vectors
    vecs = pd.read_table(input_dir + '/vectors.tsv', skiprows=1, skip_blank_lines=True, header=None, index_col=0)
    vecs = vecs[vecs.index.isin(internal_ids)]
    vecs.to_csv(output_dir + '/vectors.tsv', sep='\t', index_label='id')

    # Read in the category vector and clean it up
    # Match internal and external IDs and replace them
    colnames = ['externalId'] + list(range(300))
    categories = pd.read_table(input_dir + '/categories.tsv', names=colnames, error_bad_lines=False)
    categories = categories[categories['externalId'].isin(ext_to_internal)]

    # join all vector columns into same column and drop other columns
    categories['category'] = categories.iloc[:, 1:].apply(tuple, axis=1)
    categories.drop(categories.columns[1:-1], axis=1, inplace=True)

    # Reindex on external id
    categories['id'] = categories['externalId'].replace(ext_to_internal)
    categories.set_index('id', inplace=True, drop=True)
    categories.reindex()

    # Change category vector to dictionary
    cat_col = []
    for id, row in categories.iterrows():
        cats = {}
        for s in row['category']:
            if type(s) == str:
                (k, v) = str(s).split(':')
                cats[k] = int(v)
        cat_col.append(json.dumps(cats))
    categories['category'] = cat_col

    # Write out category labels
    categories.to_csv(output_dir + '/categories.tsv', sep='\t', index_label='id', columns=['externalId', 'category'])

    # Write out links sample
    if os.path.isfile(input_dir + '/links.tsv'):
        out = open(output_dir + '/links.tsv', 'w')
        for line in open(input_dir + '/links.tsv'):
            if line.startswith('id'): continue  # skip header
            tokens = line.split('\t')
            id = tokens[0]
            if int(id) not in internal_ids: continue
            links = sorted(set(i for i in tokens[1:] if int(i) in internal_ids))
            out.write(id + '\t' + '\t'.join(links) + '\n')

    # Write out example coordinates
    # xy = bh_sne(vecs.as_matrix(), pca_d=None)
    # xy_df = pd.DataFrame(data={'x': xy[:, 0], 'y': xy[:, 1]}, index=vecs.index)
    # xy_df.to_csv(output_dir + '/coords.tsv', sep='\t')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    input_dir = 'data/simple_all'
    make_dataset('data/simple_all', 'data/simple_500', 500)