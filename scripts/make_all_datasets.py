import os
import shutil
import sys

sys.path.insert(0, '..')

from clumbed.wrangle import make_samples

if __name__ == '__main__':
    # Download the simple all data files if necessary
    if not os.path.isfile('./data/simple_all/links.tsv'):
        import urllib
        import zipfile

        url = "https://www.dropbox.com/s/44c18hpeqqln9xv/simple_all.zip?dl=1"
        print('download and extracting ' + url + '. Please be patient.')
        urllib.urlretrieve(url, './data/simple_all.zip')

        zip_ref = zipfile.ZipFile('./data/simple_all.zip', 'r')
        zip_ref.extractall('./data')
        zip_ref.close()

        os.unlink('./data/simple_all.zip')

    reload(sys)
    sys.setdefaultencoding('utf8')
    input_dir = 'data/simple_all'

    for n in 500, 5000, 50000:
        output_dir = 'data/simple_%d' % n
        print('creating dataset ' +  `output_dir`)
        shutil.rmtree(output_dir, True)
        os.makedirs(output_dir)
        make_samples.make_dataset(input_dir, output_dir, n)
