import os
import shutil
import sys

sys.path.insert(0, '..')

from zalimar.wrangle.augment import augment_everything

from zalimar.wrangle import make_samples

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    input_dir = 'data/simple_all'

    for n in 500, 5000, 50000:
        output_dir = 'data/simple_%d' % n
        print('creating dataset ' +  `output_dir`)
        shutil.rmtree(output_dir, True)
        os.makedirs(output_dir)
        make_samples.make_dataset(input_dir, output_dir, n)
        augment_everything(output_dir)