from glob import glob
import numpy as np
import os
import pdb
import scipy.misc
import time


trainset_path    = "path-to-folder-holding-normal-training-images"
test_normal_path = "path-to-folder-holding-normal-test-images"
test_anom_path   = "path-to-folder-holding-anom-test-images"

def get_nr_training_samples(batch_size):
    files = glob(os.path.join(trainset_path, "*.png"))
    total_nr_samples = len(files)
    nr_training_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)

    return nr_training_samples


def make_generator(data_set, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')

        if data_set == 'train_normal':
            files = glob(os.path.join(trainset_path, "*.png"))
        elif data_set == 'test_normal':
            files = glob(os.path.join(test_normal_path, "*.png"))
        elif data_set == 'test_anom':
            files = glob(os.path.join(test_anom_path, "*.png"))

        assert(len(files) > 0)

        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, f in enumerate(files):
            image = scipy.misc.imread(f, mode='L')
            if np.random.rand()>=0.5:
                image = image[:,::-1]
            images[n % batch_size] = np.expand_dims( image, 0)
            
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def load(batch_size):
    return (
        make_generator('train_normal', batch_size),
        make_generator('test_normal', batch_size)
        make_generator('test_anom', batch_size)
    )


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for n, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if n == 1000:
            break
        t0 = time.time()